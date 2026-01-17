import os
import sys
import time
import glob
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import logging
import utils
from model import *
from data_process import *
from torch_geometric.utils import subgraph
import time



def main(args):
    # Check for GPU availability
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    # Set random seeds and CUDA configurations
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('GPU device = %d' % args.gpu)
    logging.info("args = %s", args)

    # Load dataset using Dataset
    try:
        dataset = Dataset(name=args.dataset, prefix=args.data)
        data = dataset.graph.cuda()
        logging.info(f"Loaded dataset: {dataset.name}, num_nodes: {data.num_nodes}, num_edges: {data.edge_index.shape[1]}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)


    model = Contrastive_model(in_dim=data.x.shape[1], out_dim=args.con_hidden_dims).cuda()
    logging.info("Parameter size = %fMB", utils.count_parameters_in_MB(model))

    # Define optimizer and scheduler
    con_optimizer = torch.optim.SGD(
        model.parameters(),
        lr = args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(con_optimizer, args.con_epochs, eta_min=args.learning_rate_min)


    # Training loop
    for epoch in range(args.con_epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch %d lr %e', epoch, lr)

        # Train
        anomaly_score, score_train = con_train(data, model, con_optimizer)
        # logging.info('Train AUROC: %.4f, AUPRC: %.4f', score_train['AUROC'], score_train['AUPRC'])



    Recon_model = Reconstruct_model(in_dim=data.x.shape[1], hidden_dim=args.Recon_hidden_dims, latent_dim=args.Recon_hidden_dims).cuda()
    logging.info("Parameter size = %fMB", utils.count_parameters_in_MB(Recon_model))

    # Define optimizer and scheduler
    Recon_optimizer = torch.optim.SGD(
        Recon_model.parameters(),
        args.Recon_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Recon_optimizer, args.Recon_epochs,eta_min=args.learning_rate_min)
    # train_mask = None
    # data_train =data
    con_anomaly_score = anomaly_score
    train_mask = data_split(args.train_portion, anomaly_score)
    train_mask = ~train_mask
    keep_nodes = train_mask.nonzero(as_tuple=True)[0]

    
    # 生成子图
    edge_index, edge_attr = subgraph(
        keep_nodes, data.edge_index, relabel_nodes=True
    )

    # 构建新的数据对象
    data_train = Data(
        x=data.x[keep_nodes],
        edge_index=edge_index,
        y=data.y[keep_nodes]  # 仅保留对应标签
    )

    # Training loop
    for epoch in range(args.Recon_epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch %d lr %e', epoch, lr)

        # Train
        anomaly_score, score_train = Recon_train(data_train, data, Recon_model, Recon_optimizer, train_mask)
        # anomaly_score, score_train = Recon_train(data, data, Recon_model, Recon_optimizer, train_mask)


    alpha = 0.3
    Recon_anomaly_score = anomaly_score
    # con_anomaly_score = utils.rank_data(con_anomaly_score)  # stage1 scores
    # Recon_anomaly_score = utils.rank_data(Recon_anomaly_score)  # stage2 scores
    con_anomaly_score = (con_anomaly_score - con_anomaly_score.min())/(con_anomaly_score.max() - con_anomaly_score.min())
    Recon_anomaly_score = (Recon_anomaly_score - Recon_anomaly_score.min()) / (Recon_anomaly_score.max() - Recon_anomaly_score.min())

    anomaly_score = alpha * con_anomaly_score + (1 - alpha) * Recon_anomaly_score
    logging.info("异常分数： con_anomaly_score:%.4f     Recon_anomaly_score:%.4f", con_anomaly_score, Recon_anomaly_score)
    score = test_eval(anomaly_score, data.y)
    logging.info('Valid AUROC: %.4f, AUPRC: %.4f',  score['AUROC'], score['AUPRC'])


def con_train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()

    try:
        # Forward pass
        embeddings, neighbor_embeddings = model(data)
        loss = model._loss(embeddings, neighbor_embeddings)
        anomaly_score = model.anomaly_score(embeddings, neighbor_embeddings)
        # Evaluate
        score = utils.test_eval(anomaly_score, data.y)
        logging.info('Con_Train loss: %.4f, AUROC: %.4f, AUPRC: %.4f', loss.item(), score['AUROC'], score['AUPRC'])
        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        return anomaly_score, score
    except Exception as e:
        logging.error(f"Con_Training error: {e}")
        raise e

def Recon_train(data_train, data_valid, model, optimizer, mask):
    model.train()
    optimizer.zero_grad()

    try:
        # Forward pass
        reconstruct_x, reconstruct_adj = model(data_train, data_valid, mask)
        n_anomaly_score, loss = model._loss(data_train.x, reconstruct_x, data_train.edge_index, reconstruct_adj)

        reconstruct_x, reconstruct_adj = model(data_valid, data_valid, mask)
        a_anomaly_score, loss1 = model._loss(data_valid.x, reconstruct_x, data_valid.edge_index, reconstruct_adj, mask)

        align_loss = torch.norm(n_anomaly_score - a_anomaly_score[mask], dim=0)

        valid_mask = data_split(0.01, a_anomaly_score[~mask])
        KL = 0.2 - utils.symmetric_kl_normal_1d_from_samples(n_anomaly_score, a_anomaly_score[~mask][valid_mask])

        # KL =  0.1 - a_anomaly_score[~mask][valid_mask].mean() + n_anomaly_score.mean()
        KL = torch.clamp_min(KL, 0.0)
        logging.info("KL损失：%.4f", KL)
        loss = loss + loss1  + KL +  0.7 *  align_loss
        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Evaluate
        reconstruct_x, reconstruct_adj = model(data_valid, data_valid, mask)
        anomaly_score, loss = model._loss(data_valid.x, reconstruct_x, data_valid.edge_index, reconstruct_adj, mask)
        score = utils.test_eval(anomaly_score, data_valid.y)
        logging.info('Recon_Valid loss: %.4f, AUROC: %.4f, AUPRC: %.4f', loss.item(), score['AUROC'], score['AUPRC'])
        return anomaly_score, score
    except Exception as e:
        logging.error(f"Recon_Training error: {e}")
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./dataset/', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='BlogCatalog',
                        help="supported dataset: [cora, Amazon, "
                             "Flickr, weibo, Reddit, BlogCatalog, Facebook, "
                             "]. Default: cora")
    parser.add_argument('--batch_size', type=int, default=0, help='batch size (set to 1 for Cora)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--Recon_learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='最小学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--con_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--Recon_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--con_hidden_dims', type=int, default=32, help='number of dimension')
    parser.add_argument('--Recon_hidden_dims', type=int, default=64, help='number of dimension')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value')
    parser.add_argument('--train_portion', type=float, default=0.01, help='k_ano')
    args = parser.parse_args()


    # Configure logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    main(args)
