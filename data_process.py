import shutil
import random
import torch
import os
from torch_geometric.data import Data
import json
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

class Dataset:
    def __init__(self,  name='cora', prefix='./dataset/'):
        self.shot_mask = None
        self.shot_idx = None
        self.graph = None
        self.x_list = None
        self.name = name

        preprocess_filename = f'{prefix}{name}.npz'
        if os.path.exists(preprocess_filename):
            with np.load(preprocess_filename, allow_pickle=True) as f:
                data = f['data'].item()
                feat = f['feat']
                edge_index_np = f['edge_index']
                edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        else:
            data = sio.loadmat(f"{prefix + name}.mat")
            adj = data['Network']
            feat = data['Attributes']
            num_nodes = feat.shape[0]

            adj_sp = sp.csr_matrix(adj)
            row, col = adj_sp.nonzero()
            edge_index_np = np.vstack((row, col))

            if edge_index_np.max() >= num_nodes:
                raise ValueError(f"edge_index max index {edge_index_np.max()} exceeds num_nodes {num_nodes}")

            edge_index = torch.tensor(edge_index_np, dtype=torch.long)

            if name in ['Amazon', 'YelpChi', 'tolokers', 'tfinance']:
                feat = sp.lil_matrix(feat)
                feat = preprocess_features(feat)
            else:
                feat = sp.lil_matrix(feat).toarray()

            feat = torch.FloatTensor(feat)
            np.savez(preprocess_filename, data=data, feat=feat, edge_index=edge_index_np)

        label = data['Label'] if ('Label' in data) else data['gnd']
        self.label = label
        self.feat = feat
        # from sklearn.preprocessing import StandardScaler
        # self.feat = StandardScaler().fit_transform(self.feat)
        ano_labels = torch.tensor(np.squeeze(np.array(self.label)), dtype=torch.float)

        num_positive = int((ano_labels == 1).sum().item())
        print(f"[INFO] Number of positive (label=1) samples: {num_positive}")

        if feat.shape[0] != ano_labels.shape[0]:
            raise ValueError(f"Feature nodes {feat.shape[0]} do not match label nodes {ano_labels.shape[0]}")

        data = Data(
            x=torch.tensor(self.feat, dtype=torch.float),
            edge_index=edge_index,
            y=ano_labels
        )
        self.graph = data

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

