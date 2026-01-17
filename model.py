import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from utils import *


def info_NCE_loss(embeddings, neighbor_embeddings, num_nodes, mask=None, temperature=0.5, num_negatives=3):
    """基于预计算邻居嵌入的 InfoNCE 对比损失"""
    import logging

    if embeddings.shape[0] != num_nodes or neighbor_embeddings.shape[0] != num_nodes:
        raise ValueError(f"Embeddings size {embeddings.shape[0]} or neighbor_embeddings size {neighbor_embeddings.shape[0]} does not match num_nodes {num_nodes}")
    if mask is not None and mask.shape[0] != num_nodes:
        raise ValueError(f"Mask size {mask.shape[0]} does not match num_nodes {num_nodes}")

    z = F.normalize(embeddings, dim=-1)
    z_pos = F.normalize(neighbor_embeddings, dim=-1)

    idx = torch.arange(num_nodes, device=z.device)

    # 采 K 个负样本，并确保不等于自身（简单重采）
    K = num_negatives
    neg_idx = torch.randint(0, num_nodes, (num_nodes, K), device=z.device)
    for _ in range(3):
        same = (neg_idx == idx.unsqueeze(1))
        if not same.any():
            break
        neg_idx[same] = torch.randint(0, num_nodes, (same.sum().item(),), device=z.device)
    z_neg = F.normalize(z_pos[neg_idx], dim=-1)  # [N, K, D]

    if mask is not None:
        z = z[mask]
        z_pos = z_pos[mask]
        z_neg = z_neg[mask]

    # 正样本相似度
    sim_pos = (z * z_pos).sum(dim=-1) / temperature           # [N]

    # 负样本相似度（对 K 个取 logsumexp）
    sim_neg = (z.unsqueeze(1) * z_neg).sum(dim=-1) / temperature   # [N, K]

    # logits 包含正与负：log(exp(sim_pos) + sum exp(sim_neg)))
    # 等价于：-sim_pos + logsumexp([sim_pos, sim_neg...])
    max_neg = torch.logsumexp(sim_neg, dim=-1)                 # [N]
    logits = torch.stack([sim_pos, max_neg], dim=-1)           # [N, 2]
    loss = -sim_pos + torch.logsumexp(torch.cat([sim_pos.unsqueeze(-1), sim_neg], dim=-1), dim=-1)

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        logging.warning("NaN/Inf in InfoNCE loss")
        raise ValueError("Invalid InfoNCE loss")

    return loss.mean()

def contra_anomaly_score(embeddings, neighbor_embeddings):
    """计算异常分数：1 - 节点嵌入与一阶邻居嵌入平均值的相似性，并应用softmax"""
    embeddings = F.normalize(embeddings, dim=-1)
    neighbor_embeddings = F.normalize(neighbor_embeddings, dim=-1)
    similarity = (embeddings * neighbor_embeddings).sum(dim=-1)
    similarity = (1 + similarity) / 2
    anomaly_scores = 1 - similarity
    # anomaly_scores = torch.sigmoid(anomaly_scores)
    # 应用softmax归一化
    try:
        # anomaly_scores = F.softmax(anomaly_scores, dim=0)
        if torch.isnan(anomaly_scores).any() or torch.isinf(anomaly_scores).any():
            logging.warning("NaN or Inf detected in anomaly scores after softmax")
            raise ValueError("Invalid anomaly scores")
        return anomaly_scores
    except Exception as e:
        logging.error(f"Error in compute_anomaly_score: {e}")
        raise e

from torch_geometric.utils import dense_to_sparse, to_dense_adj


def reconstruction_loss(original_x, reconstructed_x, edge_index, adj_logits,
                        mask=None, lambda_adj=0.2, num_neg_per_pos=0, num_nodes=None):
    """
    original_x: [N, F]
    reconstructed_x: [N, F]
    edge_index: [2, E] （无向图建议包含双向）
    adj_logits: [N, N] （decode 输出的 logits）
    mask: [N] or None （bool）
    lambda_adj: 特征/邻接损失的权重
    num_neg_per_pos: 每条正边采样的负边数
    num_nodes: N（若 original_x.shape[0] 不等于全图节点数，可显式传入）
    """
    if num_nodes is None:
        num_nodes = original_x.size(0)

    device = original_x.device
    if mask is not None:
        # 统一到 bool & 同设备
        mask = mask.to(device=device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

    # -------- 特征损失（MSE） --------
    x_true = original_x if mask is None else original_x[mask]
    x_pred = reconstructed_x if mask is None else reconstructed_x[mask]
    feature_loss = F.mse_loss(x_pred, x_true)

    # -------- 邻接 BCE（正/负） --------
    row, col = edge_index.to(device)
    # 去除自环（如果不希望训练它们）
    keep = (row != col)
    row, col = row[keep], col[keep]

    # 正样本 logits / targets
    pos_logits = adj_logits[row, col]
    pos_targets = torch.ones_like(pos_logits, device=device)

    # 负样本采样：对每条正边，随机采样 num_neg_per_pos 条“非边”
    # 简单起见：随机挑 j，若 (i,j) 是边或 i==j 则重采（期望常数次命中）
    if num_neg_per_pos > 0:
        i_rep = row.repeat_interleave(num_neg_per_pos)
        j_rand = torch.randint(0, num_nodes, (i_rep.size(0),), device=device)

        # 避免自环
        mask_not_self = (i_rep != j_rand)

        # 为了避免抽到正边，可用一个哈希集合；但为 O(1) 的近似，我们只过滤掉“高概率”边：
        # 这里选择简单方案：将 (i,j) 与 (j,i) 是否在 edge_index 中的判断替换为再次重采，直到通过。
        # 为防止极端情况，限制尝试次数。
        max_tries = 5
        tries = 0
        adj_pos_flag = torch.ones_like(mask_not_self, dtype=torch.bool, device=device)
        while tries < max_tries:
            # 估计正边：使用稀疏张量快速判断（更稳的是预建 set，但 Python set 在 GPU 不可用）
            # 简化：只过滤自环；正负边轻微污染对稳定训练影响不大
            ok = mask_not_self
            if ok.all():
                break
            need = (~ok).nonzero(as_tuple=False).squeeze(-1)
            j_rand[need] = torch.randint(0, num_nodes, (need.size(0),), device=device)
            mask_not_self = (i_rep != j_rand)
            tries += 1

        neg_logits = adj_logits[i_rep, j_rand]
        neg_targets = torch.zeros_like(neg_logits, device=device)

        # 合并
        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_targets = torch.cat([pos_targets, neg_targets], dim=0)
    else:
        all_logits = pos_logits
        all_targets = pos_targets

    adj_loss = F.binary_cross_entropy_with_logits(all_logits, all_targets)

    # -------- 异常分数（节点级） --------
    # 用“邻边误差 + 采样非边误差”的局部近似，避免 O(N^2)。
    with torch.no_grad():
        # 概率用于误差
        adj_prob = torch.sigmoid(adj_logits)

        # 每个节点的正边误差：|p_ij - 1|
        pos_err = torch.abs(adj_prob[row, col] - 1.0)
        node_pos_err = torch.zeros(num_nodes, device=device).index_add_(0, row, pos_err)
        deg = torch.bincount(row, minlength=num_nodes).clamp_min(1)
        node_pos_err = node_pos_err / deg

        # 采样非边误差：|p_ij - 0| = p_ij
        # 重用上面的 i_rep, j_rand；如果 num_neg_per_pos==0，就给一个退化近似
        if num_neg_per_pos > 0:
            neg_err = adj_prob[i_rep, j_rand]
            node_neg_err = torch.zeros(num_nodes, device=device).index_add_(0, i_rep, neg_err)
            deg_neg = torch.bincount(i_rep, minlength=num_nodes).clamp_min(1)
            node_neg_err = node_neg_err / deg_neg
        else:
            # 没有负采样时，给个 0 作为占位，不引入偏差
            node_neg_err = torch.zeros(num_nodes, device=device)

        node_adj_error = 0.5 * (node_pos_err + node_neg_err)

        # 与特征误差做同尺度归一化
        # feature_error 取样本级 MSE，再归一化到 [0,1]
        with torch.no_grad():
            sample_feat_err = F.mse_loss(reconstructed_x, original_x, reduction='none').mean(dim=1)  # [N]
            fe_min, fe_max = sample_feat_err.min(), sample_feat_err.max()
            feat_err_nrm = (sample_feat_err - fe_min) / (fe_max - fe_min + 1e-8)

            ae_min, ae_max = node_adj_error.min(), node_adj_error.max()
            adj_err_nrm = (node_adj_error - ae_min) / (ae_max - ae_min + 1e-8)

            anomaly_scores = (1 - lambda_adj) * feat_err_nrm + lambda_adj * adj_err_nrm
            # 不再额外 sigmoid，避免压缩动态范围
            if torch.isnan(anomaly_scores).any() or torch.isinf(anomaly_scores).any():
                raise ValueError("Invalid anomaly scores")

    total_loss = (1 - lambda_adj) * feature_loss + lambda_adj * adj_loss
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
        raise ValueError("Invalid loss value")

    return anomaly_scores, total_loss



class MeanAggregator(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, x, edge_index):
        # 不加自环
        return self.propagate(edge_index, x=x)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            activation
        ])

    def forward(self, features):
        h = features
        for layer in self.encoder:
            h = layer(h)
        h = F.normalize(h, p=2, dim=1)  # row normalize
        return h

class Contrastive_model(nn.Module):
    def __init__(self, in_dim,  out_dim=32, drop_out = 0., activation = nn.PReLU(),temperature=0.5):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: GCN 隐藏层维度
            out_dim: 输出嵌入维度
            temperature: 对比损失温度
        """
        super().__init__()
        self.encoder = MLP(in_dim, out_dim, activation)
        self.neigh = MeanAggregator()
        self.temperature = temperature

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.encoder(x)
        x_neigh = self.neigh(x, edge_index)
        return x, x_neigh

    def _loss(self, embeddings, neighbor_embeddings, mask=None):
        """
        对比学习损失，使用 InfoNCE
        """
        num_nodes = embeddings.size(0)

        loss = info_NCE_loss(embeddings, neighbor_embeddings, num_nodes, mask, temperature=self.temperature)
        return loss

    def anomaly_score(self, embeddings, neighbor_embeddings):
        """
        计算节点的异常分数
        """
        scores = contra_anomaly_score(embeddings, neighbor_embeddings)
        return scores

class FullAttention(nn.Module):

    def __init__(self, dim, scale=None, attention_dropout=0.0, qk_dim=32):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.q_fc = nn.Linear(dim, qk_dim)
        self.k_fc = nn.Linear(dim, qk_dim)
        self.v_fc = nn.Linear(dim, dim)
        self.att_weights = None

    def forward(self, queries, keys, values):
        N, K, D = keys.shape
        queries = self.q_fc(queries)  # (n_q,32)

        query_global = queries.mean(dim=0, keepdim=True)  # [1, qk_dim]
        queries = query_global.repeat(N, 1)  # [n_k, qk_dim]

        keys = self.k_fc(keys)  # (n_k,3,32)
        values = self.v_fc(values)  # (n_k,3,128)
        scale = self.scale or 1.0 / math.sqrt(D)
        queries = queries.unsqueeze(1)
        keys = keys.transpose(1, 2)
        scores = queries @ keys  # [n,1,k+1]
        A = scale * scores
        A = self.dropout(torch.softmax(A, dim=-1))

        V = A @ values
        return V.contiguous()

class SaConv(nn.Module):
    def __init__(self, in_feats, out_feats, k, qk_dim=32, dropout=0.5):
        super(SaConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.dropout = dropout
        self.att = FullAttention(out_feats, qk_dim=qk_dim)

    def unnLaplacian(self, x, D_invsqrt, edge_index):
        """High pass: x - D^-1/2 A D^-1/2 x"""
        # 消息传递
        h = x * D_invsqrt
        row, col = edge_index
        m = h[row]  # 按边发送信息
        out = scatter_add(m, col, dim=0, dim_size=x.size(0))
        return x - out * D_invsqrt

    def A_tile(self, x, D_invsqrt, edge_index):
        """Low pass: 2I - L^sym"""
        h = x * D_invsqrt
        row, col = edge_index
        m = h[row]
        out = scatter_add(m, col, dim=0, dim_size=x.size(0))
        L_sym = x - out * D_invsqrt
        return 2 * x - L_sym


    def forward(self, train_data, h_train, h_ori ):
        x, edge_index = h_train, train_data.edge_index


        # 度的逆平方根
        deg = degree(edge_index[0], num_nodes=x.size(0)).clamp(min=1)
        D_invsqrt = deg.pow(-0.5).unsqueeze(-1).to(x.device)

        L_stack =[]
        # Low-pass 特征
        a_feat = self.A_tile(x, D_invsqrt, edge_index)
        L_stack = [a_feat]

        # High-pass 多阶卷积
        for _ in range(self._k):
            x = self.unnLaplacian(x, D_invsqrt, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            L_stack.append(x)

        # 堆叠成 [num_nodes, k+1, out_feats]
        L_stack = torch.stack(L_stack, dim=0).transpose(0, 1)  # [n, k, d]

        # Self-Attention
        h = self.att(h_ori, L_stack, L_stack)
        h = torch.sum(h, dim=1)  # 聚合 k 维

        return h

class Reconstruct_model(nn.Module):
    def __init__(self, in_dim, hidden_dim = 64, latent_dim = 64, dropout = 0.1):
        """
        :param in_channels: 输入特征维度
        :param hidden_channels: 第一层GCN的隐藏维度
        :param latent_dim: 第二层GCN（编码后）的维度
        :param out_channels: 解码后输出特征维度（用于重构）
        """
        super(Reconstruct_model, self).__init__()
        self.act = nn.ReLU()
        self.dropout = dropout
        self.latent_dim = hidden_dim
        # 编码器
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, latent_dim)

        # 混合频率加权
        self.linear = nn.Linear(in_dim, hidden_dim )
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv = SaConv(hidden_dim, hidden_dim , 2, dropout=self.dropout)

        # 解码器 (MLP)
        self.decoder_x = nn.Linear(hidden_dim, in_dim)  # 重构节点特征
        self.decoder_adj = nn.Linear(hidden_dim, hidden_dim)  # 输出邻接矩阵 logit 的向量形式


    def encode(self, train_data, original_data, mask):
        x_ori = original_data.x
        h_ori = self.linear(x_ori)
        h_ori = self.act(h_ori)
        h_ori = self.linear2(h_ori)
        h_ori = self.act(h_ori)

        x_train = train_data.x
        h_train = self.linear(x_train)
        h_train = self.act(h_train)
        h_train = self.linear2(h_train)
        h_train = self.act(h_train)

        h = self.conv(train_data, h_train, h_ori[~mask])
        return h

    def decode(self, z):
        reconstruct_x = self.decoder_x(z)
        reconstruct_adj =  z @ z.T
        # reconstruct_adj = torch.matmul(z, z.t())
        return reconstruct_x, reconstruct_adj


    def forward(self, train_data, original_data, mask):
        z = self.encode(train_data, original_data, mask)
        x_hat, adj_hat = self.decode(z)

        return x_hat, adj_hat

    def _loss(self, x_true, x_hat, adj_true, adj_hat, mask = None ):
        anomaly_scores, loss = reconstruction_loss(x_true, x_hat, adj_true, adj_hat, mask)
        return anomaly_scores, loss


class ReconstructGCN(nn.Module):
    """
    GCN-based reconstruction model.
    Interface matches your current Reconstruct_model:
      - forward(train_data, original_data, mask) -> (x_hat, adj_hat)
      - _loss(x_true, x_hat, adj_true, adj_hat, mask=None) -> (anomaly_scores, loss)
    """
    def __init__(self, in_dim, hidden_dim=64, latent_dim=64, dropout=0.1,
                 lambda_adj=0.2, num_neg_per_pos=0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        # ===== Encoder (GCN) =====
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, latent_dim)

        # ===== Decoder =====
        # 特征重构：latent -> hidden -> in_dim（比单层 Linear 稍微强一点）
        self.dec_x1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_x2 = nn.Linear(hidden_dim, in_dim)

        # 邻接重构：inner-product (logits) = z @ z.T
        # （与你原始实现一致，不额外加 decoder_adj）

        # reconstruction_loss 内部用到的超参（可选保持一致）
        self.lambda_adj = lambda_adj
        self.num_neg_per_pos = num_neg_per_pos

        self.act = nn.ReLU()

    def encode(self, train_data, original_data=None, mask=None):
        """
        train_data: torch_geometric.data.Data
        original_data/mask: 为了保持接口一致，GCN版本不强依赖它们
        """
        x = train_data.x
        edge_index = train_data.edge_index

        h = self.gcn1(x, edge_index)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        z = self.gcn2(h, edge_index)
        # z 是否加激活一般都行；这里给一个轻量激活+dropout
        z = self.act(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return z

    def decode(self, z):
        # 重构特征
        x_hat = self.dec_x1(z)
        x_hat = self.act(x_hat)
        x_hat = self.dec_x2(x_hat)

        # 重构邻接 logits（inner-product）
        adj_hat = z @ z.T
        return x_hat, adj_hat

    def forward(self, train_data, original_data=None, mask=None):
        z = self.encode(train_data, original_data, mask)
        x_hat, adj_hat = self.decode(z)
        return x_hat, adj_hat

    def _loss(self, x_true, x_hat, adj_true, adj_hat, mask=None):
        """
        直接复用你已有的 reconstruction_loss，保证 anomaly_scores 的含义一致。
        """
        anomaly_scores, loss = reconstruction_loss(
            original_x=x_true,
            reconstructed_x=x_hat,
            edge_index=adj_true,
            adj_logits=adj_hat,
            mask=mask,
            lambda_adj=self.lambda_adj,
            num_neg_per_pos=self.num_neg_per_pos,
            num_nodes=x_true.size(0)
        )
        return anomaly_scores, loss


