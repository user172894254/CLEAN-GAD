import shutil
import random
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
import logging

def rank_data(x):
    """
    用 PyTorch 实现秩排序，返回排名（从1开始）
    x: 1D tensor
    """
    sorted_vals, sorted_idx = torch.sort(x)
    ranks = torch.empty_like(sorted_idx)
    ranks[sorted_idx] = torch.arange(1, len(x)+1, device=x.device)
    return ranks

def fit_gaussian_1d(x, unbiased=False):
    """
    Fit 1D Gaussian parameters (mean, variance) from samples x.
    x: 1D tensor (any device)
    unbiased: if True use ddof=1 (divide by n-1); else use MLE (divide by n)
    Returns (mu, var) as float64 tensors on same device.
    """
    x = x.to(dtype=torch.float64)
    n = x.numel()
    if n == 0:
        raise ValueError("Empty sample")
    mu = x.mean()
    if unbiased and n > 1:
        var = x.var(unbiased=True)  # divides by (n-1)
    else:
        # MLE variance: sum((x-mu)^2)/n
        var = ((x - mu)**2).sum() / n
    return mu, var

def kl_normal_1d(mu_p, var_p, mu_q, var_q, eps=1e-12):
    """
    Compute KL(N(mu_p, var_p) || N(mu_q, var_q)) for 1D Gaussians.
    All inputs can be float or 0-dim tensors. Returns a scalar tensor (float64).
    Numerical stability: var_* are clamped by eps.
    """
    # ensure double precision
    mu_p = mu_p.to(dtype=torch.float64)
    mu_q = mu_q.to(dtype=torch.float64)
    var_p = var_p.to(dtype=torch.float64)
    var_q = var_q.to(dtype=torch.float64)

    var_p = var_p.clamp_min(eps)
    var_q = var_q.clamp_min(eps)

    term1 = var_p / var_q
    term2 = ((mu_p - mu_q)**2) / var_q
    log_term = torch.log(var_q / var_p)
    kl = 0.5 * (term1 + term2 - 1.0 + log_term)
    return kl

def symmetric_kl_normal_1d_from_samples(x_p, x_q, unbiased=False, eps=1e-12):
    """
    Fit Gaussians to samples x_p (normal) and x_q (anomaly), then compute
    both KL(P||Q), KL(Q||P), and symmetric KL.
    x_p, x_q: 1D tensors (can be on GPU)
    Returns dict with 'kl_pq', 'kl_qp', 'symmetric_kl' (all float scalars).
    """
    mu_p, var_p = fit_gaussian_1d(x_p, unbiased=unbiased)
    mu_q, var_q = fit_gaussian_1d(x_q, unbiased=unbiased)

    kl_pq = kl_normal_1d(mu_p, var_p, mu_q, var_q, eps=eps)
    kl_qp = kl_normal_1d(mu_q, var_q, mu_p, var_p, eps=eps)
    sym = (kl_pq + kl_qp)
    # return as float tensors (or .item() if you want python floats)
    # return {'kl_pq': kl_pq, 'kl_qp': kl_qp, 'symmetric_kl': sym}
    return sym



def idx_sample(idxes, num_nodes):
    """生成随机负样本索引，避免选择自身节点"""
    num_idx = len(idxes)
    if num_idx != num_nodes:
        raise ValueError(f"idxes length {num_idx} does not match num_nodes {num_nodes}")

    random_add = torch.randint(low=1, high=num_idx, size=(num_idx,), device=idxes.device)
    idx = torch.arange(0, num_idx, device=idxes.device)
    shuffled_idx = torch.remainder(idx + random_add, num_idx)
    # shuffled_idx = torch.where(shuffled_idx == idx, (shuffled_idx + 1) % num_idx, shuffled_idx)
    return shuffled_idx


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def test_eval(anomaly_scores, labels, mask=None):
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if mask is not None:
            anomaly_scores = anomaly_scores[mask]
            labels = labels[mask.cpu().numpy()]
        anomaly_scores = anomaly_scores.cpu().numpy()
        score['AUROC'] = roc_auc_score(labels, anomaly_scores)
        score['AUPRC'] = average_precision_score(labels, anomaly_scores)
    return score


def data_split(train_portion, anomaly_scores, normal_only=True):
    """
    分割数据集，可选择仅包含正常节点

    Args:
        args: 命令行参数
        data: 图数据对象(包含ano_labels)
        normal_only: 是否仅选择正常节点作为验证集
    Returns:
        val_mask: 验证集掩码(仅包含正常节点)
    """
    top_ratio = train_portion
    num_nodes = anomaly_scores.shape[0]
    if not (0 < top_ratio <= 1):
        raise ValueError("top_ratio 必须在 (0, 1] 区间内")

    num_to_select = max(1, int(num_nodes * top_ratio))

    # 获取分数排序的索引（从大到小）
    sorted_indices = torch.argsort(-anomaly_scores)

    # 选前 num_to_select 个索引
    selected_indices = sorted_indices[:num_to_select]

    # 构造 mask
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=anomaly_scores.device)
    mask[selected_indices] = True

    return mask


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))
