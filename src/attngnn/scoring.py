"""Anomaly scores and attention-derived influence scores.

Anomaly scores (all per node, higher = more anomalous)
  attr_err     ||x_i - x_hat_i||_2                                        (attribute reconstruction)
  struct_err   mean_{j in N(i)} (1 - s_ij)^2 + mean_{k in K_i} s_ik^2      with s = sigmoid(z_i . z_j)
               (positives = real neighbours, K_i = ``neg_per_node`` uniformly sampled non-edges, fixed seed)
  rec          alpha * z(attr_err) + (1 - alpha) * z(struct_err)           (DOMINANT-style combination)
  att_only     sum_{j in N(i)} a~_ji * ||x_i - x_j||_2 / sqrt(d)          (attention-weighted feature
               discrepancy; a~ = head- and layer-averaged GAT coefficient renormalised over the non-self
               in-neighbours of i;  uses NO decoder output)
  feat_dev_uniform  same with a~ = 1/deg_i                                  (non-attentive control)
  combined     0.5 * z(rec) + 0.5 * z(att_only)                            (weights predeclared, not tuned)
z(.) = robust standardisation (median / 1.4826*MAD) with statistics from TRAIN nodes only (no labels).

Influence scores (per node j)
  recv_att     R_j = sum over arcs j->i (i != j) of the head/layer-averaged coefficient alpha_{j->i}
               = attention that source j receives from its neighbouring targets.  (Summing alpha over the
               arcs *arriving* at a target is identically 1 and therefore useless as a ranking.)
  recv_att_adj residual of log R_j after an OLS fit on log(1+deg_j) over TRAIN nodes (degree-adjusted)
  recv_att_per_deg  R_j / deg_j
  model_pred   influence head output (one column per cascade probability)
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def attention_edges(model, x, ei):
    """Per-arc attention for the *input* arcs (self-loops dropped).

    Returns dict(src, dst, alpha[E, L, H]) or None for non-attentive encoders.  ``alpha`` are the raw GAT
    coefficients (softmax over the incoming arcs of each target including its self loop), eval mode.
    """
    model.eval()
    out = model(x, ei, return_attention=True)
    if not out["attn"]:
        return None
    layers = []
    for ei2, alpha in out["attn"]:
        keep = ei2[0] != ei2[1]
        assert torch.equal(ei2[:, keep], ei), "attention arcs not aligned with input edge order"
        layers.append(alpha[keep])
    alpha = torch.stack(layers, 1)  # [E, L, H]
    return {"src": ei[0].cpu().numpy(), "dst": ei[1].cpu().numpy(), "alpha": alpha.cpu().numpy()}


def mean_alpha(att, layers=None, heads=None):
    a = att["alpha"]
    if layers is not None:
        a = a[:, layers:layers + 1] if isinstance(layers, int) else a[:, layers]
    if heads is not None:
        a = a[:, :, heads:heads + 1] if isinstance(heads, int) else a[:, :, heads]
    return a.mean((1, 2))


def robust_z(s, ref_idx):
    ref = s[ref_idx]
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    if mad < 1e-12:
        mad = ref.std() if ref.std() > 1e-12 else 1.0
    return (s - med) / mad


@torch.no_grad()
def recon_errors(model, x, ei, seed=0, neg_per_node=20):
    model.eval()
    n = x.shape[0]
    out = model(x, ei)
    attr_err = (x - out["xhat"]).norm(dim=1)
    z = out["z"]
    s = torch.sigmoid(model.struct_logits(z, ei[0], ei[1]))
    pos_sum = torch.zeros(n).index_add_(0, ei[0], (1 - s) ** 2)
    deg = torch.zeros(n).index_add_(0, ei[0], torch.ones(ei.shape[1])).clamp(min=1)
    gen = torch.Generator().manual_seed(seed)
    neg = torch.randint(0, n, (n, neg_per_node), generator=gen)
    sn = torch.sigmoid((z.unsqueeze(1) * z[neg]).sum(-1))
    struct_err = pos_sum / deg + (sn ** 2).mean(1)
    return attr_err.numpy(), struct_err.numpy()


@torch.no_grad()
def feature_discrepancy(x, ei, w, chunk=100_000):
    """Weighted mean over in-neighbours of ||x_i - x_j|| / sqrt(d).  ``w`` = per-arc weights (raw)."""
    n, d = x.shape
    src, dst = ei[0], ei[1]
    dist = torch.empty(ei.shape[1])
    for a in range(0, ei.shape[1], chunk):
        b = min(a + chunk, ei.shape[1])
        dist[a:b] = (x[src[a:b]] - x[dst[a:b]]).norm(dim=1) / np.sqrt(d)
    w = torch.as_tensor(w, dtype=torch.float32)
    den = torch.zeros(n).index_add_(0, dst, w)
    num = torch.zeros(n).index_add_(0, dst, w * dist)
    return (num / den.clamp(min=1e-12)).numpy(), den.numpy()


def attention_only_score(x, ei, att):
    s, _ = feature_discrepancy(x, ei, mean_alpha(att))
    return s


def uniform_discrepancy(x, ei):
    s, _ = feature_discrepancy(x, ei, np.ones(ei.shape[1], np.float32))
    return s


def received_attention(att, n):
    a = mean_alpha(att)
    return np.bincount(att["src"], weights=a, minlength=n)


def degree_adjust(R, deg, train_idx, eps=1e-12):
    lr = np.log(R + eps)
    ld = np.log1p(deg)
    X = np.stack([np.ones(len(train_idx)), ld[train_idx]], 1)
    beta, *_ = np.linalg.lstsq(X, lr[train_idx], rcond=None)
    return lr - (beta[0] + beta[1] * ld), beta.tolist()


def all_anomaly_scores(model, x, ei, train_idx, alpha, seed=0, neg_per_node=20):
    """Returns dict name -> score array (numpy)."""
    ae, se = recon_errors(model, x, ei, seed, neg_per_node)
    za, zs = robust_z(ae, train_idx), robust_z(se, train_idx)
    rec = alpha * za + (1 - alpha) * zs
    sc = {"attr_err": ae, "struct_err": se, "rec": rec}
    att = attention_edges(model, x, ei)
    sc["feat_dev_uniform"] = uniform_discrepancy(x, ei)
    if att is not None:
        ao = attention_only_score(x, ei, att)
        sc["att_only"] = ao
        sc["combined"] = 0.5 * robust_z(rec, train_idx) + 0.5 * robust_z(ao, train_idx)
    return sc


def all_influence_scores(model, x, ei, train_idx, p_grid):
    n = x.shape[0]
    deg = np.bincount(ei[0].numpy(), minlength=n).astype(float)
    with torch.no_grad():
        pred = model(x, ei)["inf"].numpy()
    sc = {}
    for j, p in enumerate(p_grid):
        sc[f"model_pred@{p}"] = pred[:, j]
    att = attention_edges(model, x, ei)
    info = {}
    if att is not None:
        R = received_attention(att, n)
        adj, beta = degree_adjust(R, deg, train_idx)
        sc["recv_att"] = R
        sc["recv_att_adj"] = adj
        sc["recv_att_per_deg"] = R / np.maximum(deg, 1)
        info["degree_adjust_beta"] = beta
    return sc, info
