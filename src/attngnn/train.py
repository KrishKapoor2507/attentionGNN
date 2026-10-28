"""Training loop: joint (reconstruction + influence) objective, full-batch, CPU friendly.

Total loss  L = w_rec * ( alpha * L_attr + (1 - alpha) * L_struct )  +  lam_inf * L_inf
  L_attr   : mean squared error of x_hat vs x, divided by mean(x^2) so it is O(1) on any scale
  L_struct : binary cross-entropy of sigmoid(z_i.z_j) on a random subset of positive arcs (label 1)
             and ``neg_ratio`` x as many uniformly sampled node pairs (label 0)  (no dense N x N matrix)
  L_inf    : MSE between the influence head and the standardised log1p(spread) of the TRAIN nodes
Variants: anomaly-only (lam_inf = 0), influence-only (w_rec = 0), joint, untrained (epochs = 0).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_pairs(ei: torch.Tensor, n: int, max_pos: int, neg_ratio: float, gen: torch.Generator):
    """Random positive arcs and uniformly sampled negative pairs. Returns src, dst, label."""
    E = ei.shape[1]
    if E > max_pos:
        sel = torch.randperm(E, generator=gen)[:max_pos]
        pos = ei[:, sel]
    else:
        pos = ei
    nneg = int(pos.shape[1] * neg_ratio)
    ns = torch.randint(0, n, (nneg,), generator=gen)
    nd = torch.randint(0, n, (nneg,), generator=gen)
    src = torch.cat([pos[0], ns])
    dst = torch.cat([pos[1], nd])
    lab = torch.cat([torch.ones(pos.shape[1]), torch.zeros(nneg)])
    return src, dst, lab


def loss_terms(model, x, ei, train_idx, inf_target, alpha, gen, max_pos=200_000, neg_ratio=1.0, x_ms=None,
               edge_keep=1.0):
    """``edge_keep`` < 1 randomly drops message-passing arcs (DropEdge-style) for speed/regularisation;
    structure-loss positives are always sampled from the FULL arc list."""
    if edge_keep < 1.0:
        mp_ei = ei[:, torch.rand(ei.shape[1], generator=gen) < edge_keep]
    else:
        mp_ei = ei
    out = model(x, mp_ei)
    x_ms = float((x ** 2).mean()) if x_ms is None else x_ms
    l_attr = F.mse_loss(out["xhat"], x) / max(x_ms, 1e-8)
    src, dst, lab = sample_pairs(ei, x.shape[0], max_pos, neg_ratio, gen)
    l_struct = F.binary_cross_entropy_with_logits(model.struct_logits(out["z"], src, dst), lab)
    l_inf = F.mse_loss(out["inf"][train_idx], inf_target) if inf_target is not None else x.new_zeros(())
    return out, l_attr, l_struct, l_inf


def train_model(model, x, ei, train_idx, inf_target, *, epochs=100, lr=5e-3, weight_decay=0.0,
                alpha=0.5, edge_keep=1.0, lam_inf=1.0, w_rec=1.0, neg_ratio=1.0, max_pos=200_000, seed=0, log=print,
                log_every=20):
    """Returns a history dict (losses per epoch, wall time). ``epochs=0`` leaves the model untrained."""
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed + 12345)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_ms = float((x ** 2).mean())
    hist = {"loss": [], "attr": [], "struct": [], "inf": [], "neg_ratio": neg_ratio, "edge_keep": edge_keep,
            "max_pos": max_pos, "neg_sampling_seed": seed + 12345}
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        _, la, ls, li = loss_terms(model, x, ei, train_idx, inf_target, alpha, gen, max_pos, neg_ratio, x_ms,
                                     edge_keep)
        loss = w_rec * (alpha * la + (1 - alpha) * ls) + lam_inf * li
        loss.backward()
        opt.step()
        for k, v in (("loss", loss), ("attr", la), ("struct", ls), ("inf", li)):
            hist[k].append(float(v))
        if log and (ep % log_every == 0 or ep == epochs - 1):
            log(f"epoch {ep + 1:4d}/{epochs}  loss={float(loss):.4f} attr={float(la):.4f} "
                f"struct={float(ls):.4f} inf={float(li):.4f}")
    hist["train_seconds"] = time.time() - t0
    model.eval()
    return hist
