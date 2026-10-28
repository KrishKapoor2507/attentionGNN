"""Shared graph encoder with task heads.

Dimensions (defaults): input d -> hidden 64 -> hidden 64 (two message-passing layers).
  GAT   : each layer uses ``heads`` heads of width hidden/heads, concatenated  -> hidden
  GATv2 : same widths, dynamic attention (Brody et al., 2022)
  GCN   : GCNConv(d,64), GCNConv(64,64), symmetric degree-normalised aggregation (Kipf & Welling)
Edge convention (PyG): ``edge_index[0]`` = source j, ``edge_index[1]`` = target i; messages flow
j -> i and attention coefficients are soft-maxed over the incoming arcs of each target i
(self-loops added inside the layer).
Heads
  * attribute decoder : one conv of the same family (hidden -> hidden) + Linear(hidden -> d)  (x_hat)
  * structure decoder : sigmoid(z_i . z_j) evaluated on sampled positive / negative pairs only
  * influence head    : MLP hidden -> 32 -> n_p  (one output per cascade probability p in the grid)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv

ATTN_KINDS = ("gat", "gatv2")


def _conv(kind, in_dim, out_dim, heads, share_weights=False):
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "gat":
        return GATConv(in_dim, out_dim // heads, heads=heads, concat=True, add_self_loops=True)
    if kind == "gatv2":
        return GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True, add_self_loops=True,
                         share_weights=share_weights)
    raise ValueError(kind)


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, layers=2, kind="gat", share_weights=False):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.kind, self.heads, self.hidden = kind, heads, hidden
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(
            [_conv(kind, dims[i], dims[i + 1], heads, share_weights) for i in range(layers)])

    def forward(self, x, edge_index, return_attention=False):
        attn = []
        h = x
        for conv in self.convs:
            if return_attention and self.kind in ATTN_KINDS:
                h, (ei2, alpha) = conv(h, edge_index, return_attention_weights=True)
                attn.append((ei2, alpha))
            else:
                h = conv(h, edge_index)
            h = F.elu(h)
        return (h, attn) if return_attention else h


class JointModel(nn.Module):
    def __init__(self, in_dim, n_p=1, hidden=64, heads=4, layers=2, kind="gat", share_weights=False):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, n_p=n_p, hidden=hidden, heads=heads, layers=layers, kind=kind,
                        share_weights=share_weights)
        self.encoder = Encoder(in_dim, hidden, heads, layers, kind, share_weights)
        # attribute decoder: one graph conv of the same family at hidden width, then a linear map to d
        # (propagating at width d would move d-dim messages along every edge: ~10x slower on CPU)
        self.dec_conv = _conv(kind, hidden, hidden, heads, share_weights)
        self.dec_lin = nn.Linear(hidden, in_dim)
        self.inf_head = nn.Sequential(nn.Linear(hidden, 32), nn.ELU(), nn.Linear(32, n_p))

    def forward(self, x, edge_index, return_attention=False):
        if return_attention:
            z, attn = self.encoder(x, edge_index, True)
        else:
            z, attn = self.encoder(x, edge_index), None
        xhat = self.dec_lin(F.elu(self.dec_conv(z, edge_index)))
        out = {"z": z, "xhat": xhat, "inf": self.inf_head(z)}
        if return_attention:
            out["attn"] = attn
        return out

    @staticmethod
    def struct_logits(z, src, dst):
        return (z[src] * z[dst]).sum(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_encoder_params(self):
        return sum(p.numel() for p in self.encoder.parameters())


def build_model(cfg: dict, in_dim: int, n_p: int) -> JointModel:
    return JointModel(in_dim, n_p=n_p, hidden=cfg.get("hidden", 64), heads=cfg.get("heads", 4),
                      layers=cfg.get("layers", 2), kind=cfg.get("encoder", "gat"),
                      share_weights=cfg.get("share_weights", False))
