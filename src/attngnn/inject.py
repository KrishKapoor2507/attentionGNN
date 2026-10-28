"""Reproducible anomaly injection (DOMINANT-style protocol, Ding et al. 2019).

Structural anomalies: ``struct_cliques`` disjoint cliques of ``clique_size`` nodes (all pairwise
edges added).  Contextual anomalies: for each chosen node, sample ``k_candidates`` other nodes and
overwrite its attributes with those of the candidate farthest in Euclidean distance (measured on the
ORIGINAL attributes).  All anomaly nodes are mutually disjoint.  Everything is driven by ``seed``.
The labels are synthetic and must never be described as observed fraud.
"""
from __future__ import annotations

import numpy as np


def inject_anomalies(x, edge_index, struct_cliques, clique_size, n_contextual, k_candidates, seed):
    n = x.shape[0]
    need = struct_cliques * clique_size + n_contextual
    if need > n:
        raise ValueError(f"cannot inject {need} anomalies into {n} nodes")
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(n)[:need]
    struct_nodes = chosen[: struct_cliques * clique_size]
    ctx_nodes = chosen[struct_cliques * clique_size:]

    new_edges = []
    cliques = struct_nodes.reshape(struct_cliques, clique_size) if struct_cliques else np.zeros((0, 0), int)
    for c in cliques:
        a, b = np.triu_indices(clique_size, k=1)
        new_edges.append(np.stack([c[a], c[b]]))
    ei = edge_index
    if new_edges:
        e = np.concatenate(new_edges, 1)
        ei = np.concatenate([edge_index, e, e[::-1]], 1)

    x_new = x.copy()
    for i in ctx_nodes:
        cand = rng.choice(np.delete(np.arange(n), i), size=min(k_candidates, n - 1), replace=False)
        d = np.linalg.norm(x[cand] - x[i], axis=1)
        x_new[i] = x[cand[int(np.argmax(d))]]

    y_type = np.zeros(n, dtype=np.int8)
    y_type[ctx_nodes] = 1
    y_type[struct_nodes] = 2
    info = {
        "protocol": "DOMINANT-style (Ding et al., SDM 2019)",
        "seed": int(seed), "struct_cliques": int(struct_cliques), "clique_size": int(clique_size),
        "n_contextual": int(n_contextual), "k_candidates": int(k_candidates),
        "n_structural_nodes": int(len(struct_nodes)), "n_contextual_nodes": int(len(ctx_nodes)),
        "injection_rate": float(need / n),
        "edges_added_undirected": int(sum(len(c) * (len(c) - 1) // 2 for c in cliques)),
        "clique_nodes": cliques.tolist(),
    }
    return x_new, ei, y_type, info
