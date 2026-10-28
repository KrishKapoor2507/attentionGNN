"""Perturbation test of attention as an explanation (association, not causation).

For a target node t and an edge budget m_t = min(m, deg//2):
  * 'high'   : remove the m_t arcs with the largest head/layer-averaged attention
  * 'low'    : remove the m_t arcs with the smallest attention
  * 'random' : remove m_t arcs chosen uniformly (n_random independent draws, averaged)
Anomaly mode looks at arcs ARRIVING at t (what t aggregates); influence mode looks at arcs LEAVING t
(attention that t receives from its targets).  Each removal deletes the undirected edge (both arcs).
The model is frozen; the output of interest at t is recomputed with an identical full forward pass for
every condition.  effect = value_before - value_after  (positive = the removal lowered the score).
If attention were a faithful explanation, 'high' should lower the output more than 'random' and 'low'.
"""
from __future__ import annotations

import numpy as np
import torch


def reverse_arc_index(ei: np.ndarray, n: int) -> np.ndarray:
    key = ei[0].astype(np.int64) * n + ei[1]
    rkey = ei[1].astype(np.int64) * n + ei[0]
    order = np.argsort(key)
    pos = np.searchsorted(key[order], rkey)
    rev = order[pos]
    assert (key[rev] == rkey).all(), "graph must be symmetric"
    return rev


def perturb_node(output_fn, x, ei, node, arcs, rev, base, n_remove, attn, rng, n_random=3):
    """Returns dict condition -> effect for one target node. ``arcs`` = candidate arc indices."""
    if len(arcs) < 2 * n_remove or n_remove < 1:
        return None
    order = np.argsort(-attn[arcs], kind="stable")
    cond = {"high": arcs[order[:n_remove]], "low": arcs[order[::-1][:n_remove]]}
    ei_np = ei.numpy()
    res = {}

    def effect(sel):
        drop = np.zeros(ei_np.shape[1], bool)
        drop[sel] = True
        drop[rev[sel]] = True
        ei2 = torch.from_numpy(ei_np[:, ~drop])
        return base - output_fn(x, ei2, node)

    for k, sel in cond.items():
        res[k] = effect(sel)
    rnd = [effect(rng.choice(arcs, size=n_remove, replace=False)) for _ in range(n_random)]
    res["random"] = float(np.mean(rnd))
    res["n_removed"] = int(n_remove)
    return res


def run_perturbation(output_fn, x, ei, att_alpha, targets, direction, n, m=3, seed=0, n_random=3):
    """output_fn(x, edge_index, node) -> float.  att_alpha: per-arc mean attention aligned with ei.
    Returns list of per-target dicts."""
    rng = np.random.default_rng(seed)
    ei_np = ei.numpy()
    rev = reverse_arc_index(ei_np, n)
    col = 1 if direction == "in" else 0
    rows = []
    for t in targets:
        arcs = np.where(ei_np[col] == t)[0]
        deg = len(arcs)
        m_t = min(m, deg // 2)
        if m_t < 1:
            continue
        base = output_fn(x, ei, t)
        r = perturb_node(output_fn, x, ei, int(t), arcs, rev, base, m_t, att_alpha, rng, n_random)
        if r is not None:
            r.update(node=int(t), degree=int(deg), base=float(base))
            rows.append(r)
    return rows


def summarise_perturbation(rows, seed=0, n_boot=2000):
    from scipy import stats

    if not rows:
        return {"n_targets": 0}
    hi = np.array([r["high"] for r in rows])
    lo = np.array([r["low"] for r in rows])
    rd = np.array([r["random"] for r in rows])
    rng = np.random.default_rng(seed)

    def ci(d):
        b = rng.choice(d, size=(n_boot, len(d))).mean(1)
        return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]

    out = {"n_targets": len(rows),
           "mean_effect": {"high": float(hi.mean()), "random": float(rd.mean()), "low": float(lo.mean())},
           "diff_high_minus_random": float((hi - rd).mean()), "diff_high_minus_random_ci95": ci(hi - rd),
           "diff_high_minus_low": float((hi - lo).mean()), "diff_high_minus_low_ci95": ci(hi - lo),
           "frac_targets_high_gt_random": float((hi > rd).mean())}
    try:
        out["wilcoxon_p_high_vs_random"] = float(stats.wilcoxon(hi, rd, alternative="greater").pvalue)
    except ValueError:
        out["wilcoxon_p_high_vs_random"] = None
    return out
