"""Metrics for anomaly detection and influence ranking, with explicit handling of degenerate cases.

Definitions
-----------
* average precision (AP) = sum_n (R_n - R_{n-1}) P_n over the thresholds of the PR curve
  (scikit-learn ``average_precision_score``, step-wise, no interpolation).
* precision@k = fraction of the k highest-scored nodes that are anomalies (ties broken by a stable sort;
  k defaults to the number of positives in the evaluated set, i.e. R-precision).
* NDCG@k uses the true spread as gain and handles score ties by averaging the gain of tied items
  (the expected value under uniformly random tie-breaking).
* top-k overlap uses the same fractional (expected) treatment of ties at the cut-off.
Undefined metrics (one-class evaluation sets, zero division) are returned as ``None`` / NaN and never
silently replaced by 0.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


# ------------------------------------------------------------------ anomaly ---------------------
def anomaly_metrics(y, score, threshold, k=None):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    n_pos, n = int(y.sum()), len(y)
    out = {"n": n, "support_pos": n_pos, "support_neg": n - n_pos, "threshold": float(threshold),
           "k": None}
    one_class = n_pos == 0 or n_pos == n
    out["one_class"] = one_class
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    out["precision"] = tp / (tp + fp) if tp + fp > 0 else None
    out["recall"] = tp / (tp + fn) if tp + fn > 0 else None
    p, r = out["precision"], out["recall"]
    out["f1"] = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) > 0 else None
    out["n_flagged"] = int(pred.sum())
    if one_class:
        out["roc_auc"] = None
        out["average_precision"] = None
        out["precision_at_k"] = None
    else:
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["average_precision"] = float(average_precision_score(y, score))
        kk = int(k) if k else n_pos
        out["k"] = kk
        order = np.argsort(-score, kind="stable")[:kk]
        out["precision_at_k"] = float(y[order].mean())
    return out


def pr_roc_curves(y, score, max_points=400):
    y = np.asarray(y).astype(int)
    if y.sum() in (0, len(y)):
        return None
    fpr, tpr, _ = roc_curve(y, score)
    pr, rc, _ = precision_recall_curve(y, score)

    def thin(a, b):
        if len(a) <= max_points:
            return a.tolist(), b.tolist()
        idx = np.linspace(0, len(a) - 1, max_points).astype(int)
        return a[idx].tolist(), b[idx].tolist()

    f, t = thin(fpr, tpr)
    r_, p_ = thin(rc, pr)
    return {"fpr": f, "tpr": t, "recall": r_, "precision": p_}


def select_threshold_f1(y_val, s_val):
    """Threshold maximising F1 on the validation set (frozen before test). None if validation is one-class."""
    y_val = np.asarray(y_val).astype(int)
    s_val = np.asarray(s_val, float)
    if y_val.sum() in (0, len(y_val)):
        return None
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    f1 = np.where(prec[:-1] + rec[:-1] > 0, 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12), 0)
    return float(thr[int(np.argmax(f1))])


# ------------------------------------------------------------------ influence -------------------
def _tie_groups(score):
    order = np.argsort(-score, kind="stable")
    s = score[order]
    bounds = np.r_[0, np.where(np.diff(s) != 0)[0] + 1, len(s)]
    return order, bounds


def ndcg_at_k(true_gain, score, k):
    true_gain = np.asarray(true_gain, float)
    score = np.asarray(score, float)
    k = min(int(k), len(score))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    order, bounds = _tie_groups(score)
    dcg = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a >= k:
            break
        grp = true_gain[order[a:b]].mean()
        top = min(b, k)
        dcg += grp * disc[a:top].sum()
    ideal = np.sort(true_gain)[::-1][:k]
    idcg = float((ideal * disc[: len(ideal)]).sum())
    return float(dcg / idcg) if idcg > 0 else float("nan")


def topk_overlap(true_score, pred_score, k):
    """Expected |top-k(pred) ∩ top-k(true)| / k with fractional credit for ties at both cut-offs."""
    true_score = np.asarray(true_score, float)
    pred_score = np.asarray(pred_score, float)
    n = len(true_score)
    k = min(int(k), n)

    def membership(s):
        order = np.argsort(-s, kind="stable")
        cut = s[order[k - 1]]
        above = s > cut
        tied = s == cut
        m = above.astype(float)
        need = k - above.sum()
        m[tied] = need / tied.sum()
        return m

    mt, mp = membership(true_score), membership(pred_score)
    # independent tie-breaking => expected overlap = sum_i P(i in true topk) * P(i in pred topk)
    return float((mt * mp).sum() / k)


def influence_metrics(true_spread, score, ks=(10, 50), frac_ks=(0.05,)):
    t = np.asarray(true_spread, float)
    s = np.asarray(score, float)
    out = {"n": len(t)}
    if np.ptp(s) == 0:
        out.update({"spearman": None, "kendall": None, "constant_scores": True})
    else:
        out["spearman"] = float(stats.spearmanr(t, s)[0])
        out["kendall"] = float(stats.kendalltau(t, s, variant="b")[0])
        out["constant_scores"] = False
    kk = list(ks) + [max(1, int(math.ceil(f * len(t)))) for f in frac_ks]
    out["ties_in_scores"] = int(len(s) - len(np.unique(s)))
    for k in kk:
        out[f"ndcg@{k}"] = ndcg_at_k(t, s, k)
        out[f"topk_overlap@{k}"] = topk_overlap(t, s, k)
    return out


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    if len(v) == 1:
        return [float(v[0]), float(v[0])]
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def bootstrap_metric_ci(fn, n_items, n_boot=200, alpha=0.05, seed=0):
    """Non-parametric bootstrap over candidate nodes. ``fn(idx)`` -> scalar metric on resampled indices."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(rng.integers(0, n_items, n_items))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]
