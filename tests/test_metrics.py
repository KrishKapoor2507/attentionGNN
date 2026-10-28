import numpy as np
import pytest

from attngnn import metrics as M


def test_auc_ap_hand_checked():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    m = M.anomaly_metrics(y, s, 0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["average_precision"] == pytest.approx(5 / 6)       # classic scikit-learn docs example
    assert m["confusion"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
    assert m["precision"] == 1.0 and m["recall"] == 0.5 and m["f1"] == pytest.approx(2 / 3)
    assert m["precision_at_k"] == 0.5 and m["k"] == 2


def test_one_class_is_undefined_not_zero():
    m = M.anomaly_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.15)
    assert m["one_class"] and m["roc_auc"] is None and m["average_precision"] is None
    assert M.select_threshold_f1([0, 0], [0.1, 0.2]) is None
    assert M.pr_roc_curves([1, 1], [0.1, 0.2]) is None


def test_no_flagged_gives_undefined_precision():
    m = M.anomaly_metrics([0, 1], [0.1, 0.2], 5.0)
    assert m["precision"] is None and m["recall"] == 0.0 and m["f1"] is None


def test_threshold_selection_maximises_val_f1():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = M.select_threshold_f1(y, s)
    assert M.anomaly_metrics(y, s, t)["f1"] == 1.0


def test_ndcg_hand_checked_and_ties():
    gain = np.array([3.0, 2.0, 1.0, 0.0])
    assert M.ndcg_at_k(gain, np.array([4, 3, 2, 1.0]), 4) == pytest.approx(1.0)
    rev = M.ndcg_at_k(gain, np.array([1, 2, 3, 4.0]), 2)
    dcg = 0 / 1 + 1 / np.log2(3)
    idcg = 3 / 1 + 2 / np.log2(3)
    assert rev == pytest.approx(dcg / idcg)
    # all-tied scores => expected gain of a random order
    tied = M.ndcg_at_k(gain, np.zeros(4), 2)
    assert tied == pytest.approx((1.5 * (1 + 1 / np.log2(3))) / idcg)


def test_topk_overlap_with_ties():
    t = np.array([10, 9, 8, 1, 0.0])
    assert M.topk_overlap(t, np.array([5, 4, 3, 2, 1.0]), 3) == pytest.approx(1.0)
    assert M.topk_overlap(t, np.array([1, 2, 3, 4, 5.0]), 3) == pytest.approx(1 / 3)
    # fully tied prediction: each item has prob 3/5 of being picked -> 3 * 0.6 / 3
    assert M.topk_overlap(t, np.zeros(5), 3) == pytest.approx(0.6)


def test_influence_metrics_constant_scores():
    m = M.influence_metrics(np.arange(10.0), np.ones(10), ks=(3,), frac_ks=())
    assert m["spearman"] is None and m["constant_scores"]
    m2 = M.influence_metrics(np.arange(10.0), np.arange(10.0), ks=(3,), frac_ks=())
    assert m2["spearman"] == pytest.approx(1.0) and m2["kendall"] == pytest.approx(1.0)


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(0).normal(5, 1, 30)
    lo, hi = M.bootstrap_ci(v)
    assert lo < v.mean() < hi
