"""Offline dataset authenticity checks (the networked re-download lives in scripts/validate_datasets.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import download_data as DL
from attngnn import data as D

RAW = ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(not (RAW / "weibo.pt").exists(), reason="datasets not downloaded")


@needs_data
@pytest.mark.parametrize("name", ["weibo", "reddit"])
def test_local_zip_equals_pinned_sha256(name):
    assert DL.sha(RAW / f"{name}.pt.zip") == DL.PINNED[name]


@needs_data
def test_weibo_is_the_original_868_anomaly_file():
    b = D.load_dataset("weibo")
    assert b.num_nodes == 8405 and int(b.y.sum()) == 868 and b.x.shape[1] == 400
    assert b.meta["preprocessing"]["edges_in"] == 407963


@needs_data
def test_reddit_matches_published_statistics():
    b = D.load_dataset("reddit")
    assert b.num_nodes == 10984 and int(b.y.sum()) == 366 and b.x.shape[1] == 64
    assert b.meta["preprocessing"]["edges_in"] == 168016


@needs_data
@pytest.mark.parametrize("seed", [0, 3])
def test_cora_injection_integrity(seed):
    x0, ei0, _ = D.load_cora_clean()
    b = D.load_cora_injected(seed)
    n = x0.shape[0]
    cl = np.array(b.meta["injection"]["clique_nodes"])
    adj = set(zip(b.edge_index[0].tolist(), b.edge_index[1].tolist()))
    assert all((i, j) in adj for c in cl for i in c for j in c if i != j)           # every clique complete
    ctx = np.where(b.y_type == 1)[0]
    rows = {x0[i].tobytes() for i in range(n)}
    assert all(b.x[i].tobytes() in rows and not np.array_equal(b.x[i], x0[i]) for i in ctx)   # copied from another node
    assert np.array_equal(b.x[b.y_type == 0], x0[b.y_type == 0])                     # nothing else touched
    again = D.load_cora_injected(seed)
    assert np.array_equal(b.x, again.x) and np.array_equal(b.edge_index, again.edge_index)
