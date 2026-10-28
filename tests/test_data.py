import numpy as np
import pytest

from attngnn import data as D
from attngnn.inject import inject_anomalies


def test_clean_edges_symmetrise_dedupe_selfloops():
    ei = np.array([[0, 0, 1, 2, 2], [1, 1, 1, 0, 3]])  # dup 0->1, self loop 1->1, reverse 2->0
    out, st = D.clean_edges(ei, 4)
    pairs = set(zip(out[0].tolist(), out[1].tolist()))
    assert pairs == {(0, 1), (1, 0), (0, 2), (2, 0), (2, 3), (3, 2)}
    assert st["self_loops_removed"] == 1 and st["duplicate_arcs_removed"] == 1


def test_clean_edges_rejects_out_of_range():
    with pytest.raises(ValueError):
        D.clean_edges(np.array([[0], [9]]), 3)


def test_splits_reproducible_disjoint_stratified():
    y = np.zeros(1000, dtype=int)
    y[:50] = 1
    a, b = D.make_splits(y, 3), D.make_splits(y, 3)
    c = D.make_splits(y, 4)
    for k in ("train", "val", "test"):
        assert (a[k] == b[k]).all()
    assert a["sha256"] == b["sha256"] != c["sha256"]
    tot = a["train"].astype(int) + a["val"] + a["test"]
    assert (tot == 1).all()
    for k in ("train", "val", "test"):
        assert 0 < y[a[k]].sum() < a[k].sum()


def test_normaliser_fit_on_train_only():
    x = np.vstack([np.zeros((10, 2)), np.full((10, 2), 100.0)]).astype(np.float32)
    n = D.Normaliser("standard").fit(x[:10])        # train rows are all zero -> std set to 1
    assert np.allclose(n.transform(x)[:10], 0) and np.allclose(n.transform(x)[10:], 100)


def test_injection_reproducible_and_counts():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8)).astype(np.float32)
    ei, _ = D.clean_edges(rng.integers(0, 200, (2, 600)), 200)
    a = inject_anomalies(x, ei, 2, 5, 6, 10, seed=1)
    b = inject_anomalies(x, ei, 2, 5, 6, 10, seed=1)
    c = inject_anomalies(x, ei, 2, 5, 6, 10, seed=2)
    assert (a[0] == b[0]).all() and (a[2] == b[2]).all() and not (a[2] == c[2]).all()
    assert (a[2] == 2).sum() == 10 and (a[2] == 1).sum() == 6
    assert a[3]["edges_added_undirected"] == 2 * 10
    ctx = np.where(a[2] == 1)[0]
    assert not np.allclose(a[0][ctx], x[ctx])        # attributes replaced


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_csv_valid_roundtrip(tmp_path):
    e = _w(tmp_path, "e.csv", "source,target\na,b\nb,c\nb,c\nc,c\n")
    f = _w(tmp_path, "f.csv", "id,f1,f2\na,1,2\nb,3,4\nc,5,6\n")
    l = _w(tmp_path, "l.csv", "id,label\na,0\nb,1\nc,0\n")
    b = D.load_csv_graph(e, f, l)
    assert b.num_nodes == 3 and b.x.shape == (3, 2) and b.y.sum() == 1
    st = b.meta["preprocessing"]
    assert st["self_loops_removed"] == 1 and st["duplicate_arcs_removed"] == 1 and st["undirected_edges"] == 2


def test_csv_without_features_derives_structural(tmp_path):
    e = _w(tmp_path, "e.csv", "source,target\n1,2\n2,3\n")
    b = D.load_csv_graph(e)
    assert b.x.shape == (3, 4) and np.isfinite(b.x).all()


@pytest.mark.parametrize("edges,feats,labels,msg", [
    ("src,dst\n1,2\n", None, None, "source"),
    ("source,target\n1\n", None, None, "columns"),
    ("", None, None, "empty"),
    ("source,target\n1,2\n", "id,f1\n1,abc\n2,1\n", None, "non-numeric"),
    ("source,target\n1,2\n", "id,f1\n1,nan\n2,1\n", None, "NaN"),
    ("source,target\n1,2\n", "id,f1\n1,1\n", None, "no feature row"),
    ("source,target\n1,2\n", "id,f1\n1,1\n1,2\n2,1\n", None, "duplicate"),
    ("source,target\n1,2\n", None, "id,label\n1,5\n", "label must be"),
    ("source,target\n1,2\n", "bad,f1\n1,1\n", None, "features header"),
])
def test_csv_malformed(tmp_path, edges, feats, labels, msg):
    e = _w(tmp_path, "e.csv", edges)
    f = _w(tmp_path, "f.csv", feats) if feats is not None else None
    l = _w(tmp_path, "l.csv", labels) if labels is not None else None
    with pytest.raises(D.CSVSchemaError, match=msg):
        D.load_csv_graph(e, f, l)


def test_csv_size_limits(tmp_path):
    e = _w(tmp_path, "e.csv", "source,target\n" + "\n".join(f"{i},{i + 1}" for i in range(50)))
    with pytest.raises(D.CSVSchemaError, match="size limit"):
        D.load_csv_graph(e, max_nodes=10)
    with pytest.raises(D.CSVSchemaError, match="size limit"):
        D.load_csv_graph(e, max_edges=10)
