"""End-to-end: tiny run -> artifacts -> checkpoint reload reproduces saved predictions."""
import json

import numpy as np
import pandas as pd
import pytest

from attngnn import experiments as E
from attngnn import inference as inf
from attngnn.runs import dump_json


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    cfg = E.load_config(smoke=True)
    d = E.run_one("cora_injected", "gat_joint", 0, cfg, tag="test", out_root=out, force=True)
    return d


def test_run_artifacts_complete(smoke_run):
    for f in ("config.json", "command.txt", "env.json", "data.json", "code.json", "log.txt", "checkpoint.pt",
              "predictions.csv.gz", "metrics.json", "curves.json", "history.json"):
        assert (smoke_run / f).exists(), f
    m = json.loads((smoke_run / "metrics.json").read_text())
    assert m["status"] == "complete" and len(m["checkpoint_sha256"]) == 64
    assert "explanation" in m and "att_only" in m["anomaly"] and "recv_att" in m["influence"]["0.1"]["methods"]
    d = json.loads((smoke_run / "data.json").read_text())
    assert len(d["split_sha256"]) == 64 and len(d["edge_index_sha256"]) == 64
    code = json.loads((smoke_run / "code.json").read_text())
    assert len(code["source_tree_sha256"]) == 64


def test_checkpoint_hash_and_reproduction(smoke_run):
    assert inf.verify_checkpoint_hash(smoke_run)
    r = inf.reproduce(smoke_run)
    assert r["ok"], r["max_abs_diff"]


def test_infer_on_bundle_matches_training_graph_shapes(smoke_run):
    model, ck = inf.load_checkpoint(smoke_run)
    from attngnn import data as D
    b = D.load_dataset("cora_injected", 0, ck["dataset_cfg"].get("data_args"))
    sc, isc = inf.infer_on_bundle(model, ck, b)
    assert len(sc["rec"]) == b.num_nodes and np.isfinite(sc["att_only"]).all()
    bad = D.GraphBundle("x", b.x[:, :10], b.edge_index, b.y, b.y_type)
    with pytest.raises(ValueError):
        inf.infer_on_bundle(model, ck, bad)


def test_predictions_contain_no_nonfinite(smoke_run):
    p = pd.read_csv(smoke_run / "predictions.csv.gz")
    num = p.select_dtypes("number")
    assert np.isfinite(num.to_numpy()).all()


def test_json_rejects_nonfinite_by_mapping_to_null(tmp_path):
    dump_json({"a": float("nan"), "b": np.float32(np.inf), "c": [np.int64(3)]}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": None, "b": None, "c": [3]}
