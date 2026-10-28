"""Checks on the built results (skipped until benchmark results exist)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts" / "results"
RUNS = ROOT / "artifacts" / "runs"

pytestmark = pytest.mark.skipif(not (RES / "results_index.json").exists(), reason="results not built yet")


def _runs():
    out = {}
    for d in RUNS.iterdir():
        mp = d / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            if m.get("tag") == "bench":
                out.setdefault((m["dataset"], m["variant"]), {})[m["seed"]] = m
    return out


def test_no_nonfinite_or_placeholder_in_tables():
    for f in (RES / "tables").glob("*.csv"):
        df = pd.read_csv(f, keep_default_na=False)
        for c in df.columns:
            for v in df[c].astype(str):
                assert v.lower() not in ("nan", "inf", "-inf", "placeholder", "tbd", "todo"), (f.name, c, v)


def test_anomaly_table_mean_std_recomputed_from_per_seed_values():
    runs = _runs()
    checked = 0
    for ds in {d for d, _ in runs}:
        t = pd.read_csv(RES / "tables" / f"anomaly_{ds}.csv", keep_default_na=False)
        for method, var, score in [("GAT reconstruction", "gat_anom", "rec"),
                                   ("GAT attention-only (weighted feature discrepancy)", "gat_anom", "att_only"),
                                   ("Isolation Forest (attributes)", "iforest_attr", "iforest")]:
            if var not in {v for d, v in runs if d == ds}:
                continue
            vals = [m["anomaly"][score]["test"]["roc_auc"] for m in runs[(ds, var)].values()]
            cell = t.loc[t.method == method, "ROC-AUC"].iloc[0]
            mean = float(re.match(r"([-0-9.]+)", cell).group(1))
            assert mean == pytest.approx(np.mean(vals), abs=6e-4)
            if len(vals) > 1:
                sd = float(re.search(r"± ([0-9.]+)", cell).group(1))
                assert sd == pytest.approx(np.std(vals, ddof=1), abs=6e-4)
            checked += 1
    assert checked > 0


def test_results_index_points_to_existing_runs_and_checkpoints():
    idx = json.loads((RES / "results_index.json").read_text())
    for name, info in {**idx["tables"], **idx["figures"]}.items():
        for rid in info["source_runs"]:
            assert (RUNS / rid / "metrics.json").exists(), (name, rid)
    for rid, h in idx["checkpoint_hashes"].items():
        from attngnn.runs import sha256_path
        assert sha256_path(RUNS / rid / "checkpoint.pt") == h


def test_every_bench_run_has_full_provenance():
    for d in RUNS.iterdir():
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        if m.get("tag") != "bench":
            continue
        for f in ("config.json", "command.txt", "env.json", "data.json", "code.json", "log.txt", "predictions.csv.gz"):
            assert (d / f).exists(), (d.name, f)
        assert len(json.loads((d / "data.json").read_text())["split_sha256"]) == 64
