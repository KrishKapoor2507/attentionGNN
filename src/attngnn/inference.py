"""Reload a saved run and reproduce / apply it.  Used by the dashboard, demos and tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import RUNS_DIR
from . import data as D
from . import experiments as E
from . import scoring as S
from .models import JointModel


def list_runs(root=RUNS_DIR, include_failed=False):
    root = Path(root)
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if (d / "metrics.json").exists():
            out.append(d)
    return out


def load_metrics(run_dir):
    return json.loads((Path(run_dir) / "metrics.json").read_text())


def load_checkpoint(run_dir):
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "checkpoint.pt", weights_only=False)  # our own file (hash recorded in metrics.json)
    c = ck["model_cfg"]
    model = JointModel(c["in_dim"], n_p=c["n_p"], hidden=c["hidden"], heads=c["heads"], layers=c["layers"],
                       kind=c["kind"], share_weights=c.get("share_weights", False))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def verify_checkpoint_hash(run_dir) -> bool:
    from .runs import sha256_path
    m = load_metrics(run_dir)
    return sha256_path(Path(run_dir) / "checkpoint.pt") == m.get("checkpoint_sha256")


def reproduce(run_dir, atol=1e-4):
    """Recompute all scores from the checkpoint and compare with the saved predictions.csv.gz."""
    run_dir = Path(run_dir)
    model, ck = load_checkpoint(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    m = load_metrics(run_dir)
    seed = ck["seed"]
    prep = E.prepare(ck["dataset"], seed, ck["dataset_cfg"], log=lambda s: None)
    tr = np.where(prep["splits"]["train"])[0]
    va = np.where(prep["splits"]["val"])[0]
    comp, _ = E.anomaly_components(model, prep, seed)
    sc, _ = E.build_scores(comp, tr, va, prep["bundle"].y, alpha_fixed=m["anomaly_selection"]["alpha_sel"])
    inf, _ = S.all_influence_scores(model, prep["x"], prep["ei"], tr, prep["p_grid"])
    pred = pd.read_csv(run_dir / "predictions.csv.gz")
    diffs = {}
    for k, v in sc.items():
        diffs["anom_" + k] = float(np.abs(pred["anom_" + k].to_numpy() - v).max())
    for k, v in inf.items():
        if "inf_" + k in pred:
            diffs["inf_" + k] = float(np.abs(pred["inf_" + k].to_numpy() - v).max())
    return {"max_abs_diff": diffs, "ok": all(d <= atol * max(1.0, 1.0) for d in diffs.values())}


def infer_on_bundle(model, ck, bundle: D.GraphBundle, alpha=0.5):
    """Apply a trained model to ANY graph with the same feature dimension.

    Normalisation = the checkpoint's TRAIN statistics.  Robust-z reference statistics for the anomaly
    scores come from the supplied graph itself (no labels are used).  Returns (anomaly_scores, influence_scores)."""
    nm = ck["norm"]
    if bundle.x.shape[1] != ck["model_cfg"]["in_dim"]:
        raise ValueError(f"feature dimension {bundle.x.shape[1]} != model input {ck['model_cfg']['in_dim']}")
    norm = D.Normaliser(nm["kind"], clip=nm["clip"], mean=nm["mean"], std=nm["std"])
    x = torch.from_numpy(norm.transform(bundle.x))
    ei = torch.from_numpy(bundle.edge_index)
    allidx = np.arange(bundle.num_nodes)
    ae, se = S.recon_errors(model, x, ei, seed=0)
    comp = {"attr_err": ae, "struct_err": se, "feat_dev_uniform": S.uniform_discrepancy(x, ei)}
    att = S.attention_edges(model, x, ei)
    if att is not None:
        comp["att_only"] = S.attention_only_score(x, ei, att)
    sc, _ = E.build_scores(comp, allidx, allidx[:0], bundle.y, alpha_fixed=alpha)
    inf, _ = S.all_influence_scores(model, x, ei, allidx, ck["p_grid"])
    return sc, inf
