"""Validation-only check of the Weibo feature preprocessing with the AUTHENTIC labels (seed 0, GAT, anomaly-only).

    python scripts/explore_weibo_preprocessing.py   -> artifacts/exploration/weibo_preprocessing_authentic_labels.json
Compares plain standardisation (used in the benchmark), clipping at +-5 sd and sign-log compression using the VALIDATION
ROC-AUC of the attribute-reconstruction error only; test labels are not looked at.
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from attngnn import experiments as E
from attngnn import metrics as M
from attngnn.models import build_model
from attngnn.train import train_model

torch.set_num_threads(4)
cfg = E.load_config()
base = cfg["datasets"]["weibo"]
out = {}
for name, over in [("standard (benchmark choice)", {}), ("standardise + clip at 5 sd", {"clip": 5.0}),
                   ("standardise + sign-log", {"normalise": "standard_slog"})]:
    d = {**base, **over}
    prep = E.prepare("weibo", 0, d, log=lambda m: None)
    tr = np.where(prep["splits"]["train"])[0]
    va = np.where(prep["splits"]["val"])[0]
    torch.manual_seed(0)
    model = build_model({"encoder": "gat", "heads": 4, **cfg["model"]}, prep["x"].shape[1], len(prep["p_grid"]))
    t0 = time.time()
    train_model(model, prep["x"], prep["ei"], torch.from_numpy(tr), None, epochs=d["epochs"], lr=cfg["train"]["lr"],
                weight_decay=cfg["train"]["weight_decay"], alpha=0.5, edge_keep=d["edge_keep"], lam_inf=0.0, w_rec=1.0, seed=0, log=None)
    comp, _ = E.anomaly_components(model, prep, 0)
    y = prep["bundle"].y
    auc = M.anomaly_metrics(y[va], comp["attr_err"][va], 0.0)["roc_auc"]
    out[name] = {"validation_auc_attr_err": auc, "train_seconds": time.time() - t0}
    print(name, f"val AUC {auc:.3f}", flush=True)
p = ROOT / "artifacts" / "exploration" / "weibo_preprocessing_authentic_labels.json"
p.write_text(json.dumps(out, indent=2))
print("written", p)
