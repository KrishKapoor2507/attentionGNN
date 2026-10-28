"""Attention-driven GNN for influence ranking and anomaly detection (CPU-first academic project)."""

import os
from pathlib import Path

__version__ = "0.1.0"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS = ROOT / "artifacts"
RUNS_DIR = Path(os.environ.get("ATTNGNN_RUNS_DIR", ARTIFACTS / "runs"))
