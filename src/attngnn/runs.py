"""Run records: provenance, environment, hashing and (de)serialisation helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import ROOT


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def code_identity() -> dict:
    """Identity of the code that produced a run.  The project folder is not required to be a git
    repository; we always record a hash of the source tree and, if git is available, HEAD + dirty state."""
    h = hashlib.sha256()
    files = []
    for sub in ("src", "configs", "scripts", "app"):
        for p in sorted((ROOT / sub).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json", ".toml"):
                h.update(str(p.relative_to(ROOT)).encode())
                h.update(p.read_bytes())
                files.append(str(p.relative_to(ROOT)))
    ident = {"source_tree_sha256": h.hexdigest(), "n_source_files": len(files), "git": None}
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT.resolve():
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
            diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            ident["git"] = {"head": head.stdout.strip() or None, "dirty": bool(status.stdout.strip()),
                            "diff_sha256": sha256_bytes(diff.stdout.encode())}
    except Exception:  # noqa: BLE001
        pass
    if ident["git"] is None:
        ident["git_note"] = "project folder is not a git repository; source_tree_sha256 is the code identity"
    return ident


def environment_info() -> dict:
    import torch
    import torch_geometric
    import sklearn
    import scipy
    import pandas

    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "pandas": pandas.__version__, "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads": torch.get_num_threads(), "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import numba
        info["numba"] = numba.__version__
    except Exception:  # noqa: BLE001
        pass
    try:  # Windows: friendly CPU name
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=20)
        if out.stdout.strip():
            info["cpu"] = out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return info


def peak_memory_mb() -> float | None:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None) or mi.rss  # Windows exposes the true peak working set
        return round(peak / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        return None


class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def sanitize(o):
    """Recursively convert numpy scalars/arrays and map non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def dump_json(obj, path):
    Path(path).write_text(json.dumps(sanitize(obj), indent=2, allow_nan=False), encoding="utf-8")


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
