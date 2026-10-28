"""Dataset loading, preprocessing, splits, normalisation and CSV import.

Conventions
-----------
* A graph is stored as ``edge_index`` of shape [2, E] with row 0 = source and row 1 = target.
* All graphs used by the models are made *symmetric* (every undirected edge appears in both
  directions), de-duplicated and without self-loops.  Self-loops are added inside the GNN layers.
* ``y`` is binary (1 = anomaly).  ``y_type`` keeps the anomaly kind for injected data
  (0 normal, 1 contextual, 2 structural); for organic datasets ``y_type`` equals ``y``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import RAW_DIR

# PINNED to the original 2022 commit: the upstream main branch changed weibo.pt in Jan 2024 and blanked the labels of
# validation/test-mask nodes (see scripts/download_data.py and docs/dataset_validation.md).
PYGOD_COMMIT = "13e75cd16e5b674ac84fd6adfea8286ef0b69203"
PYGOD_BASE = f"https://github.com/pygod-team/data/raw/{PYGOD_COMMIT}/"
# Checksums of downloaded files are recorded in data/raw/checksums.json by scripts/download_data.py.

MAX_NODES_CSV = 100_000
MAX_EDGES_CSV = 2_000_000
MAX_FEATS_CSV = 2_000


# ----------------------------------------------------------------------------------------------
@dataclass
class GraphBundle:
    name: str
    x: np.ndarray                 # [N, d] float32 (raw, un-normalised)
    edge_index: np.ndarray        # [2, E] int64 symmetric, no self loops, no duplicates
    y: np.ndarray                 # [N] int8 binary anomaly label (-1 = unknown)
    y_type: np.ndarray            # [N] int8
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges_directed(self) -> int:
        return int(self.edge_index.shape[1])

    def has_labels(self) -> bool:
        return bool((self.y >= 0).all())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
def clean_edges(edge_index: np.ndarray, num_nodes: int) -> tuple[np.ndarray, dict]:
    """Symmetrise, drop self-loops and duplicates.  Returns (edge_index, stats)."""
    ei = np.asarray(edge_index, dtype=np.int64)
    stats = {"edges_in": int(ei.shape[1])}
    if ei.size and (ei.min() < 0 or ei.max() >= num_nodes):
        raise ValueError("edge_index contains node ids outside [0, num_nodes)")
    loops = ei[0] == ei[1]
    stats["self_loops_removed"] = int(loops.sum())
    ei = ei[:, ~loops]
    # asymmetry before symmetrisation (fraction of arcs without a reverse arc)
    key = ei[0] * num_nodes + ei[1]
    rkey = ei[1] * num_nodes + ei[0]
    ukey = np.unique(key)
    stats["duplicate_arcs_removed"] = int(key.size - ukey.size)
    has_rev = np.isin(ukey, rkey)
    stats["arcs_without_reverse_before_symmetrise"] = int((~has_rev).sum())
    stats["directed_in_source"] = bool((~has_rev).any())
    allkey = np.unique(np.concatenate([ukey, (ukey % num_nodes) * num_nodes + ukey // num_nodes]))
    out = np.stack([allkey // num_nodes, allkey % num_nodes]).astype(np.int64)
    stats["edges_out_directed_arcs"] = int(out.shape[1])
    stats["undirected_edges"] = int(out.shape[1] // 2)
    return out, stats


def _load_pygod_pt(path: Path):
    """Load a PyG ``Data`` object saved by PyGOD with the *safe* unpickler (weights_only=True)."""
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
    return torch.load(path, weights_only=True)


PYGOD_INFO = {
    "weibo": {
        "source": "PyGOD data repository (pygod-team/data), BOND benchmark",
        "url": PYGOD_BASE + "weibo.pt.zip",
        "label_meaning": "organic anomaly label of the ORIGINAL 2022 benchmark file (868 suspicious users; BOND Appendix A.1.1)",
        "organic": True,
    },
    "reddit": {
        "source": "PyGOD data repository (pygod-team/data), BOND benchmark",
        "url": PYGOD_BASE + "reddit.pt.zip",
        "label_meaning": "organic anomaly label shipped with the benchmark (banned users; see docs/literature/dataset_facts.md)",
        "organic": True,
    },
}


def load_organic(name: str, raw_dir: Path = RAW_DIR) -> GraphBundle:
    if name not in PYGOD_INFO:
        raise ValueError(f"unknown organic dataset {name!r}")
    pt = Path(raw_dir) / f"{name}.pt"
    if not pt.exists():
        raise FileNotFoundError(
            f"{pt} not found. Run:  python scripts/download_data.py   (see README troubleshooting)")
    d = _load_pygod_pt(pt)
    x = d.x.numpy().astype(np.float32)
    n = x.shape[0]
    ei, st = clean_edges(d.edge_index.numpy(), n)
    y = d.y.numpy().astype(np.int8)
    y = (y > 0).astype(np.int8)
    info = dict(PYGOD_INFO[name])
    zip_path = Path(raw_dir) / f"{name}.pt.zip"
    meta = {
        **info,
        "preprocessing": st,
        "pt_sha256": sha256_file(pt),
        "zip_sha256": sha256_file(zip_path) if zip_path.exists() else None,
        "raw_node_count": n, "raw_feature_dim": int(x.shape[1]),
        "injected": False,
    }
    return GraphBundle(name=name, x=x, edge_index=ei, y=y, y_type=y.copy(), meta=meta)


def load_cora_clean(raw_dir: Path = RAW_DIR):
    """Plain Planetoid Cora (topic classes are NOT anomaly labels). Returns x, edge_index, topic."""
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(root=str(Path(raw_dir) / "planetoid"), name="Cora")
    d = ds[0]
    return d.x.numpy().astype(np.float32), d.edge_index.numpy(), d.y.numpy()


def load_cora_injected(seed: int, struct_cliques: int = 7, clique_size: int = 10,
                       n_contextual: int = 70, k_candidates: int = 50,
                       raw_dir: Path = RAW_DIR) -> GraphBundle:
    from .inject import inject_anomalies

    x0, ei0, topic = load_cora_clean(raw_dir)
    n = x0.shape[0]
    ei0, st = clean_edges(ei0, n)
    x, ei, y_type, inj = inject_anomalies(x0, ei0, struct_cliques, clique_size, n_contextual,
                                          k_candidates, seed)
    ei, st2 = clean_edges(ei, n)
    y = (y_type > 0).astype(np.int8)
    meta = {
        "source": "Planetoid Cora (McCallum et al. 2000; Yang et al. 2016) via torch_geometric",
        "url": "https://github.com/kimiyoung/planetoid",
        "label_meaning": "INJECTED anomalies (synthetic, DOMINANT-style protocol); NOT observed fraud. "
                         "1 = contextual, 2 = structural.",
        "organic": False, "injected": True,
        "injection": inj, "injection_seed": seed,
        "preprocessing_clean_graph": st, "preprocessing_after_injection": st2,
        "raw_node_count": n, "raw_feature_dim": int(x0.shape[1]),
        "topic_classes": int(topic.max() + 1),
    }
    return GraphBundle(name="cora_injected", x=x, edge_index=ei, y=y, y_type=y_type.astype(np.int8),
                       meta=meta)


def load_dataset(name: str, seed: int = 0, cfg: dict | None = None) -> GraphBundle:
    cfg = cfg or {}
    if name in PYGOD_INFO:
        return load_organic(name)
    if name == "cora_injected":
        kw = {k: cfg[k] for k in ("struct_cliques", "clique_size", "n_contextual", "k_candidates")
              if k in cfg}
        return load_cora_injected(seed, **kw)
    raise ValueError(f"unknown dataset {name!r}")


# ----------------------------------------------------------------------------------------------
def make_splits(y: np.ndarray, seed: int, fractions=(0.3, 0.2, 0.5)) -> dict:
    """Stratified train/val/test node split. Returns dict of boolean masks + checksum.

    Stratification by the binary label guarantees both classes in every split whenever there are
    at least 3 members of the class.  If labels are unknown (-1) the split is simply random.
    """
    assert abs(sum(fractions) - 1) < 1e-9
    rng = np.random.default_rng(seed)
    n = len(y)
    split = np.empty(n, dtype=np.int8)
    groups = [np.where(y == c)[0] for c in np.unique(y)]
    for idx in groups:
        idx = rng.permutation(idx)
        n_tr = int(round(fractions[0] * len(idx)))
        n_va = int(round(fractions[1] * len(idx)))
        split[idx[:n_tr]] = 0
        split[idx[n_tr:n_tr + n_va]] = 1
        split[idx[n_tr + n_va:]] = 2
    masks = {"train": split == 0, "val": split == 1, "test": split == 2}
    masks["sha256"] = sha256_array(split)
    return masks


@dataclass
class Normaliser:
    kind: str = "standard"      # 'standard', 'standard_slog' (sign*log1p|z| after standardising) or 'none'
    clip: float | None = None   # winsorise standardised values to [-clip, clip]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x_train: np.ndarray):
        if self.kind in ("standard", "standard_slog"):
            self.mean = x_train.mean(0)
            s = x_train.std(0)
            self.std = np.where(s < 1e-8, 1.0, s)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return x.astype(np.float32)
        z = (x - self.mean) / self.std
        if self.kind == "standard_slog":
            z = np.sign(z) * np.log1p(np.abs(z))
        if self.clip:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)


def graph_summary(b: GraphBundle) -> dict:
    """Plain-Python summary used by the dashboard, README tables and the report."""
    n, d = b.x.shape
    deg = np.bincount(b.edge_index[0], minlength=n)
    s = {
        "name": b.name, "nodes": n, "undirected_edges": b.num_edges_directed // 2,
        "directed_arcs": b.num_edges_directed, "features": int(d),
        "isolated_nodes": int((deg == 0).sum()), "mean_degree": float(deg.mean()),
        "max_degree": int(deg.max()), "median_degree": float(np.median(deg)),
        "missing_values": int(np.isnan(b.x).sum()),
        "feature_min": float(np.nanmin(b.x)), "feature_max": float(np.nanmax(b.x)),
        "direction": "symmetrised (undirected) for modelling",
        "self_loops_after_cleaning": 0,
    }
    if b.has_labels():
        s["anomalies"] = int(b.y.sum())
        s["anomaly_rate"] = float(b.y.mean())
        s["label_provenance"] = b.meta.get("label_meaning")
        if b.meta.get("injected"):
            s["contextual"] = int((b.y_type == 1).sum())
            s["structural"] = int((b.y_type == 2).sum())
    s["meta"] = {k: v for k, v in b.meta.items() if k not in ("injection",)}
    if "injection" in b.meta:
        s["injection"] = {k: v for k, v in b.meta["injection"].items() if k != "clique_nodes"}
    return s


# ----------------------------------------------------------------------------------------------
class CSVSchemaError(ValueError):
    pass


def load_csv_graph(edges_path, features_path=None, labels_path=None,
                   max_nodes=MAX_NODES_CSV, max_edges=MAX_EDGES_CSV, max_feats=MAX_FEATS_CSV,
                   name="csv_upload") -> GraphBundle:
    """Import a graph from CSV.

    Schemas
    -------
    edges.csv     header ``source,target`` (extra columns ignored); IDs are arbitrary strings/ints.
                  Edges are treated as undirected; duplicates and self-loops are dropped and counted.
    features.csv  (optional) header ``id,<feature columns...>``; every numeric; one row per node id.
                  If omitted, four structural features are derived (degree, log-degree,
                  mean-neighbour-degree, PageRank).
    labels.csv    (optional) header ``id,label`` with label in {0,1}; for evaluation only.
    Node ids appearing only in features/labels but not in edges become isolated nodes.
    """
    ids: dict[str, int] = {}

    def nid(s: str) -> int:
        s = s.strip()
        if s == "":
            raise CSVSchemaError("empty node id")
        if s not in ids:
            if len(ids) >= max_nodes:
                raise CSVSchemaError(f"more than {max_nodes} nodes (size limit)")
            ids[s] = len(ids)
        return ids[s]

    src, dst = [], []
    with open(edges_path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        try:
            header = [h.strip().lower() for h in next(r)]
        except StopIteration:
            raise CSVSchemaError("edges file is empty")
        if "source" not in header or "target" not in header:
            raise CSVSchemaError(f"edges header must contain 'source' and 'target', got {header}")
        si, ti = header.index("source"), header.index("target")
        for ln, row in enumerate(r, start=2):
            if not row or all(c.strip() == "" for c in row):
                continue
            if len(row) <= max(si, ti):
                raise CSVSchemaError(f"edges line {ln}: expected at least {max(si, ti) + 1} columns")
            src.append(nid(row[si]))
            dst.append(nid(row[ti]))
            if len(src) > max_edges:
                raise CSVSchemaError(f"more than {max_edges} edges (size limit)")

    feat_rows = None
    if features_path:
        feat_rows = {}
        with open(features_path, newline="", encoding="utf-8-sig") as f:
            r = csv.reader(f)
            header = [h.strip() for h in next(r)]
            if header[0].lower() != "id" or len(header) < 2:
                raise CSVSchemaError("features header must be 'id,<f1>,<f2>,...'")
            if len(header) - 1 > max_feats:
                raise CSVSchemaError(f"more than {max_feats} feature columns")
            for ln, row in enumerate(r, start=2):
                if not row:
                    continue
                if len(row) != len(header):
                    raise CSVSchemaError(f"features line {ln}: expected {len(header)} columns, got {len(row)}")
                try:
                    vals = [float(v) for v in row[1:]]
                except ValueError:
                    raise CSVSchemaError(f"features line {ln}: non-numeric value")
                if not np.isfinite(vals).all():
                    raise CSVSchemaError(f"features line {ln}: NaN/inf value")
                if row[0].strip() in feat_rows:
                    raise CSVSchemaError(f"features line {ln}: duplicate id {row[0]!r}")
                feat_rows[row[0].strip()] = vals
        for k in feat_rows:
            nid(k)

    lab_rows = None
    if labels_path:
        lab_rows = {}
        with open(labels_path, newline="", encoding="utf-8-sig") as f:
            r = csv.reader(f)
            header = [h.strip().lower() for h in next(r)]
            if header[:2] != ["id", "label"]:
                raise CSVSchemaError("labels header must be 'id,label'")
            for ln, row in enumerate(r, start=2):
                if not row:
                    continue
                if row[1].strip() not in ("0", "1"):
                    raise CSVSchemaError(f"labels line {ln}: label must be 0 or 1")
                lab_rows[row[0].strip()] = int(row[1])
        for k in lab_rows:
            nid(k)

    n = len(ids)
    if n == 0:
        raise CSVSchemaError("no nodes found")
    if feat_rows is not None:
        missing = [k for k in ids if k not in feat_rows]
        if missing:
            raise CSVSchemaError(f"{len(missing)} node ids have no feature row, e.g. {missing[:3]}")
    ei, st = clean_edges(np.array([src, dst], dtype=np.int64).reshape(2, -1), n)
    if feat_rows is not None:
        x = np.zeros((n, len(next(iter(feat_rows.values())))), dtype=np.float32)
        for k, i in ids.items():
            x[i] = feat_rows[k]
        feat_note = "user-supplied"
    else:
        x = structural_features(ei, n)
        feat_note = "derived structural features: degree, log1p(degree), mean neighbour degree, PageRank"
    if lab_rows is not None:
        y = np.full(n, -1, dtype=np.int8)
        for k, v in lab_rows.items():
            y[ids[k]] = v
    else:
        y = np.full(n, -1, dtype=np.int8)
    meta = {"source": "user CSV upload", "preprocessing": st, "features": feat_note,
            "label_meaning": "user supplied" if lab_rows is not None else "none", "injected": False,
            "organic": None, "id_map_size": n}
    b = GraphBundle(name=name, x=x, edge_index=ei, y=y, y_type=y.copy(), meta=meta)
    b.meta["id_to_index"] = ids
    return b


def structural_features(ei: np.ndarray, n: int) -> np.ndarray:
    import scipy.sparse as sp

    a = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    deg = np.asarray(a.sum(1)).ravel()
    nd = np.asarray(a @ deg).ravel() / np.maximum(deg, 1)
    # power-iteration PageRank (alpha 0.85)
    pr = np.full(n, 1.0 / n)
    inv = 1.0 / np.maximum(deg, 1)
    for _ in range(60):
        pr = 0.15 / n + 0.85 * (a.T @ (pr * inv) + (pr[deg == 0].sum() / n))
    return np.stack([deg, np.log1p(deg), nd, pr * n], 1).astype(np.float32)
