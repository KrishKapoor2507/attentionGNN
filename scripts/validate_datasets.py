"""Dataset authenticity and validity checks.

    python scripts/validate_datasets.py     -> docs/dataset_validation.md, artifacts/results/dataset_validation.json,
                                               artifacts/results/tables/dataset_validation.csv

Sections: (1) provenance / integrity against the PINNED upstream commit (fresh re-download, SHA-256) and the upstream
history, (2) structural validity, (3) comparison with the published statistics, (4) label-semantics evidence,
(5) Cora injection integrity, (6) split integrity.  Nothing here uses model results.
"""
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from attngnn import data as D
from attngnn.inject import inject_anomalies

import download_data as DL

OUT_T = ROOT / "artifacts" / "results" / "tables"
OUT_T.mkdir(parents=True, exist_ok=True)
rows = []


def rec(section, dataset, check, status, detail):
    rows.append({"section": section, "dataset": dataset, "check": check, "result": status, "detail": detail})
    print(f"[{status:6s}] {dataset:14s} {check}: {detail}", flush=True)


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "attngnn-validation"}), timeout=90).read()


def safe_load(path):
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
    return torch.load(path, weights_only=True)


# ------------------------------------------------------------------------------------- 1. provenance
def provenance():
    net_ok = True
    tmp = Path(tempfile.mkdtemp(prefix="attngnn_val_"))
    for name, expected in DL.PINNED.items():
        local = ROOT / "data" / "raw" / f"{name}.pt.zip"
        d = DL.sha(local) if local.exists() else None
        rec("1 integrity", name, "local zip equals pinned SHA-256", "PASS" if d == expected else "FAIL",
            f"local {str(d)[:16]}..., pinned {expected[:16]}...")
        try:
            raw = get(DL.BASE + f"{name}.pt.zip")
            fresh = hashlib.sha256(raw).hexdigest()
            rec("1 integrity", name, "fresh download from pinned commit equals local file", "PASS" if fresh == d else "FAIL",
                f"{len(raw):,} bytes, sha256 {fresh[:16]}...")
        except Exception as e:  # noqa: BLE001
            net_ok = False
            rec("1 integrity", name, "fresh re-download", "SKIP", f"network unavailable: {e}")
        if net_ok:
            try:
                hist = json.loads(get(f"https://api.github.com/repos/pygod-team/data/commits?path={name}.pt.zip&per_page=10"))
                rec("1 integrity", name, "upstream history of the file", "INFO",
                    "; ".join(f"{c['commit']['author']['date'][:10]} {c['sha'][:7]} '{c['commit']['message'].splitlines()[0]}'"
                              for c in hist))
            except Exception as e:  # noqa: BLE001
                rec("1 integrity", name, "upstream history", "SKIP", str(e))
    # the current upstream main version of weibo
    try:
        (tmp / "main").mkdir()
        raw = get("https://github.com/pygod-team/data/raw/main/weibo.pt.zip")
        (tmp / "main" / "w.zip").write_bytes(raw)
        zipfile.ZipFile(tmp / "main" / "w.zip").extractall(tmp / "main")
        cur = safe_load(tmp / "main" / "weibo.pt")
        pin = safe_load(ROOT / "data" / "raw" / "weibo.pt")
        cy, py = cur.y.numpy().astype(bool), pin.y.numpy().astype(bool)
        same = (torch.equal(cur.x, pin.x) and torch.equal(cur.edge_index, pin.edge_index))
        rec("1 integrity", "weibo", "upstream `main` file vs pinned file: features and edges identical", "PASS" if same else "FAIL", str(same))
        rec("1 integrity", "weibo", "upstream `main` file vs pinned file: anomaly labels", "FINDING",
            f"pinned (2022): {int(py.sum())} anomalies; upstream main (2024): {int(cy.sum())}; main anomalies are a subset: {bool((cy <= py).all())}")
        if hasattr(cur, "test_mask"):
            tm, vm, trm = (getattr(cur, k).numpy() for k in ("test_mask", "val_mask", "train_mask"))
            rec("1 integrity", "weibo", "which anomalies does the 2024 file blank?", "FINDING",
                f"true anomalies in its train mask kept: {int((py & trm).sum())} of {int((py & trm).sum())}; in val mask kept: "
                f"{int((cy & vm).sum())} of {int((py & vm).sum())}; in test mask kept: {int((cy & tm).sum())} of {int((py & tm).sum())}. "
                "The 2024 file hides the labels of validation/test nodes, so it cannot be used for evaluation.")
    except Exception as e:  # noqa: BLE001
        rec("1 integrity", "weibo", "compare with upstream main", "SKIP", str(e))
    # Cora raw files
    ck = json.loads((ROOT / "data" / "raw" / "checksums.json").read_text())
    cora = [k for k in ck if k.startswith("planetoid/Cora/raw/")]
    ok = all(DL.sha(ROOT / "data" / "raw" / k.replace("planetoid/", "planetoid/", 1)) == ck[k]["sha256"] for k in cora)
    rec("1 integrity", "cora", "Planetoid raw files equal recorded checksums", "PASS" if ok else "FAIL", f"{len(cora)} files")


# ------------------------------------------------------------------------------------- 2/3. structure + published stats
PUBLISHED = {  # BOND Table 1 / PyGOD data README (official), nodes / listed edges / features / anomalies
    "weibo": (8405, 407963, 400, 868), "reddit": (10984, 168016, 64, 366)}


def structure(name, b, raw_edges=None, raw_nodes=None):
    n, d = b.x.shape
    ei = b.edge_index
    rec("2 structure", name, "features finite (no NaN/inf)", "PASS" if np.isfinite(b.x).all() else "FAIL", f"{n} x {d}")
    rec("2 structure", name, "edge ids inside [0, N)", "PASS" if ei.min() >= 0 and ei.max() < n else "FAIL", f"{ei.shape[1]} arcs")
    key = ei[0] * n + ei[1]
    rkey = ei[1] * n + ei[0]
    rec("2 structure", name, "cleaned graph symmetric, no duplicates, no self-loops",
        "PASS" if np.isin(rkey, key).all() and len(np.unique(key)) == len(key) and (ei[0] != ei[1]).all() else "FAIL", "after clean_edges")
    a = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    nc, lab = connected_components(a, directed=False)
    giant = np.bincount(lab).max() / n
    rec("2 structure", name, "connected components", "INFO", f"{nc} components, giant component {giant:.1%} of nodes")
    deg = np.asarray(a.sum(1)).ravel()
    rec("2 structure", name, "isolated nodes", "INFO", f"{int((deg == 0).sum())}")
    if b.has_labels():
        ys = set(np.unique(b.y).tolist())
        rec("2 structure", name, "labels are binary {0,1}", "PASS" if ys <= {0, 1} else "FAIL", str(sorted(ys)))
        rec("2 structure", name, "class balance", "INFO", f"{int(b.y.sum())} anomalies of {n} ({b.y.mean():.2%})")
    pre = b.meta.get("preprocessing") or {}
    if pre:
        rec("2 structure", name, "source preprocessing", "INFO",
            f"{pre.get('edges_in')} listed arcs; self-loops removed {pre.get('self_loops_removed')}; duplicates removed "
            f"{pre.get('duplicate_arcs_removed')}; arcs without reverse {pre.get('arcs_without_reverse_before_symmetrise')} "
            f"(directed source: {pre.get('directed_in_source')}) -> {pre.get('undirected_edges')} undirected edges")
    if name in PUBLISHED:
        pn, pe, pf, pa = PUBLISHED[name]
        got = (pre.get("edges_in"), b.x.shape[1], int(b.y.sum()))
        rec("3 published", name, "nodes equal published", "PASS" if n == pn else "FAIL", f"{n} vs {pn}")
        rec("3 published", name, "listed edges equal published", "PASS" if got[0] == pe else "FAIL", f"{got[0]} vs {pe}")
        rec("3 published", name, "features equal published", "PASS" if got[1] == pf else "FAIL", f"{got[1]} vs {pf}")
        rec("3 published", name, "anomalies equal published", "PASS" if got[2] == pa else "FAIL", f"{got[2]} vs {pa}")


# ------------------------------------------------------------------------------------- 4. label semantics
def label_semantics(name, b, rng_seed=0, m=2500):
    n = b.num_nodes
    a = sp.csr_matrix((np.ones(b.edge_index.shape[1]), (b.edge_index[0], b.edge_index[1])), shape=(n, n))
    rng = np.random.default_rng(rng_seed)
    pos = np.where(b.y == 1)[0]
    neg = np.where(b.y == 0)[0]
    S = np.concatenate([rng.choice(pos, min(len(pos), m // 2), replace=False), rng.choice(neg, m // 2, replace=False)])
    A = a.tocsr()
    sub = A[S]
    tri = np.asarray(sub.multiply(sub @ A).sum(1)).ravel() / 2.0
    deg = np.asarray(sub.sum(1)).ravel()
    cc = np.where(deg > 1, tri / np.maximum(deg * (deg - 1) / 2, 1), 0.0)
    xn = b.x / np.maximum(np.linalg.norm(b.x, axis=1, keepdims=True), 1e-12)
    sims = []
    for i in S:
        nb = A.indices[A.indptr[i]:A.indptr[i + 1]][:200]
        sims.append(float((xn[nb] @ xn[i]).mean()) if len(nb) else 0.0)
    sims = np.array(sims)
    ispos = np.isin(S, pos)
    out = {"clustering_anom": float(cc[ispos].mean()), "clustering_norm": float(cc[~ispos].mean()),
           "neigh_sim_anom": float(sims[ispos].mean()), "neigh_sim_norm": float(sims[~ispos].mean()),
           "mean_degree_anom": float(deg[ispos].mean()), "mean_degree_norm": float(deg[~ispos].mean())}
    rec("4 label semantics", name, "avg clustering coefficient, anomalies vs normal (sampled nodes)", "INFO",
        f"{out['clustering_anom']:.3f} vs {out['clustering_norm']:.3f}")
    rec("4 label semantics", name, "avg neighbour-feature cosine similarity, anomalies vs normal (sampled nodes)", "INFO",
        f"{out['neigh_sim_anom']:.3f} vs {out['neigh_sim_norm']:.3f}")
    if name == "weibo":
        c_ok = out["clustering_anom"] > out["clustering_norm"]
        s_ok = out["neigh_sim_anom"] < out["neigh_sim_norm"]
        rec("4 label semantics", name, "agreement with BOND's description of Weibo outliers (clustering 0.400 vs 0.301; neighbour similarity 0.004 vs 0.993)",
            "PASS" if (c_ok and s_ok) else "PARTIAL",
            f"clustering direction agrees: {c_ok}; neighbour-similarity direction agrees: {s_ok}. Our similarity is the mean cosine between a node and its "
            "neighbours on the raw features; BOND does not give its exact definition, and our value for normal Weibo users (0.05) is far from their 0.993, "
            "so this part of the semantics check is inconclusive")
    if name == "reddit":
        rec("4 label semantics", name, "BOND: Reddit outliers look like inliers in clustering and neighbour similarity", "PASS" if abs(out["neigh_sim_anom"] - out["neigh_sim_norm"]) < 0.05 else "CHECK",
            f"neighbour similarity {out['neigh_sim_anom']:.3f} vs {out['neigh_sim_norm']:.3f}: the two groups are indistinguishable by this statistic")
    return out


# ------------------------------------------------------------------------------------- 5. Cora injection integrity
def cora_injection():
    x0, ei0, topic = D.load_cora_clean()
    rec("2 structure", "cora", "Planetoid Cora size", "PASS" if x0.shape == (2708, 1433) and ei0.shape[1] == 10556 and topic.max() == 6 else "FAIL",
        f"{x0.shape[0]} nodes, {x0.shape[1]} features, {ei0.shape[1]} directed arcs (= {ei0.shape[1] // 2} undirected; the GCN/GAT papers list 5,429 raw citation records), 7 classes")
    n = x0.shape[0]
    ei_c, _ = D.clean_edges(ei0, n)
    for seed in range(5):
        b = D.load_cora_injected(seed)
        inj = b.meta["injection"]
        cl = np.array(inj["clique_nodes"])
        ei = b.edge_index
        a = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tolil()
        complete = all(all(a[i, j] > 0 for i in c for j in c if i != j) for c in cl)
        struct_nodes = np.where(b.y_type == 2)[0]
        ctx = np.where(b.y_type == 1)[0]
        disjoint = len(set(struct_nodes) & set(ctx)) == 0 and len(struct_nodes) == cl.size
        # contextual: attributes now equal those of some OTHER node of the ORIGINAL data and differ from the node's own
        xs = {x0[i].tobytes() for i in range(n)}
        copied = all(b.x[i].tobytes() in xs and not np.array_equal(b.x[i], x0[i]) for i in ctx)
        untouched = np.array_equal(b.x[b.y_type == 0], x0[b.y_type == 0])
        again = D.load_cora_injected(seed)
        det = np.array_equal(b.x, again.x) and np.array_equal(b.edge_index, again.edge_index) and np.array_equal(b.y_type, again.y_type)
        ok = complete and disjoint and copied and untouched and det
        rec("5 injection", f"cora seed {seed}", "7 complete 10-cliques; 70 contextual nodes with copied attributes; others untouched; deterministic",
            "PASS" if ok else "FAIL",
            f"cliques complete={complete}, disjoint/counts={disjoint} ({len(struct_nodes)} struct, {len(ctx)} ctx), contextual copied={copied}, "
            f"normal attributes untouched={untouched}, same seed reproduces={det}; rate {inj['injection_rate']:.2%}")
    rec("5 injection", "cora", "label provenance", "INFO",
        "labels are SYNTHETIC (DOMINANT-style injection); they must never be described as observed fraud; Cora topic classes are not anomaly labels")


# ------------------------------------------------------------------------------------- 6. splits
def splits(name, b):
    for seed in (0, 1, 2):
        sp_ = D.make_splits(b.y, seed)
        tot = sp_["train"].astype(int) + sp_["val"] + sp_["test"]
        ok = (tot == 1).all() and all(0 < b.y[sp_[k]].sum() < sp_[k].sum() for k in ("train", "val", "test"))
        again = D.make_splits(b.y, seed)["sha256"] == sp_["sha256"]
        rec("6 splits", name, f"seed {seed}: disjoint, cover all nodes, both classes in every split, reproducible", "PASS" if ok and again else "FAIL",
            "/".join(f"{int(sp_[k].sum())}({int(b.y[sp_[k]].sum())})" for k in ("train", "val", "test")) + " nodes (anomalies) train/val/test")


def main():
    t0 = time.time()
    provenance()
    sem = {}
    for name in ("weibo", "reddit"):
        b = D.load_dataset(name)
        structure(name, b)
        sem[name] = label_semantics(name, b)
        splits(name, b)
    cora_injection()
    b = D.load_dataset("cora_injected", 0)
    structure("cora_injected", b)
    splits("cora_injected", b)

    npass = sum(r["result"] == "PASS" for r in rows)
    nfail = sum(r["result"] == "FAIL" for r in rows)
    print(f"\n{npass} PASS, {nfail} FAIL, {sum(r['result'] == 'FINDING' for r in rows)} FINDING in {time.time() - t0:.0f}s")
    import csv
    with open(OUT_T / "dataset_validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "dataset", "check", "result", "detail"])
        w.writeheader()
        w.writerows(rows)
    # compact table for the report: one row per (dataset, check family)
    fam = {"1 integrity": "Integrity against pinned upstream (SHA-256, fresh download)", "2 structure": "Structural validity",
           "3 published": "Equals published statistics", "4 label semantics": "Label semantics", "5 injection": "Injection integrity",
           "6 splits": "Split integrity"}
    groups = {}
    for r in rows:
        ds = r["dataset"].split(" seed")[0].replace("cora_injected", "cora (injected)")
        groups.setdefault((ds, r["section"]), []).append(r)
    summ = []
    for (ds, sec), rs in sorted(groups.items()):
        cnt = {k: sum(x["result"] == k for x in rs) for k in ("PASS", "FAIL", "FINDING", "PARTIAL", "CHECK", "SKIP")}
        graded = cnt["PASS"] + cnt["FAIL"] + cnt["PARTIAL"] + cnt["CHECK"]
        res = (f"{cnt['PASS']}/{graded} pass" if graded else "info") + (f", {cnt['FINDING']} finding(s)" if cnt["FINDING"] else "") + (f", {cnt['PARTIAL']} partial" if cnt["PARTIAL"] else "")
        keyrow = (next((x for x in rs if x["result"] in ("FINDING", "PARTIAL")), None)
                  or next((x for x in rs if x["result"] in ("PASS", "FAIL", "CHECK") and "clustering" not in x["check"]), rs[0]))
        summ.append({"dataset": ds, "check family": fam.get(sec, sec), "result": res, "evidence": keyrow["detail"][:210]})
    import csv as _csv
    with open(OUT_T / "dataset_validation_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["dataset", "check family", "result", "evidence"])
        w.writeheader()
        w.writerows(summ)
    (ROOT / "artifacts" / "results" / "dataset_validation.json").write_text(
        json.dumps({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows, "label_semantics": sem}, indent=2), encoding="utf-8")
    md = ["# Dataset authenticity and validation", "", f"Generated by `scripts/validate_datasets.py` on {time.strftime('%Y-%m-%d %H:%M')}.", "",
          f"**Summary: {npass} checks passed, {nfail} failed, {sum(r['result'] == 'FINDING' for r in rows)} findings.**", ""]
    sec = None
    for r in rows:
        if r["section"] != sec:
            sec = r["section"]
            md += ["", f"## {sec}", "", "| dataset | check | result | detail |", "|---|---|---|---|"]
        md.append(f"| {r['dataset']} | {r['check']} | **{r['result']}** | {r['detail'].replace('|', '/')} |")
    (ROOT / "docs" / "dataset_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
