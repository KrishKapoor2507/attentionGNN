"""End-to-end experiment runs (one run = dataset x variant x seed) with full provenance.

Protocol (see docs/protocol.md for the long version)
  * Transductive: the encoder sees ALL node features and ALL edges (including validation/test nodes) but
    no anomaly labels and no influence targets of validation/test nodes.
  * Splits are stratified by the anomaly label (seeded).  Normalisation statistics come from TRAIN nodes.
  * Anomaly: training is label-free.  Validation labels are used for (i) the reconstruction mixing weight
    alpha_sel (grid) and (ii) the F1-optimal decision threshold; both are frozen before test evaluation.
    This makes model *selection* label-informed even though encoder training is unsupervised.
  * Influence: independent-cascade Monte-Carlo spread.  Train-node targets, validation-node targets and
    held-out test-node targets come from three independent random streams; every ranking method is scored
    on the same held-out test candidates.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import ARTIFACTS, ROOT, RUNS_DIR
from . import data as D
from . import influence as I
from . import metrics as M
from . import scoring as S
from .explain import run_perturbation, summarise_perturbation
from .models import build_model
from .runs import (code_identity, dump_json, environment_info, now_iso, peak_memory_mb, sha256_path)
from .train import train_model

VARIANTS = {
    # name: (encoder, heads, w_rec, lam_inf, epochs_scale)
    "gcn_anom": dict(encoder="gcn", heads=4, w_rec=1.0, lam_inf=0.0),
    "gat_anom": dict(encoder="gat", heads=4, w_rec=1.0, lam_inf=0.0),
    "gatv2_anom": dict(encoder="gatv2", heads=4, w_rec=1.0, lam_inf=0.0),
    "gat_inf": dict(encoder="gat", heads=4, w_rec=0.0, lam_inf=1.0),
    "gcn_joint": dict(encoder="gcn", heads=4, w_rec=1.0, lam_inf=1.0),
    "gat_joint": dict(encoder="gat", heads=4, w_rec=1.0, lam_inf=1.0, explain=True),
    "gatv2_joint": dict(encoder="gatv2", heads=4, w_rec=1.0, lam_inf=1.0),
    "gat1h_joint": dict(encoder="gat", heads=1, w_rec=1.0, lam_inf=1.0),
    "gat_untrained": dict(encoder="gat", heads=4, w_rec=1.0, lam_inf=1.0, untrained=True),
}
BASELINES = ("iforest_attr", "iforest_attr_struct", "centrality")
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_config(path=None, smoke=False) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(path.read_text())
    if smoke:
        sm = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
        for k, v in sm.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        for ds in cfg["datasets"].values():
            for k, v in sm.get("dataset_overrides", {}).items():
                ds[k] = v
    return cfg


class RunLogger:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ----------------------------------------------------------------------------------------------
def prepare(dataset: str, seed: int, dcfg: dict, log=print):
    """Load graph, splits, normaliser, tensors and cached influence targets."""
    b = D.load_dataset(dataset, seed, dcfg.get("data_args"))
    n = b.num_nodes
    splits = D.make_splits(b.y, seed, tuple(dcfg.get("split", (0.3, 0.2, 0.5))))
    norm = D.Normaliser(dcfg.get("normalise", "standard"), clip=dcfg.get("clip")).fit(b.x[splits["train"]])
    x = torch.from_numpy(norm.transform(b.x))
    ei = torch.from_numpy(b.edge_index)
    deg = np.bincount(b.edge_index[0], minlength=n).astype(float)
    p_grid = list(dcfg["p_grid"])
    reps = dcfg["reps"]
    key_src = (sha(b.edge_index) + json.dumps(p_grid) + json.dumps(reps) + splits["sha256"] + str(seed))
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cache = ARTIFACTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cpath = cache / f"targets_{dataset}_{key}.npz"
    seed_base = 1000 + seed
    if cpath.exists():
        z = np.load(cpath)
        targets = {p: {"mean": z[f"mean_{i}"], "se": z[f"se_{i}"]} for i, p in enumerate(p_grid)}
        log(f"influence targets loaded from cache {cpath.name}")
    else:
        t0 = time.time()
        targets = I.simulate_targets(b.edge_index, n, p_grid, splits, reps, seed_base)
        np.savez(cpath, **{f"mean_{i}": targets[p]["mean"] for i, p in enumerate(p_grid)},
                 **{f"se_{i}": targets[p]["se"] for i, p in enumerate(p_grid)})
        log(f"influence targets simulated in {time.time() - t0:.1f}s -> {cpath.name}")
    return dict(bundle=b, n=n, splits=splits, norm=norm, x=x, ei=ei, deg=deg, p_grid=p_grid,
                targets=targets, seed_base=seed_base, target_cache=cpath.name)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def train_targets(prep, p_grid):
    tr = np.where(prep["splits"]["train"])[0]
    cols = []
    stats = []
    for p in p_grid:
        v = np.log1p(prep["targets"][float(p)]["mean"][tr])
        mu, sd = float(v.mean()), float(v.std() + 1e-8)
        cols.append((v - mu) / sd)
        stats.append((mu, sd))
    return torch.tensor(np.stack(cols, 1), dtype=torch.float32), stats


# ----------------------------------------------------------------------------------------------
def anomaly_components(model, prep, seed, neg_per_node=20):
    """Raw score components (numpy) for one model: attr_err, struct_err, att_only, feat_dev_uniform."""
    x, ei = prep["x"], prep["ei"]
    ae, se = S.recon_errors(model, x, ei, seed, neg_per_node)
    comp = {"attr_err": ae, "struct_err": se, "feat_dev_uniform": S.uniform_discrepancy(x, ei)}
    att = S.attention_edges(model, x, ei)
    if att is not None:
        comp["att_only"] = S.attention_only_score(x, ei, att)
    return comp, att


def build_scores(comp, train_idx, val_idx, y, alpha_fixed=None):
    """Combine components. alpha_sel chosen on validation AUC (grid) unless fixed. Returns scores, info."""
    za, zs = S.robust_z(comp["attr_err"], train_idx), S.robust_z(comp["struct_err"], train_idx)
    aucs = {}
    for a in ALPHA_GRID:
        r = a * za + (1 - a) * zs
        m = M.anomaly_metrics(y[val_idx], r[val_idx], 0.0)
        aucs[a] = m["roc_auc"]
    valid = {a: v for a, v in aucs.items() if v is not None}
    a_sel = alpha_fixed if alpha_fixed is not None else (max(valid, key=valid.get) if valid else 0.5)
    rec = a_sel * za + (1 - a_sel) * zs
    sc = {"attr_err": comp["attr_err"], "struct_err": comp["struct_err"], "rec": rec,
          "feat_dev_uniform": comp["feat_dev_uniform"]}
    if "att_only" in comp:
        sc["att_only"] = comp["att_only"]
        sc["combined"] = 0.5 * S.robust_z(rec, train_idx) + 0.5 * S.robust_z(comp["att_only"], train_idx)
    return sc, {"alpha_sel": float(a_sel), "val_auc_by_alpha": {str(k): v for k, v in aucs.items()},
                "z_ref": {"median_attr": float(np.median(comp["attr_err"][train_idx])),
                          "median_struct": float(np.median(comp["struct_err"][train_idx]))}}


def evaluate_anomaly(scores, y, splits, curves_for=("rec", "att_only", "combined", "iforest", "attr_err",
                                                    "feat_dev_uniform")):
    val, test = np.where(splits["val"])[0], np.where(splits["test"])[0]
    out, curves = {}, {}
    for name, s in scores.items():
        thr = M.select_threshold_f1(y[val], s[val])
        entry = {"threshold": thr}
        if thr is None:
            entry["note"] = "validation split is one-class: threshold undefined"
            thr_use = float(np.median(s))
        else:
            thr_use = thr
        entry["val"] = M.anomaly_metrics(y[val], s[val], thr_use)
        entry["test"] = M.anomaly_metrics(y[test], s[test], thr_use)
        k50 = M.anomaly_metrics(y[test], s[test], thr_use, k=50)
        entry["test"]["precision_at_50"] = k50["precision_at_k"]
        out[name] = entry
        if name in curves_for:
            curves[name] = M.pr_roc_curves(y[test], s[test])
    return out, curves


def pick_topk(score, idx, k, rng):
    jitter = rng.random(len(idx))
    order = np.lexsort((jitter, -score[idx]))
    return idx[order[:k]]


def evaluate_influence(prep, scores: dict, cfg_inf: dict, seed: int, log=print, with_sets=True):
    """Ranking metrics on held-out test nodes (+ seed-set spread). scores: name -> array over all nodes
    (or name@p keys for p-specific model predictions)."""
    test = np.where(prep["splits"]["test"])[0]
    b = prep["bundle"]
    res = {}
    rng = np.random.default_rng(seed)
    ks = tuple(cfg_inf.get("set_ks", (10, 20)))
    reps_set = int(cfg_inf.get("set_reps", 500))
    set_seed = 7_000_000 + seed
    for pi, p in enumerate(prep["p_grid"]):
        truth = prep["targets"][float(p)]["mean"][test]
        se = prep["targets"][float(p)]["se"][test]
        pres = {"truth_mean_spread": float(truth.mean()), "truth_median": float(np.median(truth)),
                "truth_max": float(truth.max()), "truth_mean_se": float(se.mean()), "methods": {}}
        # reference sets
        refs = {}
        if with_sets:
            for k in ks:
                oracle = pick_topk(prep["targets"][float(p)]["mean"], test, k, rng)
                m, s_, _ = I.seed_set_spread(b.edge_index, prep["n"], p, oracle, reps_set, set_seed)
                refs[f"oracle_top{k}"] = (m, s_)
                rs = []
                for _ in range(10):
                    r = rng.choice(test, size=k, replace=False)
                    rs.append(I.seed_set_spread(b.edge_index, prep["n"], p, r, max(reps_set // 5, 50), set_seed)[0])
                refs[f"random_{k}"] = (float(np.mean(rs)), float(np.std(rs, ddof=1) / np.sqrt(len(rs))))
        pres["reference_sets"] = {k: {"spread": v[0], "se": v[1]} for k, v in refs.items()}
        for name, s in scores.items():
            if "@" in name:
                base, pp = name.split("@")
                if float(pp) != float(p):
                    continue
                label = base
            else:
                label = name
            sv = np.asarray(s)[test]
            m = M.influence_metrics(truth, sv)
            if m["spearman"] is not None:
                m["spearman_ci95"] = M.bootstrap_metric_ci(
                    lambda idx: M.influence_metrics(truth[idx], sv[idx], ks=(), frac_ks=())["spearman"]
                    if np.ptp(sv[idx]) > 0 else None, len(test), n_boot=100, seed=seed)
            if with_sets:
                sets = {}
                for k in ks:
                    sel = pick_topk(np.asarray(s), test, k, rng)
                    mm, ss, _ = I.seed_set_spread(b.edge_index, prep["n"], p, sel, reps_set, set_seed)
                    sets[f"top{k}"] = {"spread": mm, "se": ss}
                m["seed_set_spread"] = sets
            pres["methods"][label] = m
        res[str(p)] = pres
    return res


# ----------------------------------------------------------------------------------------------
def _write_predictions(path, prep, scores_anom, scores_inf):
    b = prep["bundle"]
    cols = {"node": np.arange(prep["n"]),
            "split": np.select([prep["splits"]["train"], prep["splits"]["val"]], ["train", "val"], "test"),
            "y": b.y, "y_type": b.y_type, "degree": prep["deg"].astype(int)}
    for k, v in scores_anom.items():
        cols[f"anom_{k}"] = v
    for k, v in scores_inf.items():
        cols[f"inf_{k}"] = v
    for p in prep["p_grid"]:
        cols[f"spread_mean@{p}"] = prep["targets"][float(p)]["mean"]
        cols[f"spread_se@{p}"] = prep["targets"][float(p)]["se"]
    pd.DataFrame(cols).to_csv(path, index=False, compression="gzip")


def make_run_id(dataset, variant, seed, tag):
    return f"{dataset}__{variant}__s{seed}__{tag}"


def run_one(dataset: str, variant: str, seed: int, cfg: dict, tag="bench", out_root=None, command=None,
            force=False, log_to_console=True):
    out_root = Path(out_root) if out_root else RUNS_DIR
    run_id = make_run_id(dataset, variant, seed, tag)
    run_dir = out_root / run_id
    if (run_dir / "metrics.json").exists() and not force:
        print(f"[skip] {run_id} already complete")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir / "log.txt")
    t_start = time.time()
    try:
        _run_one(dataset, variant, seed, cfg, tag, run_dir, run_id, command, log)
    except Exception as e:  # noqa: BLE001
        import traceback
        log("FAILED: " + repr(e))
        log(traceback.format_exc())
        dump_json({"run_id": run_id, "status": "failed", "error": repr(e)}, run_dir / "FAILED.json")
        raise
    finally:
        log(f"total wall time {time.time() - t_start:.1f}s")
        log.close()
    return run_dir


def _run_one(dataset, variant, seed, cfg, tag, run_dir, run_id, command, log):
    torch.set_num_threads(cfg.get("threads", 4))
    dcfg = cfg["datasets"][dataset]
    log(f"run {run_id}")
    prep = prepare(dataset, seed, dcfg, log)
    b, n, splits = prep["bundle"], prep["n"], prep["splits"]
    y = b.y
    tr, va, te = (np.where(splits[k])[0] for k in ("train", "val", "test"))
    ei_np = b.edge_index
    # ---- provenance
    dump_json({"run_id": run_id, "tag": tag, "dataset": dataset, "variant": variant, "seed": seed,
               "config": {"dataset": dcfg, "model": cfg.get("model", {}), "variant": VARIANTS.get(variant),
                          "inference": cfg.get("influence_eval", {})},
               "alpha_grid": ALPHA_GRID, "started": now_iso()}, run_dir / "config.json")
    (run_dir / "command.txt").write_text(command or
        f"python -m attngnn.cli run --dataset {dataset} --variant {variant} --seed {seed} --tag {tag}\n")
    dump_json(environment_info(), run_dir / "env.json")
    summary = D.graph_summary(b)
    dump_json({"summary": summary, "split_sha256": splits["sha256"],
               "split_sizes": {k: int(splits[k].sum()) for k in ("train", "val", "test")},
               "split_anomalies": {k: int(y[splits[k]].sum()) for k in ("train", "val", "test")},
               "edge_index_sha256": sha(ei_np), "features_sha256": sha(b.x), "labels_sha256": sha(y),
               "influence_target_cache": prep["target_cache"],
               "transductive_note": "encoder sees all nodes' features and all edges; no val/test labels"},
              run_dir / "data.json")
    dump_json(code_identity(), run_dir / "code.json")
    metrics = {"run_id": run_id, "dataset": dataset, "variant": variant, "seed": seed, "tag": tag,
               "status": "complete"}

    if variant == "centrality":
        t0 = time.time()
        piv = dcfg.get("betweenness_pivots")
        scores_inf = {"degree": prep["deg"], "pagerank": I.pagerank(ei_np, n),
                      "uniform_recv_att": I.uniform_received_attention(ei_np, n)}
        log("computing betweenness (%s)" % ("exact" if piv is None else f"{piv} sampled pivots"))
        scores_inf["betweenness"] = I.betweenness(ei_np, n, piv, seed)
        metrics["centrality_seconds"] = time.time() - t0
        metrics["betweenness_pivots"] = piv
        metrics["influence"] = evaluate_influence(prep, scores_inf, cfg.get("influence_eval", {}), seed, log)
        _write_predictions(run_dir / "predictions.csv.gz", prep, {}, scores_inf)
        metrics["peak_memory_mb"] = peak_memory_mb()
        dump_json(metrics, run_dir / "metrics.json")
        return

    if variant.startswith("iforest"):
        from sklearn.ensemble import IsolationForest
        feats = prep["x"].numpy()
        if variant == "iforest_attr_struct":
            import scipy.sparse as sp
            a = sp.csr_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(n, n))
            deg = prep["deg"]
            nd = np.asarray(a @ deg).ravel() / np.maximum(deg, 1)
            pr = I.pagerank(ei_np, n) * n
            st = np.stack([np.log1p(deg), np.log1p(nd), np.log1p(pr)], 1)
            st = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-8)
            feats = np.concatenate([feats, st.astype(np.float32)], 1)
        t0 = time.time()
        iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(feats)
        s = -iso.score_samples(feats)
        metrics["train_seconds"] = time.time() - t0
        metrics["anomaly"], curves = evaluate_anomaly({"iforest": s}, y, splits)
        dump_json(curves, run_dir / "curves.json")
        _write_predictions(run_dir / "predictions.csv.gz", prep, {"iforest": s}, {})
        metrics["peak_memory_mb"] = peak_memory_mb()
        metrics["notes"] = ("IsolationForest(200 trees) fitted on all nodes' normalised attributes "
                            "(unlabelled, transductive) " +
                            ("plus log-degree, log mean-neighbour-degree, log PageRank" if variant.endswith("struct") else ""))
        dump_json(metrics, run_dir / "metrics.json")
        return

    # ---------------- GNN variants
    v = VARIANTS[variant]
    mcfg = {**cfg.get("model", {}), "encoder": v["encoder"], "heads": v["heads"]}
    tcfg = cfg.get("train", {})
    n_p = len(prep["p_grid"])
    model = build_model(mcfg, prep["x"].shape[1], n_p)
    log(f"model {variant}: encoder params={model.n_encoder_params()} total={model.n_params()}")
    inf_target, tstats = train_targets(prep, prep["p_grid"])
    epochs = 0 if v.get("untrained") else int(dcfg.get("epochs", tcfg.get("epochs", 100)))
    torch.manual_seed(seed)
    hist = train_model(model, prep["x"], prep["ei"], torch.from_numpy(tr), inf_target, epochs=epochs,
                       lr=tcfg.get("lr", 5e-3), weight_decay=tcfg.get("weight_decay", 0.0),
                       alpha=tcfg.get("alpha", 0.5), edge_keep=dcfg.get("edge_keep", 1.0),
                       lam_inf=v["lam_inf"], w_rec=v["w_rec"], neg_ratio=tcfg.get("neg_ratio", 1.0),
                       max_pos=tcfg.get("max_pos", 200_000), seed=seed, log=log)
    trained_rec = bool(v["w_rec"] > 0 and epochs > 0)
    trained_inf = bool(v["lam_inf"] > 0 and epochs > 0)
    metrics.update({"trained_rec": trained_rec, "trained_inf": trained_inf, "epochs": epochs,
                    "train_seconds": hist["train_seconds"], "final_losses": {k: hist[k][-1] for k in
                    ("loss", "attr", "struct", "inf") if hist[k]},
                    "n_params": model.n_params(), "n_encoder_params": model.n_encoder_params(),
                    "neg_sampling": {"ratio": hist["neg_ratio"], "seed": hist["neg_sampling_seed"],
                                     "max_pos_arcs_per_epoch": hist["max_pos"]},
                    "edge_keep_during_training": hist["edge_keep"]})
    dump_json({"hist": {k: hist[k] for k in ("loss", "attr", "struct", "inf")}}, run_dir / "history.json")

    # ---- checkpoint
    ck = {"state_dict": model.state_dict(), "model_cfg": model.cfg, "run_id": run_id, "dataset": dataset,
          "variant": variant, "seed": seed, "p_grid": prep["p_grid"], "target_stats": tstats,
          "norm": {"kind": prep["norm"].kind, "clip": prep["norm"].clip, "mean": prep["norm"].mean,
                   "std": prep["norm"].std}, "dataset_cfg": dcfg}
    torch.save(ck, run_dir / "checkpoint.pt")
    metrics["checkpoint_sha256"] = sha256_path(run_dir / "checkpoint.pt")

    # ---- inference timing + scores
    t0 = time.time()
    comp, att = anomaly_components(model, prep, seed)
    metrics["inference_seconds_anomaly_scores"] = time.time() - t0
    sc_anom, ainfo = build_scores(comp, tr, va, y)
    ainfo["trained_rec"] = trained_rec
    metrics["anomaly_selection"] = ainfo
    if not trained_rec:
        # reconstruction scores of an untrained decoder are not a trained detector: keep but flag
        metrics["anomaly_note"] = ("reconstruction head not trained in this variant: attr_err/struct_err/rec/"
                                   "combined are NOT valid detectors; att_only and feat_dev_uniform use the "
                                   "attention of this encoder")
    metrics["anomaly"], curves = evaluate_anomaly(sc_anom, y, splits)
    dump_json(curves, run_dir / "curves.json")

    sc_inf, iinfo = S.all_influence_scores(model, prep["x"], prep["ei"], tr, prep["p_grid"])
    metrics["influence_info"] = iinfo
    metrics["influence_note"] = None if trained_inf else (
        "influence head not trained in this variant: model_pred is NOT a valid predictor")
    if not trained_inf:
        sc_inf = {k: s for k, s in sc_inf.items() if not k.startswith("model_pred")}
    metrics["influence"] = evaluate_influence(prep, sc_inf, cfg.get("influence_eval", {}), seed, log)
    metrics["peak_memory_mb"] = peak_memory_mb()
    _write_predictions(run_dir / "predictions.csv.gz", prep, sc_anom, sc_inf)

    # ---- attention statistics (for the report)
    if att is not None:
        a_mean = S.mean_alpha(att)
        R = S.received_attention(att, n)
        deg = prep["deg"]
        from scipy import stats as st
        metrics["attention_stats"] = {
            "mean_alpha": float(a_mean.mean()),
            "spearman_received_attention_vs_degree": float(st.spearmanr(R, deg)[0]),
            "per_head_mean_entropy_ratio": _entropy_ratio(att, n),
            "uniform_baseline_mean_alpha": float((1.0 / (np.bincount(att["dst"], minlength=n) + 1)[att["dst"]]).mean()),
            "n_layers": int(att["alpha"].shape[1]), "n_heads": int(att["alpha"].shape[2])}

    # ---- explanation perturbation (association test)
    ecfg = cfg.get("explain", {})
    if v.get("explain") and ecfg.get("enabled", True) and att is not None and trained_rec:
        metrics["explanation"] = _explain(model, prep, att, ainfo, sc_anom, sc_inf, ecfg, seed, log, ei_np)

    dump_json(metrics, run_dir / "metrics.json")
    log("run complete")


def _entropy_ratio(att, n):
    """Mean normalised attention entropy (1 = uniform) per layer, over nodes with >=2 neighbours."""
    a = att["alpha"].mean(2)  # [E, L]
    dst = att["dst"]
    deg = np.bincount(dst, minlength=n)
    out = []
    for l in range(a.shape[1]):
        w = a[:, l]
        den = np.bincount(dst, weights=w, minlength=n)
        pr = w / np.maximum(den[dst], 1e-12)
        h = -np.bincount(dst, weights=pr * np.log(np.maximum(pr, 1e-12)), minlength=n)
        ok = deg >= 2
        out.append(float((h[ok] / np.log(deg[ok])).mean()))
    return out


def _explain(model, prep, att, ainfo, sc_anom, sc_inf, ecfg, seed, log, ei_np):
    x, ei, n = prep["x"], prep["ei"], prep["n"]
    y = prep["bundle"].y
    te = np.where(prep["splits"]["test"])[0]
    tr = np.where(prep["splits"]["train"])[0]
    a_sel = ainfo["alpha_sel"]
    ae0, se0 = S.recon_errors(model, x, ei, seed)
    ref = {"ma": np.median(ae0[tr]), "da": 1.4826 * np.median(np.abs(ae0[tr] - np.median(ae0[tr]))) or 1.0,
           "ms": np.median(se0[tr]), "ds": 1.4826 * np.median(np.abs(se0[tr] - np.median(se0[tr]))) or 1.0}

    @torch.no_grad()
    def anomaly_out(x_, ei_, node):
        ae, se = S.recon_errors(model, x_, ei_, seed)
        return float(a_sel * (ae[node] - ref["ma"]) / ref["da"] + (1 - a_sel) * (se[node] - ref["ms"]) / ref["ds"])

    pi = len(prep["p_grid"]) // 2

    @torch.no_grad()
    def influence_out(x_, ei_, node):
        return float(model(x_, ei_)["inf"][node, pi])

    a_mean = S.mean_alpha(att)
    rng = np.random.default_rng(seed + 99)
    nT = int(ecfg.get("n_targets", 20))
    out = {"edge_budget_m": int(ecfg.get("m", 3)), "n_random_draws": int(ecfg.get("n_random", 3)),
           "note": "association test: effect = output_before - output_after removing arcs; model frozen"}
    anomalies = te[y[te] == 1]
    cand = anomalies[prep["deg"][anomalies] >= 2]
    targets = rng.choice(cand, size=min(nT, len(cand)), replace=False) if len(cand) else []
    t0 = time.time()
    rows = run_perturbation(anomaly_out, x, ei, a_mean, targets, "in", n, m=out["edge_budget_m"],
                            seed=seed, n_random=out["n_random_draws"])
    out["anomaly"] = {"rows": rows, **summarise_perturbation(rows, seed)}
    pred_key = f"model_pred@{prep['p_grid'][pi]}"
    if pred_key in sc_inf:
        order = te[np.argsort(-sc_inf[pred_key][te])]
        order = order[prep["deg"][order] >= 2][:nT]
        rows = run_perturbation(influence_out, x, ei, a_mean, order, "out", n, m=out["edge_budget_m"],
                                seed=seed, n_random=out["n_random_draws"])
        out["influence"] = {"rows": rows, **summarise_perturbation(rows, seed)}
    out["seconds"] = time.time() - t0
    log(f"explanation perturbation done in {out['seconds']:.1f}s")
    return out
