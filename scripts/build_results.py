"""Rebuild every results table and figure from saved run artifacts.

    python scripts/build_results.py            # reads artifacts/runs/*__bench, writes artifacts/results/

Only runs tagged ``bench`` are used (smoke / trial / ui runs are ignored).  Means and standard deviations
are recomputed here from the per-seed values in each run's metrics.json (sample std, ddof=1).  The script
refuses to write a table containing non-finite numbers, and marks missing combinations 'not run'.
"""
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attngnn import RUNS_DIR
from attngnn import inference as inf
from attngnn.runs import dump_json, sha256_path, code_identity

OUT = ROOT / "artifacts" / "results"
TAB = OUT / "tables"
FIG = OUT / "figures"
NAVY, TEAL, ORANGE, GREY, SAND = "#0b2545", "#13808b", "#d9822b", "#8d99ae", "#e9c46a"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

INDEX = {"tables": {}, "figures": {}}


# ----------------------------------------------------------------------------------------- loading
def load_runs():
    runs = defaultdict(dict)   # (dataset, variant) -> seed -> metrics
    dirs = {}
    for d in inf.list_runs(RUNS_DIR):
        m = inf.load_metrics(d)
        if m.get("tag") != "bench":
            continue
        runs[(m["dataset"], m["variant"])][m["seed"]] = m
        dirs[(m["dataset"], m["variant"], m["seed"])] = d
    return runs, dirs


def ms(vals, nd=3):
    v = [x for x in vals if x is not None]
    if not v:
        return "not run"
    a = np.array(v, float)
    if not np.isfinite(a).all():
        raise ValueError(f"non-finite value in {vals}")
    sd = a.std(ddof=1) if len(a) > 1 else float("nan")
    s = f"{a.mean():.{nd}f}" + (f" ± {sd:.{nd}f}" if np.isfinite(sd) else " (n=1)")
    return s + f" [n={len(a)}]"


def vals(runs, ds, var, getter):
    out = {}
    for sd, m in sorted(runs.get((ds, var), {}).items()):
        try:
            out[sd] = getter(m)
        except (KeyError, TypeError):
            out[sd] = None
    return out


def check_finite(df, name):
    for c in df.columns:
        for v in df[c]:
            if isinstance(v, float) and not np.isfinite(v):
                raise ValueError(f"{name}: non-finite value in column {c}")
            if isinstance(v, str) and v.lower() in ("nan", "inf", "placeholder", "tbd", "todo"):
                raise ValueError(f"{name}: placeholder/non-finite string in column {c}")


def save_table(df, name, sources):
    check_finite(df, name)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / name, index=False)
    INDEX["tables"][name] = {"source_runs": sorted(sources), "built_by": "scripts/build_results.py"}


def save_fig(fig, name, sources):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    INDEX["figures"][name] = {"source_runs": sorted(sources), "built_by": "scripts/build_results.py"}


def run_ids(dirs, ds, vars_):
    return [d.name for (a, b, c), d in dirs.items() if a == ds and b in vars_]


# ----------------------------------------------------------------------------------------- tables
ANOM_ROWS = [  # label, variant, score, note
    ("Isolation Forest (attributes)", "iforest_attr", "iforest", "baseline"),
    ("Isolation Forest (attributes + structure)", "iforest_attr_struct", "iforest", "baseline"),
    ("GCN reconstruction (DOMINANT-style)", "gcn_anom", "rec", "baseline"),
    ("GAT reconstruction", "gat_anom", "rec", "ours"),
    ("GATv2 reconstruction", "gatv2_anom", "rec", "ours"),
    ("GAT attention-only (weighted feature discrepancy)", "gat_anom", "att_only", "ours"),
    ("GAT combined (reconstruction + attention-only)", "gat_anom", "combined", "ours"),
    ("GAT joint, reconstruction score", "gat_joint", "rec", "ours"),
    ("GAT joint, attention-only score", "gat_joint", "att_only", "ours"),
    ("GAT joint, combined score", "gat_joint", "combined", "ours"),
    ("GCN joint, reconstruction score", "gcn_joint", "rec", "ours"),
    ("GATv2 joint, reconstruction score", "gatv2_joint", "rec", "ours"),
    ("Control: uniform-neighbour feature deviation (no attention)", "gat_anom", "feat_dev_uniform", "control"),
    ("Control: untrained-GAT attention-only", "gat_untrained", "att_only", "control"),
]
ANOM_COLS = [("roc_auc", "ROC-AUC"), ("average_precision", "AP"), ("f1", "F1"), ("precision", "Precision"),
             ("recall", "Recall"), ("precision_at_k", "P@k (k=#pos)")]


def get_a(score, key):
    return lambda m: m["anomaly"][score]["test"][key]


def anomaly_tables(runs, dirs, ds):
    rows, per_seed = [], []
    for label, var, score, kind in ANOM_ROWS:
        row = {"method": label, "kind": kind}
        seeds = sorted(runs.get((ds, var), {}))
        row["seeds"] = len(seeds)
        for key, nm in ANOM_COLS:
            v = vals(runs, ds, var, get_a(score, key))
            row[nm] = ms(v.values()) if seeds else "not run"
            for sd, x in v.items():
                per_seed.append({"method": label, "seed": sd, "metric": nm, "value": x})
        rows.append(row)
    df = pd.DataFrame(rows)
    save_table(df, f"anomaly_{ds}.csv", run_ids(dirs, ds, {r[1] for r in ANOM_ROWS}))
    ps = pd.DataFrame(per_seed)
    if len(ps):
        ps2 = ps.dropna()
        save_table(ps2, f"anomaly_per_seed_{ds}.csv", run_ids(dirs, ds, {r[1] for r in ANOM_ROWS}))
        pv = ps2[ps2.metric == "ROC-AUC"].pivot_table(index="method", columns="seed", values="value", sort=False)
        pv.columns = [f"seed {c}" for c in pv.columns]
        pv = pv.round(3).reset_index().rename(columns={"method": "method (test ROC-AUC per seed)"})
        save_table(pv, f"anomaly_auc_by_seed_{ds}.csv", run_ids(dirs, ds, {r[1] for r in ANOM_ROWS}))
    return df, ps


INF_ROWS = [  # label, variant, method key (in metrics['influence'][p]['methods'])
    ("Degree", "centrality", "degree"), ("PageRank", "centrality", "pagerank"),
    ("Betweenness", "centrality", "betweenness"),
    ("Uniform-attention control (non-learned)", "centrality", "uniform_recv_att"),
    ("GAT received attention (raw), anomaly-only training", "gat_anom", "recv_att"),
    ("GAT received attention (raw), joint", "gat_joint", "recv_att"),
    ("GAT received attention (degree-adjusted), joint", "gat_joint", "recv_att_adj"),
    ("GAT received attention / degree, joint", "gat_joint", "recv_att_per_deg"),
    ("GATv2 received attention (raw), joint", "gatv2_joint", "recv_att"),
    ("GAT influence head, joint", "gat_joint", "model_pred"),
    ("GAT influence head, influence-only", "gat_inf", "model_pred"),
    ("GCN influence head, joint", "gcn_joint", "model_pred"),
    ("GATv2 influence head, joint", "gatv2_joint", "model_pred"),
    ("GAT 1-head influence head, joint", "gat1h_joint", "model_pred"),
    ("Control: untrained-GAT received attention", "gat_untrained", "recv_att"),
]


def primary_p(runs, ds):
    for (d, v), s in runs.items():
        if d == ds:
            m = next(iter(s.values()))
            if "influence" in m:
                ks = list(m["influence"].keys())
                return ks[len(ks) // 2], ks
    return None, []


def influence_tables(runs, dirs, ds):
    p0, plist = primary_p(runs, ds)
    if p0 is None:
        return None, None
    allrows = []
    for p in plist:
        rows = []
        for label, var, key in INF_ROWS:
            seeds = sorted(runs.get((ds, var), {}))
            row = {"method": label, "p": p, "seeds": len(seeds)}

            def g(metric):
                return vals(runs, ds, var, lambda m, metric=metric: m["influence"][p]["methods"][key][metric])

            def gs(kk):
                return vals(runs, ds, var, lambda m: m["influence"][p]["methods"][key]["seed_set_spread"][kk]["spread"])
            for metric, nm in [("spearman", "Spearman"), ("kendall", "Kendall tau"), ("ndcg@50", "NDCG@50"),
                               ("topk_overlap@50", "Top-50 overlap")]:
                row[nm] = ms(g(metric).values()) if seeds else "not run"
            for kk in ("top10", "top20"):
                row[f"Seed-set spread {kk}"] = ms(gs(kk).values(), 1) if seeds else "not run"
            rows.append(row)
        # reference sets
        ref = {}
        for var in ("centrality",):
            for kk in ("oracle_top10", "random_10", "oracle_top20", "random_20"):
                ref[kk] = vals(runs, ds, var, lambda m, kk=kk: m["influence"][p]["reference_sets"][kk]["spread"])
        rows.append({"method": "Reference: random seed set (mean of 10 draws)", "p": p, "seeds": len(ref["random_10"]),
                     "Spearman": "-", "Kendall tau": "-", "NDCG@50": "-", "Top-50 overlap": "-",
                     "Seed-set spread top10": ms(ref["random_10"].values(), 1),
                     "Seed-set spread top20": ms(ref["random_20"].values(), 1)})
        rows.append({"method": "Reference: oracle top-k by true individual spread", "p": p, "seeds": len(ref["oracle_top10"]),
                     "Spearman": "-", "Kendall tau": "-", "NDCG@50": "-", "Top-50 overlap": "-",
                     "Seed-set spread top10": ms(ref["oracle_top10"].values(), 1),
                     "Seed-set spread top20": ms(ref["oracle_top20"].values(), 1)})
        allrows += rows
    df = pd.DataFrame(allrows)
    src = run_ids(dirs, ds, {r[1] for r in INF_ROWS})
    save_table(df[df.p == p0].drop(columns="p"), f"influence_{ds}.csv", src)
    save_table(df, f"influence_by_p_{ds}.csv", src)
    return df, p0


def ablation_table(runs, dirs, ds):
    p0, _ = primary_p(runs, ds)
    rows = []

    def A(var, score, key="roc_auc"):
        if not runs.get((ds, var)):
            return "not run"
        if score in ("rec", "combined") and not all(m.get("trained_rec", True) for m in runs[(ds, var)].values()):
            return "n/a (head untrained)"
        return ms(vals(runs, ds, var, get_a(score, key)).values())

    def Iv(var, key, metric="spearman"):
        return ms(vals(runs, ds, var, lambda m: m["influence"][p0]["methods"][key][metric]).values()) \
            if runs.get((ds, var)) else "not run"

    def P(var, nm):
        return ms(vals(runs, ds, var, lambda m: m[nm]).values(), 0) if runs.get((ds, var)) else "not run"

    spec = [
        ("1. Encoder family (reconstruction only)", [("GCN", "gcn_anom"), ("GAT", "gat_anom"), ("GATv2", "gatv2_anom")], "rec", None),
        ("1b. Encoder family (joint training)", [("GCN", "gcn_joint"), ("GAT", "gat_joint"), ("GATv2", "gatv2_joint")], "rec", "model_pred"),
        ("2. Heads (joint training)", [("1 head x 64", "gat1h_joint"), ("4 heads x 16", "gat_joint")], "rec", "model_pred"),
        ("4. Training mode (GAT)", [("anomaly-only", "gat_anom"), ("influence-only", "gat_inf"), ("joint", "gat_joint")], "rec", "model_pred"),
    ]
    for group, items, score, ikey in spec:
        for lab, var in items:
            rows.append({"ablation": group, "configuration": lab, "variant": var,
                         "Anomaly ROC-AUC": A(var, score),
                         "Anomaly AP": A(var, score, "average_precision"),
                         "Influence Spearman (head)": Iv(var, ikey) if ikey and var != "gat_anom" else "n/a",
                         "Influence Spearman (recv. attention)": Iv(var, "recv_att") if var != "gcn_joint" and var != "gcn_anom" else "n/a",
                         "Encoder params": P(var, "n_encoder_params"), "Total params": P(var, "n_params"),
                         "seeds": len(runs.get((ds, var), {}))})
    for lab, score in [("reconstruction-only score", "rec"), ("attention-only score", "att_only"),
                       ("combined score (0.5 / 0.5)", "combined"), ("uniform-neighbour deviation (control)", "feat_dev_uniform")]:
        for var in ("gat_anom", "gat_joint"):
            rows.append({"ablation": "3. Anomaly score type", "configuration": f"{lab} [{var}]", "variant": var,
                         "Anomaly ROC-AUC": A(var, score), "Anomaly AP": A(var, score, "average_precision"),
                         "Influence Spearman (head)": "n/a", "Influence Spearman (recv. attention)": "n/a",
                         "Encoder params": P(var, "n_encoder_params"), "Total params": P(var, "n_params"),
                         "seeds": len(runs.get((ds, var), {}))})
    for lab, key in [("received attention, raw", "recv_att"), ("degree-adjusted (OLS residual of log R)", "recv_att_adj"),
                     ("received attention / degree", "recv_att_per_deg")]:
        rows.append({"ablation": "5. Degree adjustment (GAT joint)", "configuration": lab, "variant": "gat_joint",
                     "Anomaly ROC-AUC": "n/a", "Anomaly AP": "n/a",
                     "Influence Spearman (head)": "n/a", "Influence Spearman (recv. attention)": Iv("gat_joint", key),
                     "Encoder params": P("gat_joint", "n_encoder_params"), "Total params": P("gat_joint", "n_params"),
                     "seeds": len(runs.get((ds, "gat_joint"), {}))})
    df = pd.DataFrame(rows)
    save_table(df, f"ablation_{ds}.csv", run_ids(dirs, ds, {"gcn_anom", "gat_anom", "gatv2_anom", "gat_inf", "gcn_joint",
                                                              "gat_joint", "gatv2_joint", "gat1h_joint", "gat_untrained"}))
    return df


def explanation_table(runs, dirs, ds):
    rows = []
    for mode in ("anomaly", "influence"):
        per_seed = []
        pooled = {"high": [], "random": [], "low": []}
        for sd, m in sorted(runs.get((ds, "gat_joint"), {}).items()):
            e = (m.get("explanation") or {}).get(mode)
            if not e or not e.get("n_targets"):
                continue
            per_seed.append((sd, e))
            for r in e["rows"]:
                for k in pooled:
                    pooled[k].append(r[k])
        if not per_seed:
            continue
        hi, rd, lo = (np.array(pooled[k]) for k in ("high", "random", "low"))
        rng = np.random.default_rng(0)

        def ci(d):
            b = rng.choice(d, size=(2000, len(d))).mean(1)
            return f"[{np.quantile(b, .025):+.4f}, {np.quantile(b, .975):+.4f}]"
        from scipy import stats
        try:
            pw = stats.wilcoxon(hi, rd, alternative="greater").pvalue
        except ValueError:
            pw = float("nan")
        rows.append({"output": mode, "seeds": len(per_seed), "targets (pooled)": len(hi),
                     "mean effect: remove high-attention": f"{hi.mean():+.4f}",
                     "mean effect: remove random": f"{rd.mean():+.4f}",
                     "mean effect: remove low-attention": f"{lo.mean():+.4f}",
                     "high - random (95% bootstrap CI over targets)": f"{(hi - rd).mean():+.4f} {ci(hi - rd)}",
                     "high - low (95% CI)": f"{(hi - lo).mean():+.4f} {ci(hi - lo)}",
                     "share of targets high > random": f"{(hi > rd).mean():.2f}",
                     "Wilcoxon p (high > random)": f"{pw:.3g}" if np.isfinite(pw) else "n/a",
                     "per-seed mean(high - random)": ", ".join(f"s{sd}: {np.mean([r['high'] - r['random'] for r in e['rows']]):+.4f}" for sd, e in per_seed)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    save_table(df, f"explanation_{ds}.csv", run_ids(dirs, ds, {"gat_joint"}))
    return df


def resource_table(runs, dirs, ds):
    rows = []
    for (d, var), s in sorted(runs.items()):
        if d != ds:
            continue
        def g(k):
            return ms([m.get(k) for m in s.values()], 1)
        rows.append({"variant": var, "seeds": len(s), "train seconds": g("train_seconds") if var != "centrality" else "-",
                     "anomaly-score inference seconds": g("inference_seconds_anomaly_scores") if "gnn" in var or "gat" in var or "gcn" in var else "-",
                     "centrality seconds": g("centrality_seconds") if var == "centrality" else "-",
                     "process peak working set, MB (cumulative upper bound)": g("peak_memory_mb"),
                     "total params": g("n_params") if ("gat" in var or "gcn" in var) else "-"})
    df = pd.DataFrame(rows)
    save_table(df, f"resources_{ds}.csv", run_ids(dirs, ds, {v for (d, v) in runs if d == ds}))
    return df



def profile_table(ds):
    """Isolated single-run profiles (tag 'prof', fresh process per run): the only valid per-run peak memory."""
    rows, srcs = [], []
    for d in inf.list_runs(RUNS_DIR):
        m = inf.load_metrics(d)
        if m.get("tag") == "prof" and m["dataset"] == ds:
            srcs.append(d.name)
            rows.append({"variant": m["variant"], "seed": m["seed"], "train seconds": round(m.get("train_seconds") or 0, 1),
                         "anomaly-score inference seconds": round(m.get("inference_seconds_anomaly_scores") or 0, 2),
                         "peak working set, MB (isolated process)": m.get("peak_memory_mb"), "total params": m.get("n_params")})
    if rows:
        save_table(pd.DataFrame(rows).sort_values("variant"), f"resources_profile_{ds}.csv", srcs)



def paired_table(runs, dirs, ds):
    """Seed-paired differences (same seed = same graph, split and cascade streams)."""
    p0, _ = primary_p(runs, ds)

    def A(var, score, key="roc_auc"):
        return lambda m: m["anomaly"][score]["test"][key]

    def I_(meth, key="spearman"):
        return lambda m: m["influence"][p0]["methods"][meth][key]
    comps = [
        ("Anomaly ROC-AUC: GAT attention-only minus uniform-neighbour control", "gat_anom", A("gat_anom", "att_only"), "gat_anom", A("gat_anom", "feat_dev_uniform")),
        ("Anomaly ROC-AUC: trained GAT attention-only minus untrained-GAT attention-only", "gat_anom", A("x", "att_only"), "gat_untrained", A("x", "att_only")),
        ("Anomaly ROC-AUC: GAT reconstruction minus GCN reconstruction", "gat_anom", A("x", "rec"), "gcn_anom", A("x", "rec")),
        ("Anomaly ROC-AUC: GATv2 reconstruction minus GAT reconstruction", "gatv2_anom", A("x", "rec"), "gat_anom", A("x", "rec")),
        ("Anomaly ROC-AUC: GAT joint rec. minus GAT anomaly-only rec.", "gat_joint", A("x", "rec"), "gat_anom", A("x", "rec")),
        ("Anomaly ROC-AUC: GAT combined minus GAT reconstruction (anomaly-only models)", "gat_anom", A("x", "combined"), "gat_anom", A("x", "rec")),
        ("Anomaly ROC-AUC: GAT reconstruction minus Isolation Forest (attributes)", "gat_anom", A("x", "rec"), "iforest_attr", A("x", "iforest")),
        ("Anomaly ROC-AUC: 4-head minus 1-head GAT (joint, reconstruction)", "gat_joint", A("x", "rec"), "gat1h_joint", A("x", "rec")),
        ("Influence Spearman: GAT received attention minus uniform-attention control", "gat_joint", I_("recv_att"), "centrality", I_("uniform_recv_att")),
        ("Influence Spearman: GAT received attention minus degree", "gat_joint", I_("recv_att"), "centrality", I_("degree")),
        ("Influence Spearman: degree-adjusted minus raw received attention", "gat_joint", I_("recv_att_adj"), "gat_joint", I_("recv_att")),
        ("Influence Spearman: GCN head minus GAT head (joint)", "gcn_joint", I_("model_pred"), "gat_joint", I_("model_pred")),
        ("Influence Spearman: GAT head joint minus influence-only", "gat_joint", I_("model_pred"), "gat_inf", I_("model_pred")),
        ("Influence Spearman: GCN head minus degree", "gcn_joint", I_("model_pred"), "centrality", I_("degree")),
        ("Influence NDCG@50: GAT received attention minus degree", "gat_joint", I_("recv_att", "ndcg@50"), "centrality", I_("degree", "ndcg@50")),
    ]
    rows = []
    for name, v1, g1, v2, g2 in comps:
        r1, r2 = runs.get((ds, v1), {}), runs.get((ds, v2), {})
        common = sorted(set(r1) & set(r2))
        diffs = []
        for sd in common:
            try:
                diffs.append(g1(r1[sd]) - g2(r2[sd]))
            except (KeyError, TypeError):
                pass
        if not diffs:
            continue
        d = np.array(diffs)
        rows.append({"comparison (paired by seed)": name, "seeds": len(d), "mean difference": f"{d.mean():+.3f}",
                     "per-seed range": f"[{d.min():+.3f}, {d.max():+.3f}]",
                     "seeds with positive difference": f"{int((d > 0).sum())}/{len(d)}"})
    if rows:
        save_table(pd.DataFrame(rows), f"paired_{ds}.csv", run_ids(dirs, ds, {c[1] for c in comps} | {c[3] for c in comps}))



def label_version_table(runs, dirs):
    """Weibo anomaly metrics with the AUTHENTIC labels (868) vs the superseded runs that used the corrupted 2024 upstream
    labels (347 labelled; validation/test labels blanked). Shows how much a silent label problem changes the conclusions."""
    sup = ROOT / "artifacts" / "superseded" / "weibo_2024_masked_labels"
    if not sup.exists() or ("weibo", "gat_anom") not in runs:
        return
    rows = []
    for label, var, score in [("Isolation Forest (attributes)", "iforest_attr", "iforest"), ("GCN reconstruction", "gcn_anom", "rec"),
                              ("GAT reconstruction", "gat_anom", "rec"), ("GAT attention-only", "gat_anom", "att_only"),
                              ("GAT combined", "gat_anom", "combined"), ("Uniform-neighbour control", "gat_anom", "feat_dev_uniform")]:
        row = {"method": label}
        for tag, getter in (("authentic labels (868 anomalies)", lambda sd: runs[("weibo", var)][sd]),
                            ("corrupted 2024 labels (347 labelled)", lambda sd: json.loads((sup / f"weibo__{var}__s{sd}__bench" / "metrics.json").read_text()))):
            for key, nm in (("roc_auc", "ROC-AUC"), ("average_precision", "AP")):
                vals_ = []
                for sd in (0, 1, 2):
                    try:
                        vals_.append(getter(sd)["anomaly"][score]["test"][key])
                    except (KeyError, FileNotFoundError):
                        pass
                row[f"{nm}, {tag}"] = ms(vals_) if vals_ else "not run"
        rows.append(row)
    save_table(pd.DataFrame(rows), "label_version_weibo.csv", run_ids(dirs, "weibo", {v for _, v, _ in
               [("", "iforest_attr", ""), ("", "gcn_anom", ""), ("", "gat_anom", "")]}))
    INDEX["tables"]["label_version_weibo.csv"]["superseded_runs_dir"] = "artifacts/superseded/weibo_2024_masked_labels"


def dataset_table(runs, dirs):
    rows = []
    for ds in sorted({d for d, _ in runs}):
        for var in ("centrality", "iforest_attr", "gat_joint"):
            if runs.get((ds, var)):
                sd = sorted(runs[(ds, var)])[0]
                dd = json.loads((dirs[(ds, var, sd)] / "data.json").read_text())
                s = dd["summary"]
                rows.append({"dataset": ds, "nodes": s["nodes"], "undirected edges": s["undirected_edges"],
                             "features": s["features"], "anomalies": s.get("anomalies"),
                             "anomaly rate": f"{s.get('anomaly_rate', 0):.3%}", "mean degree": f"{s['mean_degree']:.1f}",
                             "max degree": s["max_degree"], "split sizes (train/val/test)": "/".join(str(dd['split_sizes'][k]) for k in ('train', 'val', 'test')),
                             "anomalies per split (seed %d)" % sd: "/".join(str(dd['split_anomalies'][k]) for k in ('train', 'val', 'test')),
                             "label provenance": s.get("label_provenance")})
                break
    df = pd.DataFrame(rows)
    if len(df):
        save_table(df, "datasets.csv", [d.name for (a, b, c), d in dirs.items() if b == "centrality"])
    return df


# ----------------------------------------------------------------------------------------- figures
def curves_figure(runs, dirs, ds):
    series = [("IForest (attr)", "iforest_attr", "iforest", GREY), ("GCN rec (DOMINANT-style)", "gcn_anom", "rec", ORANGE),
              ("GAT rec", "gat_anom", "rec", NAVY), ("GAT attention-only", "gat_anom", "att_only", TEAL),
              ("GAT combined", "gat_anom", "combined", "#7b2cbf")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    used = []
    for lab, var, score, col in series:
        sd = sorted(runs.get((ds, var), {}))
        if not sd:
            continue
        d = dirs[(ds, var, sd[0])]
        cv = json.loads((d / "curves.json").read_text()).get(score)
        if not cv:
            continue
        used.append(d.name)
        ax[0].plot(cv["fpr"], cv["tpr"], color=col, label=lab, lw=1.6)
        ax[1].plot(cv["recall"], cv["precision"], color=col, label=lab, lw=1.6)
    ax[0].plot([0, 1], [0, 1], ":", color="#999")
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC (test, seed 0)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test, seed 0)")
    ax[0].legend(fontsize=7, loc="lower right")
    save_fig(fig, f"roc_pr_{ds}.png", used)


def confusion_figure(runs, dirs, ds):
    sd = sorted(runs.get((ds, "gat_anom"), {}))
    if not sd:
        return
    m = runs[(ds, "gat_anom")][sd[0]]
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, score, title in zip(axs, ("rec", "att_only", "combined"), ("Reconstruction", "Attention-only", "Combined")):
        c = m["anomaly"][score]["test"]["confusion"]
        z = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(z, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(z[i, j]), ha="center", va="center", color="white" if z[i, j] > z.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred normal", "pred anomaly"], fontsize=7)
        ax.set_yticks([0, 1], ["true normal", "true anomaly"], fontsize=7)
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"Test confusion matrices at validation-selected thresholds (GAT, seed {sd[0]}, {ds})", fontsize=9)
    save_fig(fig, f"confusion_{ds}.png", [dirs[(ds, 'gat_anom', sd[0])].name])


def bars_figure(df, col, title, name, ds, sources, ylabel, highlight=("GAT",), ncol_label="method", xlim=None):
    labels, means, sds, colors = [], [], [], []
    for _, r in df.iterrows():
        s = r[col]
        if not isinstance(s, str) or s in ("not run", "-", "n/a"):
            continue
        parts = s.split(" ")
        mu = float(parts[0])
        sd = float(parts[2]) if len(parts) > 2 and parts[1] == "±" else 0.0
        labels.append(r[ncol_label])
        means.append(mu)
        sds.append(sd)
        kind = r.get("kind", "")
        colors.append(GREY if kind == "baseline" else (SAND if kind == "control" else TEAL if "attention" in r[ncol_label].lower() else NAVY))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, xerr=sds, color=colors, ecolor="#333", capsize=2)
    ax.set_yticks(y, [l if len(l) < 55 else l[:52] + "..." for l in labels], fontsize=7.5)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=10)
    if xlim:
        lo = xlim[0]
        if lo == 0.5 and min(means) < 0.52:
            lo = round(min(means) - 0.06, 2)
        ax.set_xlim(lo, xlim[1])
        if lo <= 0.5:
            ax.axvline(0.5, color="#444", ls=":", lw=1)
    from matplotlib.patches import Patch
    if "kind" in df.columns:
        ax.legend(handles=[Patch(color=GREY, label="baseline"), Patch(color=NAVY, label="our GNN variants"),
                           Patch(color=TEAL, label="attention-based score"), Patch(color=SAND, label="control")],
                  fontsize=7, loc="lower right")
    save_fig(fig, name, sources)


def influence_figures(runs, dirs, ds, df, p0):
    sub = df[df.p == p0]
    src = run_ids(dirs, ds, {r[1] for r in INF_ROWS})
    bars_figure(sub[sub.Spearman != "-"], "Spearman", f"Influence ranking vs held-out simulated spread (p={p0})",
                f"influence_spearman_{ds}.png", ds, src, "Spearman rank correlation (mean ± std over seeds)")
    # seed-set spread
    sel = sub.copy()
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(sel) + 1.4))
    labs, vals_, errs, cols = [], [], [], []
    for _, r in sel.iterrows():
        s = r["Seed-set spread top10"]
        if not isinstance(s, str) or s == "not run":
            continue
        parts = s.split(" ")
        labs.append(r["method"])
        vals_.append(float(parts[0]))
        errs.append(float(parts[2]) if len(parts) > 2 and parts[1] == "±" else 0.0)
        cols.append(ORANGE if r["method"].startswith("Reference") else NAVY)
    y = np.arange(len(labs))[::-1]
    ax.barh(y, vals_, xerr=errs, color=cols, ecolor="#333", capsize=2)
    ax.set_yticks(y, [l if len(l) < 55 else l[:52] + "..." for l in labs], fontsize=7.5)
    ax.set_xlabel("Expected spread of the selected 10-node seed set (held-out cascades)")
    ax.set_title(f"Seed-set spread, top-10 from test candidates (p={p0})", fontsize=10)
    save_fig(fig, f"seedset_spread_{ds}.png", src)
    # sensitivity to p
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ps = sorted(df.p.unique(), key=float)
    for lab, col in [("Degree", GREY), ("PageRank", SAND), ("GAT received attention (raw), joint", TEAL),
                     ("GAT influence head, joint", NAVY), ("GAT received attention (degree-adjusted), joint", ORANGE)]:
        ys = []
        for p in ps:
            r = df[(df.p == p) & (df.method == lab)]
            s = r["Spearman"].iloc[0] if len(r) else "not run"
            ys.append(float(s.split(" ")[0]) if isinstance(s, str) and s not in ("not run", "-") else np.nan)
        ax.plot([float(p) for p in ps], ys, "o-", color=col, label=lab if len(lab) < 40 else lab[:37] + "...")
    ax.set(xlabel="cascade probability p", ylabel="Spearman vs held-out spread", title="Sensitivity to cascade probability")
    ax.legend(fontsize=6.5)
    save_fig(fig, f"influence_p_sensitivity_{ds}.png", src)


def scatter_figure(runs, dirs, ds):
    sd = sorted(runs.get((ds, "gat_joint"), {}))
    if not sd:
        return
    d = dirs[(ds, "gat_joint", sd[0])]
    p = pd.read_csv(d / "predictions.csv.gz")
    m = runs[(ds, "gat_joint")][sd[0]]
    ps = list(m["influence"].keys())
    pp = ps[len(ps) // 2]
    t = p[p.split == "test"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    for a, (col, lab) in zip(ax, [("inf_recv_att", "received attention (raw)"), ("inf_recv_att_adj", "degree-adjusted received attention"),
                                  (f"inf_model_pred@{pp}", "influence head output")]):
        a.scatter(t[col], t[f"spread_mean@{pp}"], s=4, alpha=.35, color=NAVY)
        a.set_yscale("log")
        a.set(xlabel=lab, ylabel="simulated spread (log)")
    fig.suptitle(f"Held-out test nodes, GAT joint seed {sd[0]}, p={pp} ({ds})", fontsize=9)
    save_fig(fig, f"influence_scatter_{ds}.png", [d.name])


def explanation_figure(df, ds, runs, dirs):
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, (_, r) in enumerate(df.iterrows()):
        vals_ = [float(r[f"mean effect: remove {k}"]) for k in ("high-attention", "random", "low-attention")]
        ax.bar(np.arange(3) + i * 0.38, vals_, 0.36, color=[NAVY, TEAL][i % 2], label=r["output"])
    ax.set_xticks(np.arange(3) + 0.19, ["high attention", "random", "low attention"])
    ax.set_ylabel("mean output drop after edge removal")
    ax.legend(title="output")
    ax.set_title("Perturbation test of attention as explanation", fontsize=10)
    save_fig(fig, f"explanation_{ds}.png", run_ids(dirs, ds, {"gat_joint"}))


def ablation_figure(df, ds, dirs):
    sub = df[df.ablation.str.startswith(("1.", "2.", "3.", "4."))].copy()
    sub = sub[sub["Anomaly ROC-AUC"].str.contains("±|n=", na=False)]
    sub["label"] = sub["ablation"].str.slice(0, 2) + sub["configuration"]
    bars_figure(sub.assign(method=sub["label"]), "Anomaly ROC-AUC", "Ablations: anomaly ROC-AUC (test)",
                f"ablation_auc_{ds}.png", ds, [], "ROC-AUC (mean ± std over seeds)", ncol_label="method")


def resource_figure(df, ds, dirs):
    d = df[df["train seconds"].astype(str).str.contains("±|n=")]
    if d.empty:
        return
    mu = [float(s.split(" ")[0]) for s in d["train seconds"]]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(d.variant, mu, color=TEAL)
    ax.set_xlabel("training wall-clock seconds (mean over seeds)")
    ax.set_title(f"Measured training cost ({ds}, CPU)", fontsize=10)
    save_fig(fig, f"resources_{ds}.png", run_ids(dirs, ds, set(d.variant)))


def attention_figure(runs, dirs, ds):
    rows = []
    for var in ("gat_joint", "gat_anom", "gatv2_joint"):
        for sd, m in sorted(runs.get((ds, var), {}).items()):
            a = m.get("attention_stats")
            if a:
                rows.append({"variant": var, "seed": sd, "mean attention entropy ratio L1": a["per_head_mean_entropy_ratio"][0],
                             "L2": a["per_head_mean_entropy_ratio"][-1],
                             "Spearman(received attention, degree)": a["spearman_received_attention_vs_degree"],
                             "mean alpha": a["mean_alpha"], "uniform-attention mean alpha": a["uniform_baseline_mean_alpha"]})
    if rows:
        df = pd.DataFrame(rows)
        agg = df.groupby("variant").agg(["mean"]).round(3)
        agg.columns = [c[0] for c in agg.columns]
        agg = agg.drop(columns=["seed"]).reset_index()
        save_table(agg, f"attention_stats_{ds}.csv", run_ids(dirs, ds, {"gat_joint", "gat_anom", "gatv2_joint"}))


def sensitivity_injection(dirs_all):
    p = RUNS_DIR
    rows = []
    for d in inf.list_runs(p):
        m = inf.load_metrics(d)
        if m.get("tag", "").startswith("inj") or (m.get("tag") == "bench" and m["dataset"] == "cora_injected"
                                                  and m["variant"] in ("gat_anom", "gcn_anom")):
            cfg = json.loads((d / "config.json").read_text())
            rate = cfg["config"]["dataset"]["data_args"]
            rows.append({"run": d.name, "variant": m["variant"], "seed": m["seed"], "struct_cliques": rate["struct_cliques"],
                         "n_contextual": rate["n_contextual"], **{f"{s}_auc": m["anomaly"][s]["test"]["roc_auc"] for s in ("rec", "att_only", "combined") if s in m["anomaly"]}})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["injection_rate"] = ((df.struct_cliques * 10 + df.n_contextual) / 2708).round(3)
    g = df.drop(columns=["run"]).groupby(["variant", "injection_rate"]).agg(["mean", "std", "count"]).round(3)
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    g = g[[c for c in g.columns if not c.startswith(("struct_cliques", "n_contextual", "seed"))]]
    g = g.astype(object).where(g.notna(), "n/a")
    save_table(g, "sensitivity_injection_rate.csv", df.run.tolist())
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for var, col in (("gat_anom", NAVY), ("gcn_anom", ORANGE)):
        for sc, ls in (("rec", "-"), ("att_only", "--")):
            s = g[g.variant == var]
            if f"{sc}_auc_mean" in s and len(s) and (s[f"{sc}_auc_mean"] != "n/a").all():
                ax.plot(s.injection_rate, s[f"{sc}_auc_mean"].astype(float), ls, marker="o", color=col, label=f"{var} {sc}")
    ax.set(xlabel="injection rate (fraction of nodes)", ylabel="test ROC-AUC", title="Sensitivity to injection rate (Cora)")
    ax.legend(fontsize=7)
    save_fig(fig, "sensitivity_injection_rate.png", df.run.tolist())


# ----------------------------------------------------------------------------------------- main
def main():
    runs, dirs = load_runs()
    if not runs:
        print("no benchmark runs found in", RUNS_DIR)
        return 1
    datasets = sorted({d for d, _ in runs})
    INDEX["datasets"] = datasets
    dataset_table(runs, dirs)
    for ds in datasets:
        adf, _ = anomaly_tables(runs, dirs, ds)
        if (ds, "gat_anom") in runs:
            curves_figure(runs, dirs, ds)
            confusion_figure(runs, dirs, ds)
            bars_figure(adf, "ROC-AUC", f"Anomaly detection, test ROC-AUC ({ds})", f"anomaly_auc_{ds}.png", ds,
                        run_ids(dirs, ds, {r[1] for r in ANOM_ROWS}), "ROC-AUC (mean ± std over seeds; axis starts at chance = 0.5)", ncol_label="method", xlim=(0.5, None))
            bars_figure(adf, "AP", f"Anomaly detection, test average precision ({ds})", f"anomaly_ap_{ds}.png", ds,
                        run_ids(dirs, ds, {r[1] for r in ANOM_ROWS}), "Average precision (mean ± std over seeds)", ncol_label="method")
        idf, p0 = influence_tables(runs, dirs, ds)
        if idf is not None:
            influence_figures(runs, dirs, ds, idf, p0)
            scatter_figure(runs, dirs, ds)
        abl = ablation_table(runs, dirs, ds)
        try:
            ablation_figure(abl, ds, dirs)
        except Exception as e:  # noqa: BLE001
            print("ablation figure skipped:", e)
        edf = explanation_table(runs, dirs, ds)
        explanation_figure(edf, ds, runs, dirs)
        rdf = resource_table(runs, dirs, ds)
        profile_table(ds)
        paired_table(runs, dirs, ds)
        resource_figure(rdf, ds, dirs)
        attention_figure(runs, dirs, ds)
    sensitivity_injection(dirs)
    label_version_table(runs, dirs)
    # completeness report
    comp = []
    for ds in datasets:
        for var in ["iforest_attr", "iforest_attr_struct", "centrality", "gcn_anom", "gat_anom", "gatv2_anom", "gat_inf",
                    "gcn_joint", "gat_joint", "gatv2_joint", "gat1h_joint", "gat_untrained"]:
            comp.append({"dataset": ds, "variant": var, "completed_seeds": sorted(runs.get((ds, var), {}).keys())})
    pd.DataFrame(comp).to_csv(TAB / "completeness.csv", index=False)
    INDEX.update({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "n_runs": len(dirs),
                  "run_ids": sorted(d.name for d in dirs.values()), "code": code_identity()["source_tree_sha256"],
                  "checkpoint_hashes": {d.name: runs[(a, b)][c].get("checkpoint_sha256") for (a, b, c), d in dirs.items()
                                        if runs[(a, b)][c].get("checkpoint_sha256")}})
    dump_json(INDEX, OUT / "results_index.json")
    print(f"built {len(INDEX['tables'])} tables and {len(INDEX['figures'])} figures from {len(dirs)} runs -> {OUT}")
    return 0



# ----------------------------------------------------------------------------------------- facts
def build_facts(runs, dirs):
    """Flat dict of headline numbers used verbatim by the report and slides ({{fact:key}})."""
    f = {}
    req_seeds = {"cora_injected": 3, "weibo": 3}
    missing = []
    for ds in sorted({d for d, _ in runs}):
        for var in ["iforest_attr", "centrality", "gcn_anom", "gat_anom", "gatv2_anom", "gat_inf", "gcn_joint",
                    "gat_joint", "gatv2_joint", "gat1h_joint", "gat_untrained"]:
            n = len(runs.get((ds, var), {}))
            f[f"{ds}.seeds.{var}"] = n
            if n < req_seeds.get(ds, 3):
                missing.append(f"{ds}/{var} ({n} seeds)")
        for var in {v for d, v in runs if d == ds}:
            seeds = sorted(runs[(ds, var)])
            m0 = runs[(ds, var)][seeds[0]]
            for score, e in (m0.get("anomaly") or {}).items():
                for key in ("roc_auc", "average_precision", "f1", "precision", "recall", "precision_at_k"):
                    v = [runs[(ds, var)][s]["anomaly"][score]["test"][key] for s in seeds]
                    v = [x for x in v if x is not None]
                    if v:
                        a = np.array(v, float)
                        f[f"{ds}.anom.{var}.{score}.{key}"] = f"{a.mean():.3f}"
                        f[f"{ds}.anom.{var}.{score}.{key}.pm"] = f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else f"{a.mean():.3f}"
            for p, pr in (m0.get("influence") or {}).items():
                for meth in pr["methods"]:
                    for key in ("spearman", "kendall", "ndcg@50", "topk_overlap@50"):
                        v = [runs[(ds, var)][s]["influence"][p]["methods"][meth][key] for s in seeds
                             if meth in runs[(ds, var)][s]["influence"][p]["methods"]]
                        v = [x for x in v if x is not None]
                        if v:
                            a = np.array(v, float)
                            f[f"{ds}.inf.{p}.{var}.{meth}.{key}"] = f"{a.mean():.3f}"
                            f[f"{ds}.inf.{p}.{var}.{meth}.{key}.pm"] = f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else f"{a.mean():.3f}"
    # seed-set spread facts at the primary cascade probability
    for ds in sorted({d for d, _ in runs}):
        p0, _pl = primary_p(runs, ds)
        if p0 is None:
            continue
        def ss(var, getter):
            v = [getter(m) for m in runs.get((ds, var), {}).values()]
            v = [x for x in v if x is not None]
            if not v:
                return None
            a = np.array(v, float)
            return f"{a.mean():.1f} ± {a.std(ddof=1):.1f}" if len(a) > 1 else f"{a.mean():.1f}"
        f[f"{ds}.seedset.oracle10"] = ss("centrality", lambda m: m["influence"][p0]["reference_sets"]["oracle_top10"]["spread"])
        f[f"{ds}.seedset.random10"] = ss("centrality", lambda m: m["influence"][p0]["reference_sets"]["random_10"]["spread"])
        for meth in ("degree", "pagerank", "betweenness", "uniform_recv_att"):
            f[f"{ds}.seedset.{meth}10"] = ss("centrality", lambda m, meth=meth: m["influence"][p0]["methods"][meth]["seed_set_spread"]["top10"]["spread"])
        for var, meth in (("gat_joint", "recv_att"), ("gat_joint", "model_pred"), ("gcn_joint", "model_pred")):
            f[f"{ds}.seedset.{var}.{meth}10"] = ss(var, lambda m, meth=meth: m["influence"][p0]["methods"][meth]["seed_set_spread"]["top10"]["spread"])
    f["n_runs"] = len(dirs)
    f["missing"] = "; ".join(missing)
    f["incomplete_banner"] = ("INCOMPLETE EXPERIMENTAL DRAFT: some required runs are missing (" + "; ".join(missing) + ")") if missing else ""
    f["abstract"] = "(abstract is finalised by the author step after results exist)"
    return f


if __name__ == "__main__":
    rc = main()
    runs, dirs = load_runs()
    if runs:
        facts = build_facts(runs, dirs)
        old = {}
        fp = OUT / "facts.json"
        if fp.exists():
            old = json.loads(fp.read_text())
        for k in ("abstract",):
            if k in old:
                facts[k] = old[k]
        dump_json(facts, fp)
        print(f"facts.json: {len(facts)} entries")
    sys.exit(rc)
