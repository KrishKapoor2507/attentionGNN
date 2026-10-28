"""Backend helpers for the Streamlit dashboard (kept out of the UI file so they can be tested)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ARTIFACTS, ROOT, RUNS_DIR
from . import data as D
from . import experiments as E
from . import inference as inf
from . import scoring as S

JOBS_DIR = ARTIFACTS / "jobs"
RESULTS_DIR = ARTIFACTS / "results"
REPORTS_DIR = ROOT / "reports"

ANOMALY_LABELS = {
    "rec": "Reconstruction (DOMINANT-style)", "att_only": "Attention-only (weighted feature discrepancy)",
    "combined": "Combined (rec + attention-only)", "attr_err": "Attribute error only",
    "struct_err": "Structure error only", "feat_dev_uniform": "Uniform-neighbour deviation (control)",
    "iforest": "Isolation Forest",
}
INFLUENCE_LABELS = {
    "model_pred": "Model influence head", "recv_att": "Received attention (raw)",
    "recv_att_adj": "Received attention (degree-adjusted)", "recv_att_per_deg": "Received attention / degree",
    "degree": "Degree", "pagerank": "PageRank", "betweenness": "Betweenness",
    "uniform_recv_att": "Uniform-attention control",
}


# ----------------------------------------------------------------------------- runs
def run_table(root=RUNS_DIR) -> pd.DataFrame:
    rows = []
    for d in inf.list_runs(root):
        m = inf.load_metrics(d)
        rows.append({"run_id": d.name, "dataset": m["dataset"], "variant": m["variant"], "seed": m["seed"],
                     "tag": m.get("tag"), "train_s": m.get("train_seconds"),
                     "has_checkpoint": (d / "checkpoint.pt").exists(), "path": str(d)})
    return pd.DataFrame(rows)


def provenance(run_dir) -> dict:
    run_dir = Path(run_dir)
    m = inf.load_metrics(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    env = json.loads((run_dir / "env.json").read_text()) if (run_dir / "env.json").exists() else {}
    return {"run_id": m["run_id"], "dataset": m["dataset"], "variant": m["variant"], "seed": m["seed"],
            "tag": m.get("tag"), "checkpoint_sha256": (m.get("checkpoint_sha256") or "n/a (no model)")[:16],
            "train_seconds": m.get("train_seconds"), "started": cfg.get("started"),
            "peak_memory_mb": m.get("peak_memory_mb"), "cpu": env.get("cpu"),
            "is_smoke": m.get("tag") == "smoke"}


def load_predictions(run_dir) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "predictions.csv.gz")


def sibling_run(run_dir, variant):
    """Run with the same dataset/seed/tag but another variant (e.g. the 'centrality' baseline run)."""
    run_dir = Path(run_dir)
    ds, _, sd, tag = run_dir.name.split("__")
    cand = run_dir.parent / f"{ds}__{variant}__{sd}__{tag}"
    return cand if (cand / "metrics.json").exists() else None


# ----------------------------------------------------------------------------- jobs
def data_ready() -> bool:
    raw = ROOT / "data" / "raw"
    return (raw / "weibo.pt").exists() and (raw / "reddit.pt").exists() and (raw / "planetoid" / "Cora").exists()


def ensure_data() -> bool:
    """Download the pinned datasets on first start (datasets are not stored in git; SHA-256 is verified)."""
    if data_ready():
        return True
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], cwd=ROOT,
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    return proc.returncode == 0 and data_ready()


def start_job(dataset, variant, seed, smoke=False):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jid = time.strftime("%Y%m%d-%H%M%S") + f"-{dataset}-{variant}-s{seed}" + ("-smoke" if smoke else "")
    log = JOBS_DIR / f"{jid}.log"
    cmd = [sys.executable, "-m", "attngnn.cli", "run", "--dataset", dataset, "--variant", variant,
           "--seed", str(seed), "--force"] + (["--smoke"] if smoke else ["--tag", "ui"])
    f = open(log, "w")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    (JOBS_DIR / f"{jid}.json").write_text(json.dumps({"id": jid, "cmd": cmd, "pid": p.pid, "started": time.time(),
                                                      "dataset": dataset, "variant": variant, "seed": seed,
                                                      "smoke": smoke}))
    return jid, p


def job_status(jid):
    meta = json.loads((JOBS_DIR / f"{jid}.json").read_text())
    log = (JOBS_DIR / f"{jid}.log").read_text(errors="ignore") if (JOBS_DIR / f"{jid}.log").exists() else ""
    if "run complete" in log:
        state = "finished"
    elif "FAILED" in log or "Traceback" in log:
        state = "failed"
    else:
        try:
            import psutil
            state = "running" if psutil.pid_exists(meta["pid"]) else "stopped (no completion line)"
        except Exception:  # noqa: BLE001
            state = "running"
    return state, log[-4000:]


def list_jobs():
    if not JOBS_DIR.exists():
        return []
    return sorted([p.stem for p in JOBS_DIR.glob("*.json")], reverse=True)


# ----------------------------------------------------------------------------- graph + attention
_cache: dict = {}


def get_context(run_dir):
    """(model, ckpt, prep) for a GNN run; cached per run dir."""
    key = str(run_dir)
    if key not in _cache:
        model, ck = inf.load_checkpoint(run_dir)
        prep = E.prepare(ck["dataset"], ck["seed"], ck["dataset_cfg"], log=lambda s: None)
        att = S.attention_edges(model, prep["x"], prep["ei"])
        _cache[key] = (model, ck, prep, att)
    return _cache[key]


def sample_subgraph(prep, center, hops=1, max_nodes=80, seed=0):
    """k-hop neighbourhood of ``center`` capped at ``max_nodes`` (random neighbour sampling, seeded).
    Returns node list (center first) and the induced arc indices."""
    rng = np.random.default_rng(seed)
    ei = prep["bundle"].edge_index
    n = prep["n"]
    import scipy.sparse as sp
    a = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    nodes = [int(center)]
    seen = {int(center)}
    frontier = [int(center)]
    for _ in range(hops):
        nxt = []
        for u in frontier:
            nb = a.indices[a.indptr[u]:a.indptr[u + 1]]
            nb = [int(v) for v in nb if int(v) not in seen]
            rng.shuffle(nb)
            budget = max_nodes - len(nodes)
            for v in nb[:max(budget, 0)]:
                seen.add(v)
                nodes.append(v)
                nxt.append(v)
        frontier = nxt
    nodeset = np.array(nodes)
    mask = np.isin(ei[0], nodeset) & np.isin(ei[1], nodeset)
    return nodes, np.where(mask)[0]


def graph_figure(prep, att, nodes, arcs, color_scores, color_name, layer="mean", head="mean", seed=0,
                 title=None, highlight=None):
    """Plotly figure of a sampled subgraph. Edge opacity/width ~ attention of arc (src -> dst)."""
    import networkx as nx
    import plotly.graph_objects as go

    ei = prep["bundle"].edge_index
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for a in arcs:
        g.add_edge(int(ei[0, a]), int(ei[1, a]))
    pos = nx.spring_layout(g, seed=seed, k=1.5 / np.sqrt(max(len(nodes), 1)))
    if att is not None:
        al = att["alpha"]
        l = slice(None) if layer == "mean" else slice(int(layer), int(layer) + 1)
        h = slice(None) if head == "mean" else slice(int(head), int(head) + 1)
        w = al[:, l, :][:, :, h].mean((1, 2))
    else:
        w = None
    fig = go.Figure()
    if w is not None:
        lo, hi = float(w[arcs].min()), float(w[arcs].max())
    pairs = {}
    for a in arcs:
        s_, d_ = int(ei[0, a]), int(ei[1, a])
        pairs.setdefault((min(s_, d_), max(s_, d_)), []).append((s_, d_, a))
    for (u, v), lst in pairs.items():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if w is None:
            width, op, txt = 1, 0.35, f"{u} - {v}"
        else:
            wm = float(np.mean([w[a] for _, _, a in lst]))
            r = (wm - lo) / (hi - lo + 1e-12)
            width, op = 0.8 + 4.5 * r, 0.15 + 0.75 * r
            txt = "<br>".join(f"{s_} -> {d_}: attention {w[a]:.3f}" for s_, d_, a in lst)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(width=width, color=f"rgba(20,60,110,{op:.2f})"), hoverinfo="text",
                                 text=txt, showlegend=False))
    xs = [pos[v][0] for v in nodes]
    ys = [pos[v][1] for v in nodes]
    cs = [float(color_scores[v]) for v in nodes]
    preds = prep.get("_pred")
    texts = [f"node {v}<br>{color_name}: {color_scores[v]:.3f}<br>degree {int(prep['deg'][v])}"
             + (f"<br>label: {int(prep['bundle'].y[v])}" if prep['bundle'].has_labels() and prep['splits']['train'][v] else "")
             for v in nodes]
    sizes = [16 if (highlight is not None and v == highlight) else 9 for v in nodes]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(
        size=sizes, color=cs, colorscale="YlOrRd", showscale=True, colorbar=dict(title=color_name),
        line=dict(width=[2.5 if (highlight is not None and v == highlight) else 0.5 for v in nodes], color="#0b2545")),
        text=texts, hoverinfo="text", showlegend=False))
    fig.update_layout(title=title or "", xaxis=dict(visible=False), yaxis=dict(visible=False), height=560,
                      margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    return fig


def graph_png(prep, att, nodes, arcs, color_scores, path, layer="mean", head="mean", seed=0, title=""):
    """Static matplotlib export of the same sampled subgraph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    ei = prep["bundle"].edge_index
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for a in arcs:
        g.add_edge(int(ei[0, a]), int(ei[1, a]))
    pos = nx.spring_layout(g, seed=seed, k=1.5 / np.sqrt(max(len(nodes), 1)))
    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.35, edge_color="#143c6e")
    sc = nx.draw_networkx_nodes(g, pos, nodelist=list(nodes), node_color=[color_scores[v] for v in nodes],
                                cmap="YlOrRd", node_size=45, ax=ax, edgecolors="#0b2545", linewidths=0.4)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def node_inspector(prep, att, run_dir, node, layer="mean", head="mean", topk=12):
    """Neighbour table with attention (both directions) and the largest standardised feature deviations."""
    ei = prep["bundle"].edge_index
    inn = np.where(ei[1] == node)[0]
    out = np.where(ei[0] == node)[0]
    if att is not None:
        al = att["alpha"]
        l = slice(None) if layer == "mean" else slice(int(layer), int(layer) + 1)
        h = slice(None) if head == "mean" else slice(int(head), int(head) + 1)
        w = al[:, l, :][:, :, h].mean((1, 2))
    else:
        w = np.full(ei.shape[1], np.nan)
    incoming = pd.DataFrame({"neighbour": ei[0, inn], "attention_given_to_neighbour": w[inn]}).sort_values(
        "attention_given_to_neighbour", ascending=False)
    outgoing = pd.DataFrame({"neighbour": ei[1, out], "attention_received_from_neighbour": w[out]}).sort_values(
        "attention_received_from_neighbour", ascending=False)
    x = prep["x"].numpy()
    dev = np.abs(x[node] - x[ei[0, inn]].mean(0)) if len(inn) else np.abs(x[node])
    top = np.argsort(-dev)[:topk]
    feats = pd.DataFrame({"feature": top, "value(normalised)": x[node][top],
                          "neighbour_mean": x[ei[0, inn]].mean(0)[top] if len(inn) else np.nan})
    return incoming, outgoing, feats


def graph_summary_for(dataset, seed=0, cfg=None):
    b = D.load_dataset(dataset, seed, (cfg or {}).get("data_args"))
    return D.graph_summary(b), b


# ----------------------------------------------------------------------------- results files
def read_table(name):
    p = RESULTS_DIR / "tables" / name
    return pd.read_csv(p, keep_default_na=False) if p.exists() else None


def results_index():
    p = RESULTS_DIR / "results_index.json"
    return json.loads(p.read_text()) if p.exists() else None
