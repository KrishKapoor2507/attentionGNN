"""Attention-driven GNN dashboard (local only).  Launch:  streamlit run app/streamlit_app.py"""
import json
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from attngnn import app_backend as B
from attngnn import data as D
from attngnn import inference as inf
from attngnn import metrics as M
from attngnn import viz_anim as VA
from ui_theme import CSS, finding_bars, hero_html, kpi_html

NAVY, TEAL, ORANGE = "#0b2545", "#13808b", "#d9822b"

st.set_page_config(page_title="Attention GNN: influence and anomaly detection", page_icon=":material/hub:",
                   layout="wide", initial_sidebar_state="expanded")
st.html(CSS)

@st.cache_resource(show_spinner="First start: downloading the pinned datasets (about 30 seconds)...")
def _prepare_data():
    return B.ensure_data()


if not _prepare_data():
    st.warning("The datasets could not be downloaded automatically. Pages that need the graph (Data, Graph explorer, "
               "Demos) will not work until you run `python scripts/download_data.py`.", icon=":material/warning:")

runs = B.run_table()
if "last_job" not in st.session_state:
    st.session_state["last_job"] = None


# ------------------------------------------------------------------------------------------ shared helpers
def banner(prov):
    """Provenance bar shown on every page that displays saved-run results."""
    with st.container(border=True, key="card_prov"):
        smoke = " :red-badge[SMOKE RUN: not a benchmark]" if prov["is_smoke"] else ""
        st.markdown(f"**Run** `{prov['run_id']}` · :blue-badge[{prov['dataset']}] :violet-badge[{prov['variant']}] · seed {prov['seed']} "
                    f"· checkpoint `{prov['checkpoint_sha256']}…` · train time {(prov['train_seconds'] or 0):.0f}s{smoke}")
        st.caption(f"started {prov['started']} · machine {prov.get('cpu') or 'n/a'}")


def pick_run(label="Saved run", gnn_only=True, key="run", prefer=("gat_joint",)):
    df = runs
    if gnn_only:
        df = df[df.has_checkpoint]
    if df.empty:
        st.info("No saved GNN runs yet. Use **Train / evaluate** (smoke mode takes about 1 minute).", icon=":material/info:")
        return None
    df = df.sort_values(["tag", "dataset", "variant", "seed"])
    opts = df.run_id.tolist()
    default = next((i for i, r in enumerate(opts) if any(p in r for p in prefer) and "bench" in r), 0)
    rid = st.selectbox(label, opts, index=default, key=key)
    return Path(df[df.run_id == rid].path.iloc[0])


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table_floats(df, col):
    """'0.771 ± 0.031 [n=5]' -> (0.771, 0.031) or None."""
    out = []
    for s in df[col]:
        m = re.match(r"\s*(-?[0-9.]+)(?: ± ([0-9.]+))?", str(s))
        out.append((float(m.group(1)), float(m.group(2) or 0)) if m else None)
    return out


# ------------------------------------------------------------------------------------------ pages
def page_overview():
    st.iframe(hero_html("Can attention find the hubs and the outliers?",
                        "A shared graph-attention encoder, tested as a hypothesis on Weibo (organic labels) and Cora (injected "
                        "anomalies). Every number on these pages is read from saved, hash-checked runs.", 300),
              height=300, alt="Animated network: attention pulses travel along edges, a teal hub emits ripples, an orange node pulses")
    bench = runs[runs.tag == "bench"] if len(runs) else runs
    val = load_json(ROOT / "artifacts" / "results" / "dataset_validation.json")
    npass = sum(r["result"] == "PASS" for r in val["rows"]) if val else 0
    st.iframe(kpi_html([("benchmark runs saved", int(len(bench)), ""), ("methods per dataset", 12, ""),
                        ("Cora seeds", 5, ""), ("Weibo seeds", 3, ""), ("dataset checks passed", int(npass), "")]),
              height=125, alt="Key numbers of the project, counting up")

    st.subheader("Headline findings (measured)")
    facts = load_json(ROOT / "artifacts" / "results" / "facts.json") or {}

    def f(k):
        try:
            return float(str(facts[k]).split(" ")[0])
        except (KeyError, ValueError):
            return None

    c1, c2 = st.columns(2)
    with c1.container(border=True, key="card_anom"):
        st.markdown("#### :material/crisis_alert: Anomaly detection (test ROC-AUC)")
        items = []
        for lab, k in [("Cora: GAT reconstruction", "cora_injected.anom.gat_anom.rec.roc_auc"),
                       ("Cora: attention-only score", "cora_injected.anom.gat_anom.att_only.roc_auc"),
                       ("Cora: uniform-neighbour control", "cora_injected.anom.gat_anom.feat_dev_uniform.roc_auc"),
                       ("Weibo: GCN reconstruction", "weibo.anom.gcn_anom.rec.roc_auc"),
                       ("Weibo: attention-only score", "weibo.anom.gat_anom.att_only.roc_auc")]:
            v = f(k)
            if v is not None:
                items.append((lab, v, f"{v:.3f}", "control" in lab))
        st.html(finding_bars(items) if items else "<i>results not built yet</i>")
        st.caption("Attention-only is matched by a uniform-weight control: the signal is neighbour feature discrepancy.")
    with c2.container(border=True, key="card_inf"):
        st.markdown("#### :material/trending_up: Influence ranking (Spearman vs held-out spread)")
        items = []
        for lab, k in [("Weibo: degree", "weibo.inf.0.003.centrality.degree.spearman"),
                       ("Weibo: GCN influence head", "weibo.inf.0.003.gcn_joint.model_pred.spearman"),
                       ("Weibo: GAT received attention", "weibo.inf.0.003.gat_joint.recv_att.spearman"),
                       ("Cora: degree", "cora_injected.inf.0.1.centrality.degree.spearman"),
                       ("Cora: GAT received attention", "cora_injected.inf.0.1.gat_joint.recv_att.spearman")]:
            v = f(k)
            if v is not None:
                items.append((lab, max(v, 0.0), f"{v:.3f}", v < 0.4))
        st.html(finding_bars(items) if items else "<i>results not built yet</i>")
        st.caption("Influence is simulated independent-cascade spread, never observed influence.")

    with st.container(key="banner"):
        st.markdown("**Scope.** Influence labels are simulated; Cora anomalies are injected; Weibo labels are the *original* 2022 "
                    "benchmark labels (the 2024 upstream file blanks validation/test labels, see **Dataset authenticity**). "
                    "Attention is tested as an association, not assumed to be an explanation.")
    if len(runs):
        with st.expander("All saved runs", icon=":material/table:"):
            st.dataframe(runs.drop(columns=["path"]), hide_index=True)


def page_data():
    st.title("Dataset loader and graph summary")
    src = st.segmented_control("Source", ["Built-in dataset", "Upload CSV edge list"], default="Built-in dataset", key="data_src")
    if src != "Upload CSV edge list":
        c1, c2 = st.columns(2)
        ds = c1.selectbox("Dataset", ["cora_injected", "weibo", "reddit"])
        seed = c2.number_input("Seed (controls Cora injection)", 0, 99, 0)
        try:
            with st.spinner("Loading graph..."):
                s, b = B.graph_summary_for(ds, int(seed))
        except FileNotFoundError as e:
            st.error(f"{e}", icon=":material/error:")
            return
        meta = s.pop("meta")
        inj = s.pop("injection", None)
        cols = st.columns(4)
        cols[0].metric("Nodes", f"{s['nodes']:,}")
        cols[1].metric("Undirected edges", f"{s['undirected_edges']:,}")
        cols[2].metric("Features", f"{s['features']:,}")
        if "anomalies" in s:
            cols[3].metric("Anomalies", f"{s['anomalies']:,}", f"{s['anomaly_rate']:.1%} of nodes", delta_color="off")
        st.dataframe(pd.DataFrame({"property": list(s), "value": [str(v) for v in s.values()]}), hide_index=True)
        with st.expander("Provenance and preprocessing", icon=":material/fact_check:"):
            st.json({k: v for k, v in meta.items() if k not in ("preprocessing_clean_graph",)}, expanded=False)
        if inj:
            with st.expander("Injection parameters (synthetic labels)", icon=":material/science:"):
                st.json(inj)
        if b.has_labels():
            fig = go.Figure(go.Bar(x=["normal", "anomalous"], y=[int((b.y == 0).sum()), int(b.y.sum())], marker_color=[TEAL, ORANGE]))
            fig.update_layout(title="Class balance", height=300, margin=dict(t=40, b=20), plot_bgcolor="white")
            st.plotly_chart(fig, width="stretch")
    else:
        st.markdown("`edges.csv`: header `source,target` (IDs are free text; undirected; duplicates and self-loops are dropped and "
                    "counted). `features.csv` (optional): `id,f1,f2,...`, all numeric. `labels.csv` (optional): `id,label` with 0/1, "
                    "evaluation only. Limits: 100k nodes, 2M edges, 2000 features. Without features, four structural features are derived.")
        ef = st.file_uploader("edges.csv", type="csv")
        ff = st.file_uploader("features.csv (optional)", type="csv")
        lf = st.file_uploader("labels.csv (optional)", type="csv")
        if ef is not None:
            tmp = Path(tempfile.mkdtemp())
            paths = []
            for up, nm in ((ef, "e.csv"), (ff, "f.csv"), (lf, "l.csv")):
                if up is None:
                    paths.append(None)
                else:
                    (tmp / nm).write_bytes(up.getvalue())
                    paths.append(tmp / nm)
            try:
                b = D.load_csv_graph(*paths)
                s = D.graph_summary(b)
                s.pop("meta", None)
                st.success("Parsed OK", icon=":material/check_circle:")
                st.dataframe(pd.DataFrame({"property": list(s), "value": [str(v) for v in s.values()]}), hide_index=True)
                st.json(b.meta["preprocessing"])
            except D.CSVSchemaError as e:
                st.error(f"Invalid input: {e}", icon=":material/error:")


def page_authenticity():
    st.title("Dataset authenticity and validation")
    val = load_json(ROOT / "artifacts" / "results" / "dataset_validation.json")
    st.caption("Checks that the data are what they claim to be: integrity against a pinned upstream commit, structural validity, "
               "agreement with published statistics, label semantics and injection integrity.")
    if st.button("Re-run validation now (needs internet, about 1-2 minutes)", icon=":material/refresh:"):
        with st.status("Running scripts/validate_datasets.py ...", expanded=True) as status:
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_datasets.py")], cwd=ROOT, capture_output=True, text=True)
            st.code(proc.stdout[-3000:] or proc.stderr[-3000:])
            status.update(label="Validation finished" if proc.returncode == 0 else "Validation reported failures", state="complete" if proc.returncode == 0 else "error")
        st.rerun()
    if not val:
        st.info("No validation report yet. Click the button above or run `python scripts/validate_datasets.py`.", icon=":material/info:")
        return
    rows = pd.DataFrame(val["rows"])
    rows["dataset"] = rows["dataset"].str.replace(r" seed \d+", "", regex=True).replace({"cora": "cora (clean)", "cora_injected": "cora (injected)"})
    c = st.columns(4)
    c[0].metric("Checks passed", int((rows.result == "PASS").sum()))
    c[1].metric("Checks failed", int((rows.result == "FAIL").sum()))
    c[2].metric("Findings", int((rows.result == "FINDING").sum()))
    c[3].metric("Report built", val["built"][:16])
    finding = rows[rows.result == "FINDING"]
    if len(finding):
        with st.container(border=True, key="card_finding"):
            st.markdown("#### :material/warning: Key finding: the upstream Weibo file was changed")
            for _, r in finding.iterrows():
                st.markdown(f"- **{r['check']}**: {r['detail']}")
            st.caption("This project therefore pins the original 2022 file (868 anomalies, matching the BOND paper) and verifies its SHA-256.")
    sec = st.segmented_control("Section", sorted(rows.section.unique()), default=None, key="val_sec")
    ds = st.pills("Dataset", sorted(rows.dataset.unique()), selection_mode="multi", key="val_ds")
    view = rows
    if sec:
        view = view[view.section == sec]
    if ds:
        view = view[view.dataset.isin(ds)]
    st.dataframe(view, hide_index=True, height=420,
                 column_config={"result": st.column_config.TextColumn("result", width="small"),
                                "detail": st.column_config.TextColumn("detail", width="large")})
    sem = val.get("label_semantics", {}).get("weibo")
    if sem:
        fig = go.Figure()
        fig.add_bar(x=["clustering coefficient", "neighbour feature similarity"], y=[sem["clustering_anom"], sem["neigh_sim_anom"]], name="anomalous", marker_color=ORANGE)
        fig.add_bar(x=["clustering coefficient", "neighbour feature similarity"], y=[sem["clustering_norm"], sem["neigh_sim_norm"]], name="normal", marker_color=TEAL)
        fig.update_layout(title="Weibo label semantics: anomalous vs normal users (sampled)", barmode="group", height=320, plot_bgcolor="white", margin=dict(t=50, b=20))
        st.plotly_chart(fig, width="stretch")
    st.markdown("Full text: `docs/dataset_validation.md`.")


@st.fragment(run_every="2s")
def job_panel(jid):
    """Live job tracker: refreshes itself every two seconds until the job ends."""
    state, log = B.job_status(jid)
    m = re.findall(r"epoch\s+(\d+)/(\d+)", log)
    frac = (int(m[-1][0]) / int(m[-1][1])) if m else 0.0
    if state == "finished":
        frac = 1.0
    icon = {"running": ":material/autorenew:", "finished": ":material/check_circle:", "failed": ":material/error:"}.get(state, ":material/help:")
    st.markdown(f"{icon} **Status:** `{state}`" + (' <span class="live-dot"></span>' if state == "running" else ""), unsafe_allow_html=True)
    st.progress(frac, text=f"training epoch {m[-1][0]}/{m[-1][1]}" if m else "preparing data and cascade targets...")
    st.code(log or "(no output yet)", height=220)
    if state == "finished" and st.session_state.get("toasted") != jid:
        st.session_state["toasted"] = jid
        st.toast(f"Job {jid} finished", icon=":material/check_circle:")


def page_train():
    st.title("Train and evaluate")
    st.markdown("Each job runs `python -m attngnn.cli run ...` in the background and writes a full run record (config, command, "
                "checksums, logs, checkpoint, predictions, metrics).")
    from attngnn.experiments import VARIANTS
    with st.container(border=True, key="card_job"):
        c1, c2, c3 = st.columns(3)
        ds = c1.selectbox("Dataset", ["cora_injected", "weibo"])
        var = c2.selectbox("Variant", list(VARIANTS) + ["iforest_attr", "iforest_attr_struct", "centrality"], index=5)
        seed = c3.number_input("Seed", 0, 99, 0)
        smoke = st.toggle("Smoke mode (4 epochs, tiny simulations; results are not reportable)", value=True)
        if not smoke:
            st.warning("Full runs use the benchmark budget (Cora about 1-2 min, Weibo about 3-8 min on this CPU).", icon=":material/schedule:")
        if st.button("Start job", icon=":material/play_arrow:", type="primary"):
            jid, _ = B.start_job(ds, var, int(seed), smoke)
            st.session_state["last_job"] = jid
            st.toast(f"Started {jid}", icon=":material/rocket_launch:")
    jobs = B.list_jobs()
    if jobs:
        jid = st.selectbox("Job", jobs, index=jobs.index(st.session_state["last_job"]) if st.session_state["last_job"] in jobs else 0)
        job_panel(jid)
    st.divider()
    st.subheader("Checkpoint load and reproduce")
    rd = pick_run("Reload checkpoint of run", key="ckrun")
    if rd is not None and st.button("Verify hash and reproduce saved predictions", icon=":material/verified:"):
        with st.spinner("Recomputing every score from the checkpoint..."):
            ok_hash = inf.verify_checkpoint_hash(rd)
            r = inf.reproduce(rd)
        (st.success if ok_hash and r["ok"] else st.error)(f"Checkpoint hash matches: **{ok_hash}** · recomputed scores equal saved predictions (tolerance 1e-4): **{r['ok']}**")
        st.json(r["max_abs_diff"])


def page_influence():
    st.title("Ranked influence table")
    rd = pick_run(key="infrun")
    if rd is None:
        return
    banner(B.provenance(rd))
    m = inf.load_metrics(rd)
    pred = B.load_predictions(rd)
    cen = B.sibling_run(rd, "centrality")
    if cen is not None:
        cp = B.load_predictions(cen)
        for c in ("degree", "pagerank", "betweenness"):
            pred["inf_" + c] = cp["inf_" + c]
    pgrid = list(m["influence"].keys())
    c1, c2, c3 = st.columns(3)
    p = c1.select_slider("Cascade probability p (IC proxy)", pgrid, value=pgrid[len(pgrid) // 2])
    score_cols = [c for c in pred.columns if c.startswith("inf_") and "@" not in c] + [c for c in pred.columns if c.startswith("inf_model_pred@") and c.endswith(p)]
    by = c2.selectbox("Rank by", score_cols, index=0)
    split = c3.segmented_control("Nodes", ["test", "val", "train", "all"], default="test")
    sub = pred if split == "all" else pred[pred.split == split]
    rows_n = st.slider("Rows", 10, 500, 50)
    tbl = sub[["node", "split", "degree"] + [c for c in score_cols if c != "inf_degree"] + [f"spread_mean@{p}", f"spread_se@{p}"]].copy()
    tbl = tbl.sort_values(by, ascending=False).head(rows_n).rename(columns={f"spread_mean@{p}": "simulated_spread (IC proxy)", f"spread_se@{p}": "spread_SE"})
    st.dataframe(tbl, hide_index=True)
    if m.get("influence_note"):
        st.warning(m["influence_note"], icon=":material/warning:")
    st.caption("simulated_spread is the Monte-Carlo expected number of activated nodes when the node is the lone seed (independent cascade; "
               "held-out stream for test nodes). It is a simulation proxy, not observed influence.")
    fig = go.Figure(go.Scattergl(x=sub[by], y=sub[f"spread_mean@{p}"], mode="markers", marker=dict(color=TEAL, size=5, opacity=0.5),
                                 text=sub.node, hovertemplate="node %{text}<br>score %{x:.3g}<br>spread %{y:.2f}<extra></extra>"))
    fig.update_layout(title=f"Simulated spread vs {by} ({split} nodes)", height=360, plot_bgcolor="white", yaxis_type="log",
                      xaxis_title=by, yaxis_title="simulated spread (log)", margin=dict(t=50, b=30), transition=dict(duration=400))
    st.plotly_chart(fig, width="stretch")
    st.subheader("Held-out ranking quality")
    rows = []
    for meth, v in m["influence"][p]["methods"].items():
        rows.append({"method": meth, "spearman": v["spearman"], "kendall": v["kendall"], **{k: v[k] for k in v if k.startswith(("ndcg", "topk"))}})
    if cen is not None:
        for meth, v in inf.load_metrics(cen)["influence"][p]["methods"].items():
            rows.append({"method": meth, "spearman": v["spearman"], "kendall": v["kendall"], **{k: v[k] for k in v if k.startswith(("ndcg", "topk"))}})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def page_anomaly():
    st.title("Anomaly scores and threshold")
    rd = pick_run(key="anrun")
    if rd is None:
        return
    prov = B.provenance(rd)
    banner(prov)
    m = inf.load_metrics(rd)
    pred = B.load_predictions(rd)
    names = [c[5:] for c in pred.columns if c.startswith("anom_")]
    if not m.get("trained_rec", True):
        st.warning(m.get("anomaly_note", ""), icon=":material/warning:")
    sname = st.selectbox("Score", names, index=names.index("combined") if "combined" in names else 0, format_func=lambda s: B.ANOMALY_LABELS.get(s, s))
    s = pred["anom_" + sname].to_numpy()
    a = m["anomaly"][sname]
    frozen = a["threshold"]
    st.markdown(f"Frozen decision threshold (selected on **validation** F1 before test): `{frozen:.4g}`" if frozen is not None else "Threshold undefined (one-class validation).")
    lo, hi = float(np.quantile(s, 0.001)), float(np.quantile(s, 0.999))
    thr = st.slider("Inference threshold", lo, hi, float(frozen if frozen is not None else np.median(s)), step=(hi - lo) / 400)
    flagged = s >= thr
    c = st.columns(4)
    c[0].metric("Flagged (all nodes)", int(flagged.sum()))
    c[1].metric("Flagged in test", int((flagged & (pred.split == "test")).sum()))
    c[2].metric("Flagged in val", int((flagged & (pred.split == "val")).sum()))
    c[3].metric("Flagged in train", int((flagged & (pred.split == "train")).sum()))
    yl = pred.y.to_numpy()
    va = (pred.split == "val").to_numpy()
    left, right = st.columns(2)
    with left:
        st.markdown("**Metrics at this threshold: VALIDATION only** (tune here). Test metrics appear only at the frozen threshold.")
        mv = M.anomaly_metrics(yl[va], s[va], thr)
        c2 = st.columns(3)
        c2[0].metric("Precision", "n/a" if mv["precision"] is None else f"{mv['precision']:.2f}")
        c2[1].metric("Recall", "n/a" if mv["recall"] is None else f"{mv['recall']:.2f}")
        c2[2].metric("F1", "n/a" if mv["f1"] is None else f"{mv['f1']:.2f}")
    with right:
        st.plotly_chart(VA.score_histogram(s[va], yl[va], thr), width="stretch")
    t = a["test"]
    st.markdown("**Test (frozen threshold):** " + " · ".join(f"{k} {t[k]:.3f}" for k in ("precision", "recall", "f1", "roc_auc", "average_precision", "precision_at_k") if t.get(k) is not None) + f" · support {t['support_pos']}")
    cm = t["confusion"]
    fig = go.Figure(go.Heatmap(z=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], x=["pred normal", "pred anomaly"], y=["true normal", "true anomaly"],
                               colorscale="Blues", text=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], texttemplate="%{text}"))
    fig.update_layout(title="Test confusion matrix at the frozen threshold", height=320, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    out = pred[["node", "split", "degree", "y", "y_type"]].copy()
    out["score"] = s
    out["flagged"] = flagged
    out.loc[out.split != "train", "y_shown"] = out.y
    out = out.drop(columns=["y"]).sort_values("score", ascending=False).head(300)
    st.dataframe(out, hide_index=True)
    st.caption("The label column `y_shown` is evaluation information; do not use test rows to tune the threshold." +
               (" y_type: 1 = injected contextual, 2 = injected structural (synthetic labels)." if prov["dataset"] == "cora_injected" else ""))


def cascade_panel(prep, centre, key):
    import networkx as nx
    st.markdown("##### :material/bolt: Cascade animation (illustration)")
    st.caption("One random independent-cascade run on the sampled neighbourhood: orange = seed, sand = newly activated, teal = already "
               "active. The probability here is only for illustration; the measured influence targets use the grids reported in Section 3.1.")
    c1, c2 = st.columns(2)
    p = c1.slider("Illustrative activation probability", 0.05, 0.9, 0.4, 0.05, key=f"{key}_p")
    rs = c2.number_input("Random seed", 0, 999, 7, key=f"{key}_s")
    nodes, arcs = B.sample_subgraph(prep, centre, 2, 45, seed=1)
    ei = prep["bundle"].edge_index
    src, dst = ei[0, arcs], ei[1, arcs]
    step, fired = VA.simulate_cascade(src, dst, nodes, centre, p, int(rs))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(zip(src.tolist(), dst.tolist()))
    pos = nx.spring_layout(g, seed=3, k=1.4 / np.sqrt(len(nodes)))
    edges = [tuple(sorted(e)) for e in g.edges()]
    st.plotly_chart(VA.cascade_figure(pos, edges, step, fired, centre), width="stretch", key=f"{key}_fig")


def page_graph():
    st.title("Interactive sampled graph and node inspector")
    rd = pick_run(key="gxrun")
    if rd is None:
        return
    banner(B.provenance(rd))
    with st.spinner("Loading model and graph..."):
        model, ck, prep, att = B.get_context(rd)
    pred = B.load_predictions(rd)
    anames = [c[5:] for c in pred.columns if c.startswith("anom_")]
    c1, c2, c3 = st.columns(3)
    score = c1.selectbox("Colour nodes by", [f"anom_{a}" for a in anames] + [c for c in pred.columns if c.startswith("inf_")])
    start = c2.selectbox("Centre node", ["top anomaly score (test, degree >= 4)", "top received attention (test, degree >= 4)", "manual"])
    te = pred[pred.split == "test"]
    te_nb = te[te.degree >= 4] if (te.degree >= 4).any() else te
    if start.startswith("top anomaly"):
        centre = int(te_nb.sort_values("anom_" + ("combined" if "combined" in anames else anames[0]), ascending=False).node.iloc[0])
    elif start.startswith("top received") and "inf_recv_att" in pred:
        centre = int(te_nb.sort_values("inf_recv_att", ascending=False).node.iloc[0])
    else:
        centre = int(c3.number_input("Node id", 0, prep["n"] - 1, 0))
    c4, c5, c6, c7 = st.columns(4)
    hops = c4.slider("Hops", 1, 2, 1)
    cap = c5.slider("Max nodes shown", 20, 200, 80)
    nl = att["alpha"].shape[1] if att is not None else 0
    nh = att["alpha"].shape[2] if att is not None else 0
    layer = c6.selectbox("Layer", ["mean"] + [str(i) for i in range(nl)])
    head = c7.selectbox("Head", ["mean"] + [str(i) for i in range(nh)])
    nodes, arcs = B.sample_subgraph(prep, centre, hops, cap, seed=0)
    st.caption(f"Sampled subgraph: {hops}-hop neighbourhood of node {centre}, capped at {cap} nodes by seeded random neighbour sampling "
               f"({len(nodes)} nodes, {len(arcs) // 2} undirected edges shown). Edge width and opacity show the selected attention "
               "coefficient (source -> target); layout positions are arbitrary.")
    if att is None:
        st.info("This encoder (GCN) has no attention; edges are drawn with uniform weight.", icon=":material/info:")
    fig = B.graph_figure(prep, att, nodes, arcs, pred[score].to_numpy(), score, layer, head, highlight=centre, title=f"Sample around node {centre} (layer={layer}, head={head})")
    st.plotly_chart(fig, width="stretch")
    st.subheader(f"Node inspector: {centre}")
    st.dataframe(pred[pred.node == centre].iloc[0].to_frame("value"))
    inc, outg, feats = B.node_inspector(prep, att, rd, centre, layer, head)
    a1, a2 = st.columns(2)
    a1.markdown("**Attention this node gives its neighbours** (messages arriving at it)")
    a1.dataframe(inc.head(15), hide_index=True)
    a2.markdown("**Attention this node receives from its neighbours** (sum = received attention)")
    a2.dataframe(outg.head(15), hide_index=True)
    st.markdown("**Features deviating most from the neighbour mean** (normalised units)")
    st.dataframe(feats, hide_index=True)
    st.divider()
    cascade_panel(prep, centre, "gx")


def page_results():
    st.title("Model comparison and ablations (from saved runs)")
    idx = B.results_index()
    if not idx:
        st.info("Results not built yet. Run `python scripts/build_results.py` after the benchmark.", icon=":material/info:")
        return
    st.caption(f"Tables rebuilt from {idx['n_runs']} run directories on {idx['built']}. Published numbers are never mixed into these locally measured tables.")
    ds = st.segmented_control("Dataset", idx["datasets"], default=idx["datasets"][0], key="res_ds") or idx["datasets"][0]
    t = B.read_table(f"anomaly_{ds}.csv")
    if t is not None:
        st.subheader("Anomaly detection (test, mean ± std over seeds)")
        vals = table_floats(t, "ROC-AUC")
        keep = [(m, v) for m, v in zip(t.method, vals) if v]
        if keep:
            colors = [("#8d99ae" if k == "baseline" else "#e9c46a" if k == "control" else TEAL if "attention" in m.lower() else NAVY)
                      for (m, _), k in zip(keep, t.kind[[i for i, v in enumerate(vals) if v]])]
            fig = VA.bars_reveal_figure([m for m, _ in keep], [v[0] for _, v in keep], [v[1] for _, v in keep], colors,
                                        title=f"Test ROC-AUC, {ds} (press Play to animate)", xlabel="ROC-AUC", vmin=0.3 if min(v[0] for _, v in keep) < 0.5 else 0.5)
            st.plotly_chart(fig, width="stretch", key=f"reveal_{ds}")
        st.dataframe(t, hide_index=True)
    for title, fname in [("Influence ranking (held-out test nodes)", f"influence_{ds}.csv"), ("Seed-paired differences", f"paired_{ds}.csv"),
                         ("Ablations", f"ablation_{ds}.csv"), ("Explanation perturbation", f"explanation_{ds}.csv"),
                         ("Resource cost (measured)", f"resources_{ds}.csv"), ("Isolated resource profiles", f"resources_profile_{ds}.csv")]:
        tt = B.read_table(fname)
        if tt is not None:
            with st.expander(title, expanded=title.startswith("Influence"), icon=":material/table:"):
                st.dataframe(tt, hide_index=True)
    fdir = B.RESULTS_DIR / "figures"
    imgs = sorted(fdir.glob(f"*{ds}*.png")) if fdir.exists() else []
    if imgs:
        with st.expander("Figures", icon=":material/image:"):
            for f in imgs:
                st.image(str(f), caption=f.name)


def page_demos():
    st.title("Deterministic demos (each tied to a saved run)")
    rd = pick_run("Run for demos", key="demorun")
    if rd is None:
        return
    prov = B.provenance(rd)
    banner(prov)
    m = inf.load_metrics(rd)
    pred = B.load_predictions(rd)
    mode = st.segmented_control("Demo", ["Known anomaly", "High-spread node", "Explanation test", "Saved model reload", "Training replay", "ROC drawing"],
                                default="Known anomaly", key="demo_mode")
    if mode == "Known anomaly":
        te = pred[(pred.split == "test") & (pred.y == 1)]
        sc = "anom_combined" if "anom_combined" in pred else "anom_rec"
        if te.empty:
            st.info("No labelled anomalies in the test split.", icon=":material/info:")
        else:
            pred["rank"] = pred[sc].rank(ascending=False, method="first")
            pick = te.assign(rank=pred.loc[te.index, "rank"]).sort_values("rank").iloc[0]
            c = st.columns(3)
            c[0].metric("Node", int(pick.node))
            c[1].metric(f"Rank by {sc[5:]}", f"{int(pick['rank'])} / {len(pred)}")
            c[2].metric("Degree", int(pick.degree))
            st.markdown(f"Highest-ranked **true anomaly in the test split** (y_type {int(pick.y_type)}); test anomalies: {len(te)}, base rate {pred.y.mean():.3%}.")
            st.dataframe(pred.sort_values(sc, ascending=False).head(10)[["node", "split", "y", "y_type", sc, "degree"]], hide_index=True)
    elif mode == "High-spread node":
        te = pred[pred.split == "test"]
        pg = list(m["influence"].keys())[len(m["influence"]) // 2]
        top = te.sort_values(f"spread_mean@{pg}", ascending=False).head(10)
        cols = ["node", "degree", f"spread_mean@{pg}", f"spread_se@{pg}"] + [c for c in pred if c.startswith("inf_") and "@" not in c]
        st.markdown(f"Test nodes with the highest simulated spread at p={pg} (independent-cascade proxy):")
        st.dataframe(top[cols], hide_index=True)
        try:
            model, ck, prep, att = B.get_context(rd)
            cascade_panel(prep, int(top.node.iloc[0]), "demo")
        except Exception as e:  # noqa: BLE001
            st.caption(f"cascade animation unavailable: {e}")
    elif mode == "Explanation test":
        ex = m.get("explanation")
        if not ex:
            st.info("This run has no explanation experiment (only gat_joint runs do).", icon=":material/info:")
        else:
            for k in ("anomaly", "influence"):
                if k in ex and ex[k].get("n_targets"):
                    e = ex[k]
                    st.markdown(f"**{k}**: {e['n_targets']} targets, at most {ex['edge_budget_m']} edges removed")
                    fig = go.Figure(go.Bar(x=list(e["mean_effect"]), y=list(e["mean_effect"].values()), marker_color=[NAVY, TEAL, ORANGE]))
                    fig.update_layout(height=300, yaxis_title="mean output drop after removal", margin=dict(t=20), plot_bgcolor="white")
                    st.plotly_chart(fig, width="stretch", key=f"ex_{k}")
                    st.write({"high - random (95% CI)": (e["diff_high_minus_random"], e["diff_high_minus_random_ci95"]),
                              "fraction of targets high > random": e["frac_targets_high_gt_random"], "Wilcoxon p (high > random)": e["wilcoxon_p_high_vs_random"]})
            st.caption(ex["note"])
    elif mode == "Saved model reload":
        st.write(f"Checkpoint file: `{rd / 'checkpoint.pt'}`")
        if st.button("Reload checkpoint and recompute all scores", icon=":material/restart_alt:"):
            with st.spinner("Recomputing..."):
                r = inf.reproduce(rd)
            st.success(f"Hash matches: {inf.verify_checkpoint_hash(rd)} · scores reproduce: {r['ok']}", icon=":material/verified:")
            st.json(r["max_abs_diff"])
    elif mode == "Training replay":
        h = load_json(rd / "history.json")
        fig = VA.loss_replay_figure(h["hist"]) if h else None
        if fig is None:
            st.info("This run has no training history (untrained variant).", icon=":material/info:")
        else:
            st.plotly_chart(fig, width="stretch", key="replay")
            st.caption("Each curve is a real per-epoch loss saved in history.json; the animation replays them.")
    else:
        cv = load_json(rd / "curves.json") or {}
        curves = {B.ANOMALY_LABELS.get(k, k): v for k, v in cv.items() if v and k in ("rec", "att_only", "combined", "feat_dev_uniform")}
        if not curves:
            st.info("No curves saved for this run.", icon=":material/info:")
        else:
            st.plotly_chart(VA.roc_draw_figure(curves), width="stretch", key="rocdraw")


def page_exports():
    st.title("Exports and documents")
    rd = pick_run(gnn_only=False, key="exrun")
    if rd is not None:
        banner(B.provenance(rd))
        c = st.columns(4)
        for col, (label, fname, mime) in zip(c, [("Predictions (CSV.gz)", "predictions.csv.gz", "application/gzip"), ("Configuration", "config.json", "application/json"),
                                                  ("Metrics", "metrics.json", "application/json"), ("Command", "command.txt", "text/plain")]):
            p = rd / fname
            if p.exists():
                col.download_button(label, p.read_bytes(), file_name=f"{rd.name}_{fname}", mime=mime, icon=":material/download:", width="stretch")
        if (rd / "checkpoint.pt").exists():
            try:
                model, ck, prep, att = B.get_context(rd)
                pred = B.load_predictions(rd)
                if st.button("Render graph image (PNG)", icon=":material/image:"):
                    nodes, arcs = B.sample_subgraph(prep, int(pred.sort_values("anom_rec", ascending=False).node.iloc[0]), 1, 80)
                    out = B.ARTIFACTS / "exports" / f"{rd.name}_graph.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    B.graph_png(prep, att, nodes, arcs, pred["anom_rec"].to_numpy(), out, title=f"{rd.name}: sampled neighbourhood coloured by reconstruction score")
                    st.image(str(out))
                    st.download_button("Download PNG", out.read_bytes(), file_name=out.name, mime="image/png", icon=":material/download:")
            except Exception as e:  # noqa: BLE001
                st.warning(f"graph export unavailable: {e}")
    st.subheader("Report and presentation")
    for nm in ["Attention_GNN_Report.pdf", "Attention_GNN_Report.docx", "Attention_GNN_Report.md", "Attention_GNN_Final.pptx", "demo_script.md"]:
        p = B.REPORTS_DIR / nm
        if p.exists():
            st.download_button(f"Download {nm}", p.read_bytes(), file_name=nm, icon=":material/description:")
        else:
            st.caption(f"{nm}: not built yet")


PAGES = {
    "Overview": ("home", page_overview), "Data & graph summary": ("dataset", page_data),
    "Dataset authenticity": ("verified_user", page_authenticity), "Train / evaluate": ("model_training", page_train),
    "Influence ranking": ("trending_up", page_influence), "Anomaly scores": ("crisis_alert", page_anomaly),
    "Graph explorer & node inspector": ("hub", page_graph), "Results: comparison & ablation": ("analytics", page_results),
    "Demos": ("movie", page_demos), "Exports & documents": ("folder_open", page_exports),
}

with st.sidebar:
    st.markdown("### Attention GNN")
    st.caption("Influence and anomaly detection · local only")
    with st.container(key="nav"):
        page = st.radio("View", list(PAGES), format_func=lambda k: f":material/{PAGES[k][0]}: {k}", label_visibility="collapsed")
    st.caption(f":material/database: {len(runs)} saved runs · :material/lock: served on localhost only")
    if len(runs) and (runs.tag == "smoke").any():
        st.warning("Smoke runs are tiny sanity checks, not results.", icon=":material/info:")

with st.container(key="page"):
    PAGES[page][1]()
