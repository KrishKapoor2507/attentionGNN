# Attention-Driven GNNs for Influence and Anomaly Detection

![CI](https://github.com/KrishKapoor2507/attentionGNN/actions/workflows/ci.yml/badge.svg)

**School of Computer Science and Engineering (SCOPE), VIT Vellore · CSI1006 project**
Krish Kapoor (23MIC0023) · A.M. Karthikeshwaran (23MID0215)

Can the weights a **graph attention network (GAT)** learns for each neighbour serve as one shared signal for two classic
network questions: *which nodes are influential?* and *which nodes are anomalous?* This project builds a CPU-only pipeline
to **test** that idea honestly against strong baselines and non-attentive controls, with a dashboard to inspect every result.

> **Headline (measured, not promised).** Reconstruction-based scores detect anomalies well (ROC-AUC about 0.77–0.79), but the
> attention-only score is matched by a uniform-weight control that learns nothing. For influence, degree and a plain GCN
> head are strong, while received attention equals (Cora) or underperforms (Weibo) a non-learned control. Removing
> high-attention edges changes the model more than removing random edges, so attention describes what the model uses without
> being a better predictor. Details, uncertainty and limits: `reports/Attention_GNN_Report.pdf`.

## What is in the box

| Area | What it provides |
|---|---|
| **Model** | Shared 2-layer encoder: GAT (4 heads), GATv2 and GCN controls; attribute / structure / influence heads |
| **Anomaly scores** | reconstruction (DOMINANT-style), attention-only (weighted feature discrepancy), combined, uniform and untrained controls |
| **Influence scores** | received attention (raw, degree-adjusted), influence head trained on independent-cascade simulations, vs degree / PageRank / betweenness |
| **Data** | Weibo (organic labels, pinned original file) and Cora with seeded injected anomalies; optional Reddit; CSV import |
| **Evidence** | 96 saved runs (config, command, hashes, checkpoint, predictions, metrics), tables and figures rebuilt from them |
| **Dashboard** | 10 pages with animations: data, **dataset authenticity**, live training jobs, rankings, thresholds, graph explorer, results, demos, exports |
| **Documents** | 32-page report (PDF/DOCX/MD), 14-slide deck, demo script, literature survey (40 verified references) |
| **Quality** | 82 automated tests, GitHub Actions CI, weekly dataset-integrity check |

## Data authenticity (important)
The current upstream `weibo.pt` (January 2024, commit "add mask") **blanks the labels of all validation and test nodes**
(347 of 868 anomalies remain). This project pins the **original 2022 commit** and verifies its SHA-256
(`scripts/download_data.py`). `python scripts/validate_datasets.py` runs 42 checks (checksums vs a fresh download, structure,
published statistics, label semantics, injection integrity); the readable report is `docs/dataset_validation.md`. Our first
Weibo run had used the corrupted file; it was discarded (`artifacts/superseded/`) and every Weibo experiment was repeated.

## Repository layout

```
src/attngnn/     data, models, training, scores, cascade simulation, metrics, explanations, runs, CLI, dashboard backend
app/             Streamlit dashboard (streamlit_app.py, ui_theme.py)
configs/         experiment settings (default.yaml, smoke.yaml)
scripts/         setup, data download, dataset validation, results/report/slide builders, verification
tests/           unit, integration and UI tests
artifacts/       runs/ (one folder per run), results/ (tables, figures, results_index.json, facts.json)
reports/         report (PDF, DOCX, MD), slides (PPTX), demo script
docs/            literature survey, protocol, dataset validation, screenshots, diagrams
samples/         small CSV graph for the dashboard's upload demo
.github/         CI workflow
```

## Quick start (Windows, local)

Run every command from the project folder in PowerShell (`cd E:\AttentionGNN`).

1. **Set up once** (about 5–10 minutes, needs internet): creates `.venv`, installs pinned dependencies, downloads and verifies the data.
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
   ```
2. **Smoke test** (about 1 minute; tiny budget, *not* results):
   ```powershell
   .\.venv\Scripts\python.exe -m attngnn.cli smoke
   ```
3. **Dashboard, localhost only** (opens `http://localhost:8501`; stop with `Ctrl+C`):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\run_dashboard.ps1
   ```
4. **Tests:**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest
   ```
5. **Full benchmark** (optional; Cora about 25 min, Weibo about 2 h on a 4-core laptop; resumable):
   ```powershell
   .\.venv\Scripts\python.exe -m attngnn.cli benchmark
   .\.venv\Scripts\python.exe -m attngnn.cli sensitivity      # optional injection-rate sensitivity
   ```
6. **Rebuild tables, figures, report and slides** (Windows with Microsoft Word and PowerPoint):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\build_docs.ps1
   ```
7. **Dataset authenticity check** (internet; about 1 minute): `.\.venv\Scripts\python.exe scripts\validate_datasets.py`
8. **Full verification** (checkpoints reproduce, results index, PPTX/PDF, citations): `.\.venv\Scripts\python.exe scripts\verify_all.py`

Linux / macOS: `pip install -r requirements-dev.txt && pip install -e . && python scripts/download_data.py`, then use the same
`python -m attngnn.cli ...`, `python -m pytest` and `streamlit run app/streamlit_app.py --server.address localhost`.
(Report and slide building needs Windows with Microsoft Office.)

## Dashboard pages

| Page | Purpose |
|---|---|
| Overview | animated summary, headline findings, scope banner |
| Data & graph summary | built-in datasets or your own CSV graph (try `samples/`) with clear validation errors |
| Dataset authenticity | the 42 integrity checks and the upstream-label finding |
| Train / evaluate | start runs (smoke or full) with live progress; reload a checkpoint and verify it reproduces |
| Influence ranking | ranked nodes vs degree, PageRank, betweenness and simulated spread |
| Anomaly scores | threshold slider with live score distribution; tuning on validation only, test metrics at the frozen threshold |
| Graph explorer & node inspector | attention-weighted neighbourhood, per-layer/head view, cascade animation |
| Results: comparison & ablation | tables with mean ± std, paired differences, ablations, cost |
| Demos | known anomaly, high-spread node, explanation test, model reload, training replay, ROC drawing |
| Exports & documents | predictions, config, metrics, graph image, report and slides |

## Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub (the repo can stay **private**).
2. Open <https://share.streamlit.io>, sign in with GitHub, and click **Create app** (private repos work after you grant access).
3. Repository: `KrishKapoor2507/attentionGNN`, branch `main`, main file path `app/streamlit_app.py`.
4. **Advanced settings → Python version: 3.11**. Click **Deploy**.
5. First start installs the dependencies and downloads the pinned datasets automatically (a few minutes).
   Pages that launch training jobs work but are slow on the free tier; viewing the saved runs is instant.

The deployed app is reachable by anyone with its link unless you restrict viewers in the Streamlit sharing settings; it shows
saved results, not private data.

## Continuous integration
`.github/workflows/ci.yml` runs on every push and pull request: install pinned dependencies, download and verify the data,
compile all sources and run the full test suite. A weekly job re-runs `scripts/validate_datasets.py` so a silent upstream
data change is caught.

## Reproducibility
Every run saves its exact command, configuration, dataset / split / feature / label SHA-256, seeds, software and hardware
versions, source-tree hash, logs, checkpoint with hash, per-node predictions and metrics. `scripts/build_results.py`
recomputes every table and figure from those files (means and standard deviations from per-seed values), and
`scripts/verify_all.py` checks that checkpoints reproduce their saved predictions.

## Troubleshooting
* **`py` not recognised**: install Python 3.11 from python.org (tick *Add to PATH*) and reopen PowerShell.
* **Scripts disabled**: use the `powershell -ExecutionPolicy Bypass -File ...` form shown above.
* **Dataset download fails**: open `https://github.com/pygod-team/data/raw/13e75cd16e5b674ac84fd6adfea8286ef0b69203/weibo.pt.zip`
  (pinned commit; do **not** use `main`), save it in `data\raw\`, run `python scripts\download_data.py` (it verifies the SHA-256).
* **Port busy**: add `--server.port 8600` to the dashboard command.
* **VS Code cannot select an interpreter**: not needed; use the full command `.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.address localhost`.
* **Out of memory on Weibo**: close other programs; a run needs about 1.3 GB.

## Honest limits
Influence is **simulated** independent-cascade spread, not observed influence. Cora anomalies are **injected** (synthetic).
Evaluation is transductive, with 3–5 seeds and a CPU budget (Weibo training subsamples 30 % of arcs per epoch). Not run:
Reddit and other BOND datasets (configured, `python -m attngnn.cli benchmark --datasets reddit`), Radar-type baselines,
GNNExplainer comparison, measured cascades. AI assistance was used to build this project; follow your institution's
disclosure rules (see `metadata.yaml`).

## Key references
Kipf & Welling (GCN, 2017) · Veličković et al. (GAT, 2018) · Brody et al. (GATv2, 2022) · Ding et al. (DOMINANT, 2019) ·
Liu et al. (BOND, 2022) · Kempe et al. (influence maximisation, 2003) · Ying et al. (GNNExplainer, 2019) ·
Gu et al. (DeepLinker/GANR, attention rank). Full verified list: `docs/literature/references.bib`.
