# PROGRESS / HANDOFF

Last updated: 2026-10-26 (second iteration: dataset validation, corrected Weibo labels, animated dashboard)

## Environment facts (measured)
- Windows 11, Python 3.11.9, Intel i7-1165G7 (4C/8T), 15.8 GB RAM, no NVIDIA GPU -> CPU only.
- venv `E:\AttentionGNN\.venv` (torch 2.14.1+cpu, torch_geometric 2.8.1); pinned in `requirements.txt`.
- MS Word + PowerPoint + Edge used for PDF export / slide rendering / screenshots. No LibreOffice, no pandoc.
- Not a git repository (git/deployment deliberately left for later); provenance stores a source-tree SHA-256.
- Dashboard is served on localhost only (`.streamlit/config.toml`).

## What changed in this iteration
1. **Dataset authenticity** (`scripts/validate_datasets.py`, `docs/dataset_validation.md`): re-download vs pinned upstream commit, structure, published statistics, label semantics, Cora injection integrity, splits. 42 checks pass, 0 fail.
2. **Found a real data problem**: the current upstream `weibo.pt` (Jan 2024, "add mask") blanks the labels of all validation/test-mask nodes (347 of 868 anomalies stay labelled). My first Weibo benchmark had used it. Fixed: data pinned to the original 2022 commit 13e75cd (SHA-256 in `scripts/download_data.py`), **all Weibo runs repeated** (36 runs + profiles). The corrupted runs are kept in `artifacts/superseded/weibo_2024_masked_labels/` and compared in `artifacts/results/tables/label_version_weibo.csv` (ROC-AUC lower by 0.015-0.05, AP about 3x lower).
3. Preprocessing check repeated with the authentic labels (`artifacts/exploration/weibo_preprocessing_authentic_labels.json`): plain standardisation 0.752 > clip 0.669 > sign-log 0.614 (validation AUC).
4. New RQ4 result with correct data: high-attention edge removal matters more than random for anomaly scores (Cora +0.019, Weibo +0.067) and for influence on Weibo (+0.298, attention sharp) but not on Cora (attention uniform).
5. Dashboard redesigned: animated hero canvas, count-up KPIs, animated finding bars, live job tracker, threshold slider with moving score distribution, cascade animation, training replay, ROC drawing, Dataset authenticity page (CSS animations are scoped to key= containers and respect reduced-motion).
6. Report, deck, README, demo script, screenshots rebuilt. Krish's e-mail set to kapoorkrishaug25@gmail.com.

## Checklist
- [x] Everything from the first iteration (see previous sections of the report) re-verified
- [x] 82 tests pass (`python -m pytest`), `scripts/verify_all.py` passes
- [ ] **Author edits (manual):** A.M. Karthikeshwaran's e-mail, the AI-assistance statement (and optionally guide name) in `metadata.yaml`, then `scripts\build_docs.ps1`
- [ ] Later (by request): git repository, deployment

## Not run (listed honestly)
- Reddit and other BOND datasets (configured, `optional: true`; `python -m attngnn.cli benchmark --datasets reddit`)
- Radar/ANOMALOUS baselines, GNNExplainer comparison, measured cascades, inductive evaluation, ablation of Weibo arc subsampling
- Similarity / AI-detector checks (not available here; no claim made)
- Dataset licences for the underlying Weibo/Reddit data (not stated in the sources opened)

## Decisions and caveats
- Weibo training uses DropEdge-style arc keep 0.3 (CPU budget); inference uses all arcs.
- Cora weight decay 5e-4 and 300 training cascades were chosen from validation metrics; exploratory runs also printed test metrics.
- Cora benchmark timings were taken while other work shared the CPU (about 2x the isolated profile); Weibo timings were measured on an idle machine.
- Code identity: runs record the source-tree hash at run time; files edited later (dashboard, builders, data pinning) post-date the Cora runs, but the modelling code used by the runs (`models.py`, `train.py`, `scoring.py`, `experiments.py`, `influence.py`, `metrics.py`) was not changed after the Cora benchmark.

## How to resume
`python scripts/validate_datasets.py` (data), `python scripts/verify_all.py` (everything), `scripts\build_docs.ps1` (tables, figures, report, slides).
