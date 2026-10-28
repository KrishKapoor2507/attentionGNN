# Experimental protocol (authoritative description of what the code does)

Everything below is implemented in `src/attngnn/` and recorded in each run's `config.json`.

## 1. Graph conventions
* Graphs are made symmetric, de-duplicated and self-loop-free (`data.clean_edges`); counts of removed items are stored in `data.json`.
  Weibo is directed in its source file (346,579 of 407,963 listed arcs have no reverse arc); we symmetrise it. Reddit contains 10,984 self-loops that are dropped.
* `edge_index[0]` = source *j*, `edge_index[1]` = target *i*. Messages and attention flow j -> i. The GNN layers add self-loops internally.

## 2. Splits, normalisation, leakage control
* Seeded, stratified (by anomaly label) node split 30 % train / 20 % validation / 50 % test (`data.make_splits`, SHA-256 stored).
* **Transductive**: the encoder sees all nodes' features and all edges (including validation/test nodes) but no labels and no influence targets of validation/test nodes.
* Normalisation statistics (mean/std) come from **train nodes only**. Weibo: standardisation (no clipping). Cora: raw binary bag-of-words.
* Anomaly training is label-free. Validation labels are used for (i) the reconstruction mixing weight alpha_sel in {0, .25, .5, .75, 1} (best validation ROC-AUC) and (ii) the F1-optimal decision threshold per score. Both are frozen before test evaluation. **This makes model selection label-informed even though encoder training is unsupervised.**
* Combination score weights are predeclared (0.5/0.5 of train-robust-z-scored `rec` and `att_only`), not tuned.
* Training hyper-parameters are shared by all variants of a dataset (Adam, lr 0.005, weight decay 5e-4, fixed epochs, no early stopping, no test-driven tuning). Weight decay and the number of Cora training cascades were chosen from validation metrics (see PROGRESS.md log).

## 3. Model (see report Sec. 3.2 for equations)
Two-layer encoder (hidden 64): GAT (4 heads x 16, concatenated), GATv2 (same widths), or GCN. Attribute decoder: one conv of the same family + linear map to d. Structure decoder: sigmoid(z_i . z_j) on sampled positive arcs (<= 200k per epoch) and uniformly sampled negative pairs (ratio 1:1, seed recorded). Influence head: MLP 64 -> 32 -> |p-grid|.
Weibo training uses DropEdge-style random arc subsampling (keep 30 %) for CPU feasibility; inference always uses the full graph.

## 4. Anomaly scores
`attr_err`, `struct_err`, `rec = alpha_sel z(attr_err) + (1-alpha_sel) z(struct_err)`, `att_only` (attention-weighted feature discrepancy over non-self in-neighbours, head/layer-averaged GAT coefficients, no decoder output), `feat_dev_uniform` (same with uniform weights: the non-attentive control), `combined`. z = robust standardisation with train-node median/MAD.
Metrics: precision/recall/F1 at the validation-selected threshold, ROC-AUC, average precision (step-wise, scikit-learn), precision@k (k = number of positives; also @50), class support, confusion matrix, PR/ROC curves. One-class splits return "undefined", never 0.

## 5. Influence
* Simulation proxy: independent cascade on the symmetrised graph (each directed arc tested once with probability p). Counter-based RNG; expected spread of a lone seed.
* p grids (below the supercritical threshold measured on each graph): Cora {0.05, 0.10, 0.15}; Weibo {0.002, 0.003, 0.005}. Repetitions: Cora train/val/test = 300/300/1000, Weibo 100/100/500; **independent random streams** for train, validation and test nodes.
* Candidates for all ranking methods = the same held-out test nodes. Ground truth = test-stream spread (SE stored). Training-set targets supervise the influence head only.
* Methods: degree, PageRank (alpha 0.85), betweenness (exact on Cora, 200-pivot approximation on Weibo), uniform-attention control, received attention (raw / OLS-degree-adjusted / per-degree), influence head.
* Metrics: Spearman, Kendall tau-b, NDCG@k (tie-averaged), top-k overlap (fractional ties; relevant set = true top-k among candidates; k = 10, 50, 5 %), seed-set spread of the top-10/20 chosen by each method (fresh 500-cascade stream, common random numbers), with random-k and oracle-top-k references. Individual spread != seed-set spread (seed sets can be redundant).

## 6. Explanation perturbation (association test)
For 20 target nodes per run (test anomalies for the anomaly output; top predicted-influence test nodes for the influence output) remove m = min(3, deg//2) undirected edges chosen by highest attention, lowest attention, or uniformly at random (3 draws averaged), recompute the frozen model's output with an identical full forward pass and compare the output drop. Attention on arriving arcs for the anomaly output, on leaving arcs for the influence output. Paired bootstrap CIs and Wilcoxon tests are over targets pooled across seeds.

## 7. Provenance per run
`config.json`, `command.txt`, `env.json` (versions, CPU, RAM), `data.json` (graph summary, split/edge/feature/label SHA-256, influence-target cache key), `code.json` (source-tree SHA-256; git head/dirty/diff hash when the folder is a git repository), `log.txt`, `history.json`, `checkpoint.pt` (hash in `metrics.json`), `predictions.csv.gz`, `curves.json`, `metrics.json`. `scripts/build_results.py` rebuilds every table and figure from these files and writes `artifacts/results/results_index.json`.
