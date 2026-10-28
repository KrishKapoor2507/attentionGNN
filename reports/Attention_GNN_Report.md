# Attention-Driven Graph Neural Networks for Influence and Anomaly Detection in Complex Networks

**Krish Kapoor** (23MIC0023), krish.kapoor2023@vitstudent.ac.in  
**A.M. Karthikeshwaran** (23MID0215), karthikeshwaran.am2023@vitstudent.ac.in  
School of Computer Science and Engineering (SCOPE), VIT Vellore  
CSI1006 (Fifth Semester) - Project Review

## i. Abstract

Influence ranking and anomaly detection in networks are usually handled by separate methods, and neither explains why a node scores as it does. This project asks whether the neighbour weights learned by a graph attention network (GAT) can serve both tasks, and tests the idea instead of assuming it. We implement a shared two-layer multi-head GAT encoder (with GATv2 and GCN controls) on attributed graphs, together with an attention-only anomaly score, a reconstruction score, a combined score, a received-attention influence score with and without degree adjustment, and a small influence head trained on independent-cascade simulations. Everything runs on a laptop CPU. We evaluate on the Weibo organic-anomaly benchmark (8,405 users, 868 labelled anomalous; we validated the data against a pinned upstream commit and found that the current upstream file blanks the validation and test labels, so we use the original 2022 file) and on Cora with seeded injected anomalies, using three and five seeds, against Isolation Forest, a DOMINANT-style GCN, degree, PageRank and betweenness, plus non-attentive controls. Influence is judged against held-out simulated spread, which is a proxy and not observed influence. Reconstruction reaches a test ROC-AUC of 0.771 ± 0.031 on Cora and 0.771 ± 0.026 on Weibo (GCN: 0.765 ± 0.030 and 0.791 ± 0.011). The attention-only score reaches 0.796 ± 0.034 on Cora but is matched by a uniform-weight control (0.790 ± 0.036), and on Weibo it is weaker (0.609 ± 0.049). Received attention does not beat degree for influence (Spearman -0.198 ± 0.055 against 0.629 ± 0.003 on Weibo) and equals a uniform-attention control on Cora, while a plain GCN influence head performs best (0.889 ± 0.005 on Weibo). Removing high-attention edges lowers anomaly scores more than removing random edges on both datasets, and lowers influence predictions more on Weibo, where attention is sharp, but not on Cora, where it is nearly uniform; attention therefore describes what the trained model uses without being a better predictor than simple controls. In these experiments joint training gave no reliable benefit. The conclusions are limited to these graphs, proxies and CPU budgets.

## ii. Keywords

graph attention network; influence ranking; anomaly detection; attributed networks; independent cascade; centrality; interpretability; limited-compute learning

## 1. Introduction

Social, communication and interaction networks are naturally described as graphs whose nodes carry attributes. Two questions are routinely asked of such graphs. Which nodes are structurally influential, in the sense that activity starting at them reaches many others? And which nodes behave unusually given their attributes and their neighbourhood? In practice these questions are answered by separate pipelines: classical centralities such as degree, PageRank and betweenness [1, 2, 3] for influence, and feature-based or reconstruction-based detectors [4, 5] for anomalies. Centralities treat every neighbour alike and say nothing about why a node matters, while deep detectors usually give a score without an account of which neighbours drove it.

Graph attention networks [6] assign a learned weight to every neighbour of a node. The proposal behind this project asks whether those learned weights can serve as a shared signal: the attention a node receives from its neighbours as an influence indicator, and the discrepancy between a node and its attention-weighted neighbourhood as an anomaly indicator. We treat this as a hypothesis to be tested, not as a result to be confirmed, and we test it on a CPU-only budget.

This report documents an implementation and an evaluation, not a new architecture. The encoder is a standard multi-head GAT, with GATv2 [7] and GCN [8] as controlled alternatives. Earlier work already uses received attention as a node-ranking score for link-prediction-trained encoders [9, 10] and already places attention inside anomaly detectors [11, 12]. What we contribute is (i) a single shared encoder evaluated on both tasks under one protocol, (ii) a careful treatment of what attention can and cannot rank (normalisation, direction, degree), (iii) comparisons against classical baselines and non-attentive controls on the same held-out nodes, and (iv) a perturbation test of whether high-attention edges matter more to the model than random ones.

### 1.1 Problem Formulation / Objectives

Let G = (V, E, X) be an attributed graph with N nodes, symmetric edge set E and attribute matrix X in R^(N x d). We want one learned representation Z = f(X, E) from which we obtain (a) an anomaly score s_a(i) for each node, and (b) an influence ranking r(i). Anomaly labels y_i in {0, 1} exist for evaluation only. Influence has no observed label in our data, so we use the expected spread sigma(i) of an independent-cascade process started at node i as a simulation proxy.

The objectives, taken from the proposal and made testable, are:

1. Represent the networks as attributed graphs and document the preprocessing, splits and label provenance (Sections 3.1 and 5.1).
2. Train a multi-head attention encoder under a limited CPU budget and record its cost (Sections 3.2 and 5.2).
3. Test whether attention-derived signals rank influential nodes and flag anomalies, including attention-only scoring rules declared in advance (Sections 3.2 and 5).
4. Benchmark against degree, PageRank, betweenness, Isolation Forest, a DOMINANT-style GCN reconstruction model and non-attentive controls (Section 5.2).
5. Evaluate with precision, recall, F1, ROC-AUC, average precision, rank correlations, NDCG, top-k overlap and seed-set spread, and validate the attention-based interpretation by edge-removal tests (Sections 4 and 5).

Research questions: **RQ1** does attention-only scoring detect anomalies as well as reconstruction? **RQ2** does received attention rank held-out spread better than degree and PageRank, and does degree adjustment help? **RQ3** does joint training help either task relative to task-only training? **RQ4** do high-attention edges matter more to the model output than equally many random or low-attention edges?

## 2. Related Work

**Graph encoders.** GCN [8] aggregates neighbours with fixed degree-normalised weights, so its weights are not uniform but they are not learned per edge either. GAT [6] learns per-edge coefficients with a softmax over each node's incoming neighbours. GATv2 [7] shows that the original scoring function is static, meaning the ranking of neighbours does not depend on the query node, and proposes a dynamic variant. Under static attention, a source's received attention is partly a node-wide quantity, which is relevant to how we interpret it for influence.

**Graph anomaly detection.** Akoglu et al. [13] organise the field by data type and anomaly kind (structural, contextual, community). DOMINANT [5] trains a graph autoencoder and scores nodes by attribute and structure reconstruction error. AnomalyDAE [11] adds attention in the structure encoder, and ResGCN [12] uses residual-based attention. What counts as anomalous depends on the neighbourhood scale considered [14], which warns against a single fixed scale. The BOND benchmark [15] (with the PyGOD library [16]) separates injected from organic anomalies and reports that detectors that look strong on injected anomalies can be near chance on organic ones; on Weibo, classical non-deep methods already perform very well. Supervised benchmarks such as GADBench [17] are not comparable with our unsupervised protocol. The proposal cites a "Zhang et al. (2022), Attention-based Anomaly Detection on Graphs". We searched arXiv, Crossref and OpenAlex and could not verify a paper with that title and author; we therefore do not cite it, and describe the real near-matches above instead (docs/literature/zhang2022_verification.md).

**Influence.** The independent-cascade model of Kempe et al. [18] defines the expected spread of a seed set and underlies seed-selection research, including GNN-based selectors [19, 20, 21]. Closest to our ranking idea, DeepLinker and GANR [9, 10] rank nodes by the normalised attention their neighbours direct at them, so received attention as a centrality signal is established; our additions are the degree adjustment, the held-out cascade evaluation and the joint use with anomaly scoring.

**Attention and explanation.** Whether attention explains a prediction is contested: Jain and Wallace [22] found weak agreement with gradient-based importance, and Wiegreffe and Pinter [23] argued the answer depends on how explanation is tested. For graphs, GNNExplainer [24] identifies prediction-relevant subgraphs and features, and later benchmarks report that explainers are fragile on attributed and heterophilous graphs [25]. We therefore treat attention as an association and test it by perturbation.

**Positioning.** The ingredients are not new. We make no claim of a first unified model or a novel architecture; the contribution is the integrated, controlled empirical study below, and its conclusions are limited to the data and budget used.

## 3. Proposed Work: Attention-Driven Graph Neural Networks for Influence and Anomaly Detection in Complex Networks

### 3.1 Graph Representation and Data Preparation

Every graph is stored as attributes X and an arc list (source j, target i) that is symmetrised, de-duplicated and free of self-loops; self-loops are added inside the message-passing layers. We use two datasets.

**Weibo** (PyGOD/BOND file [15, 16]) is a user-user graph with 400-dimensional user features in which a label marks users with suspicious posting behaviour. These labels are organic, not injected. The source file is directed (346,579 of 407,963 listed arcs have no reverse arc); we symmetrise it. We use the original 2022 file (commit 13e75cd of the PyGOD data repository, SHA-256 pinned in `scripts/download_data.py`), which contains 868 anomalous users (10.3 %), the number given in the benchmark paper. The current upstream file (January 2024, commit "add mask") adds train/validation/test masks and sets the labels of every validation and test node to normal, so only 347 of the 868 anomalies remain labelled; evaluating on it would treat 521 true anomalies as normal. Our first Weibo benchmark used that file. The dataset validation of Section 5.1 exposed the problem; we discarded those runs (kept in `artifacts/superseded/` for transparency) and repeated every Weibo experiment with the original labels.

**Cora** [26, 27] is a citation graph with 2,708 nodes, 1,433 binary features and seven topic classes. Topic classes are not anomaly labels. We inject synthetic anomalies with a seeded protocol following the scheme used by DOMINANT [5]: seven disjoint cliques of ten nodes (structural), and 70 nodes whose attributes are replaced by those of the farthest of 50 random candidates (contextual), 140 nodes in total (5.2 %). These labels are synthetic and are never described as observed fraud. The injection seed also determines the split and the model initialisation.

Nodes are split 30/20/50 percent into train, validation and test sets, stratified by label and seeded; the split hash is stored with each run. Normalisation statistics come from the train nodes only. The setting is **transductive**: the encoder sees all node attributes and all edges, including those of validation and test nodes, but never their anomaly labels or influence targets. Weibo has extremely heavy-tailed attributes (standardised values reach 1,814); in a validation-only exploration, clipping them at five standard deviations or compressing them logarithmically removed label signal (validation ROC-AUC of the attribute error, seed 0: 0.669 and 0.614, against 0.752 with plain standardisation), so we use plain standardisation. This check was first run on the corrupted label file and repeated with the original labels, which gave the same ordering.

**Influence proxy.** No measured cascades are available. We use the independent-cascade model [18] on the symmetrised graph: each directed arc is tested once with probability p when its source becomes active, and the target is the expected number of activated nodes when a single node is the seed. The probability grids lie below the point where cascades become giant (Cora p in {0.05, 0.10, 0.15}; Weibo p in {0.002, 0.003, 0.005}). Spread targets of train, validation and test nodes come from independent random streams with 300/300/1000 repetitions on Cora and 100/100/500 on Weibo; the test-stream estimates are the held-out ground truth. These are simulation outputs, never observed real-world influence.

### 3.2 Attention Encoder, Joint Objectives, Influence Ranking and Anomaly Scoring

**Encoder.** Layer l maps node states z_i^(l) to z_i^(l+1) with H heads of width 16 that are concatenated to width 64. For GAT [6], with N(i) the in-neighbours of i including i itself:

$$e_{ij}^{h}=\mathrm{LeakyReLU}_{0.2}\left(\mathbf{a}_{h}^{\top}\,[\,W^{h}z_{i}\,\Vert\,W^{h}z_{j}\,]\right),\qquad \alpha_{ij}^{h}=\frac{\exp(e_{ij}^{h})}{\sum_{k\in N(i)}\exp(e_{ik}^{h})}$$

$$z_{i}^{(l+1)}=\mathrm{ELU}\left(\Vert_{h=1}^{H}\sum_{j\in N(i)}\alpha_{ij}^{h}\,W^{h}z_{j}^{(l)}\right)$$

GATv2 [7] replaces the score by a^T LeakyReLU(W [z_i || z_j]). The GCN control [8] uses the degree-normalised propagation Z' = ELU(D^(-1/2) (A + I) D^(-1/2) Z W). Two layers give the latent matrix Z (N x 64). Edge convention: the coefficient alpha_ij belongs to the arc j -> i and is normalised over the arcs arriving at i.

**Heads and objectives.** The attribute decoder is one graph convolution of the same family followed by a linear map, giving X_hat in R^(N x d). The structure decoder scores a pair with p_ij = sigmoid(z_i . z_j), evaluated only on sampled pairs: up to 200,000 real arcs per epoch and the same number of uniformly sampled non-edges (ratio 1:1, seed stored), so no dense N x N matrix is formed. An influence head (MLP 64-32-|p|) predicts the standardised log(1 + spread) at each cascade probability. With normalised reconstruction loss and train-node influence targets, the training objective is

$$\mathcal{L}=w_{rec}\left[a\,\mathcal{L}_{attr}+(1-a)\,\mathcal{L}_{struct}\right]+\lambda\,\mathcal{L}_{inf}$$

$$\mathcal{L}_{attr}=\frac{\Vert X-\hat{X}\Vert_{F}^{2}}{N\,d\,\overline{x^{2}}},\qquad \mathcal{L}_{inf}=\frac{1}{|V_{tr}|}\sum_{i\in V_{tr}}\Vert\hat{\sigma}_{i}-\sigma_{i}\Vert^{2}$$

$$\mathcal{L}_{struct}=\mathrm{BCE}\left(\sigma(z_{i}^{\top}z_{j}),\,\mathbf{1}[(j,i)\in E]\right)$$

with a = 0.5, lambda = 1; w_rec = 0 gives influence-only training and lambda = 0 gives anomaly-only training. This joint model, with a shared encoder and task-specific outputs, is our extension beyond the proposal's stronger hypothesis that attention alone suffices; the attention-only scores below test that stronger hypothesis separately. Unit tests verify that each of the three losses sends gradient into the shared encoder. Adam [28] is used with learning rate 0.005 and weight decay 5e-4. On Weibo the message-passing arcs are randomly subsampled to 30 % in each training epoch (DropEdge-style [29]) to fit the CPU budget; inference always uses the full graph.

**Anomaly scores.** All scores are per node and larger means more anomalous. Let z(.) denote robust standardisation with the median and MAD of the train nodes.

$$s_{attr}(i)=\Vert x_{i}-\hat{x}_{i}\Vert_{2},\qquad s_{struct}(i)=\frac{1}{|N(i)|}\sum_{j\in N(i)}(1-p_{ij})^{2}+\frac{1}{K}\sum_{k\in K_{i}}p_{ik}^{2}$$

$$s_{rec}(i)=a_{sel}\,z(s_{attr})+(1-a_{sel})\,z(s_{struct})$$

where K_i are 20 sampled non-neighbours and a_sel is chosen from {0, 0.25, 0.5, 0.75, 1} by validation ROC-AUC. The **attention-only score** is declared in advance as the attention-weighted feature discrepancy to the neighbours,

$$s_{att}(i)=\sum_{j\in N(i)\setminus\{i\}}\tilde{\alpha}_{ij}\,\frac{\Vert x_{i}-x_{j}\Vert_{2}}{\sqrt{d}},\qquad \tilde{\alpha}_{ij}=\frac{\bar{\alpha}_{ij}}{\sum_{k\neq i}\bar{\alpha}_{ik}}$$

where the bar denotes the average over heads and layers; it uses no decoder output. Replacing alpha-tilde by 1/deg(i) gives the non-attentive control **uniform-neighbour deviation**, and an encoder with random weights gives an untrained-attention control. The **combined score** is 0.5 z(s_rec) + 0.5 z(s_att) with fixed weights. The decision threshold is the F1-optimal value on the validation nodes and is frozen before test evaluation. Because validation labels choose a_sel and the threshold, model selection is label-informed even though encoder training is unsupervised.

**Influence ranking.** Because the softmax normalises over the arcs arriving at a target, the coefficients arriving at any node sum to exactly one, so summing them at a target cannot rank anything (a unit test confirms this). We instead define the **received attention** of a source j as the attention its neighbouring targets assign to it,

$$R_{j}=\sum_{i:\,(j\to i)\in E,\;i\neq j}\bar{\alpha}_{ij}$$

with the head and layer average shown above. Isolated nodes receive zero. Since R depends on degree, we compare the raw score, R_j / deg_j, and the **degree-adjusted score**, the residual of log R_j after an ordinary-least-squares fit on log(1 + deg_j) over train nodes. The non-learned control R_uniform_j = sum over targets of 1/(deg_i + 1) shows what uniform attention would give. The influence head output is the second model-based ranking. Baselines are degree, PageRank with damping 0.85 [2], and betweenness (exact on Cora, a 200-pivot approximation [30] on Weibo).

### 3.3 System Architecture

![System architecture of the implemented pipeline. Orange marks the training objective; green marks stored evidence and user-facing outputs.](E:/AttentionGNN/docs/figures/architecture.png)

The pipeline is a Python package (`attngnn`) that runs entirely on the local CPU. Each run, defined by dataset, variant and seed, writes a complete record: configuration, exact command, environment versions, data and split checksums, source-tree hash, log, checkpoint with hash, per-node predictions and machine-readable metrics. A separate builder recomputes every table and figure from those records and writes an index linking each result to its runs. A Streamlit dashboard reads the same records, and can launch new runs as background jobs. The implementation uses PyTorch Geometric [31].

### 3.4 Functional Block Diagram

![Functional block diagram. Labels are used only in the orange step (validation) and in evaluation; the encoder is trained without anomaly labels.](E:/AttentionGNN/docs/figures/block_diagram.png)

Part A of the diagram is run once per seed; part B uses the frozen model on held-out test nodes, and every compared method is scored on exactly the same nodes.


## 4. Experimental Analysis and observations

**What was run.** The benchmark contains 96 saved runs: twelve methods (three baselines, nine GNN variants) on injected Cora with five seeds and on Weibo with three seeds. The seed fixes the split, the model initialisation, the negative-sampling stream and, for Cora, the injected anomalies and the cascade simulations. All runs used one Intel i7-1165G7 CPU (four cores) with 15.8 GB of memory and no GPU, Python 3.11, PyTorch 2.14.1 (CPU) and PyTorch Geometric 2.8.1. Means and standard deviations in all tables are recomputed from the per-seed values by `scripts/build_results.py` (sample standard deviation); `[n=k]` in the raw CSV files gives the number of seeds, and tables here show the mean ± standard deviation. Three seeds (Weibo) or five seeds (Cora) support only coarse statements about uncertainty, so comparisons below also use differences paired by seed (same graph, split and cascade streams).

Table: Datasets and splits used by the benchmark (anomaly counts per split are for seed 0).
Table: 
| dataset | nodes | undirected edges | features | anomalies | anomaly rate | mean degree | max degree | split sizes (train/val/test) | anomalies per split (seed 0) | label provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| cora_injected | 2708 | 5593 | 1433 | 140 | 5.170% | 4.1 | 168 | 812/542/1354 | 42/28/70 | INJECTED anomalies (synthetic, DOMINANT-style protocol); NOT observed fraud. 1 = contextual, 2 = structural. |
| weibo | 8405 | 377271 | 400 | 868 | 10.327% | 89.8 | 4359 | 2521/1681/4203 | 260/174/434 | organic anomaly label of the ORIGINAL 2022 benchmark file (868 suspicious users; BOND Appendix A.1.1) |

**Anomaly detection (RQ1).** On injected Cora the reconstruction score of the GAT reaches a test ROC-AUC of 0.771 ± 0.031, against 0.765 ± 0.030 for the DOMINANT-style GCN and 0.676 ± 0.027 for Isolation Forest. The attention-only score reaches 0.796 ± 0.034. The two non-attentive controls are almost as good: uniform-neighbour feature deviation 0.790 ± 0.036 and the attention of an untrained GAT 0.791 ± 0.036. Paired by seed, the trained attention-only score exceeds the uniform control by only +0.006 ROC-AUC (positive in five of five seeds; see the paired-difference tables in Section 5.2), which is consistent in sign but much smaller than the seed-to-seed spread of about 0.03. We read this as: the useful signal is the feature discrepancy between a node and its neighbours; learned attention contributes at most a negligible amount. The combined score is the best on injected Cora (0.803 ± 0.033 for the joint model), because reconstruction and neighbour discrepancy capture different anomaly types.

On Weibo the picture differs. The best scores are the GCN reconstruction (0.791 ± 0.011) and the GATv2 reconstruction (0.773 ± 0.012), with the GAT at 0.771 ± 0.026; paired by seed the GAT is 0.020 below the GCN (below it in all three seeds). The attention-only score is weak (0.609 ± 0.049) and lies below the uniform control (0.630 ± 0.027) in every seed. Averaging attention-only and reconstruction scores lowers the ROC-AUC (0.714 ± 0.011, a paired loss of 0.057). Isolation Forest scores 0.458 ± 0.024, below chance: the benchmark's anomalous users are not isolated in attribute space, so a detector that looks for rare attribute values is misled. Average precision reaches 0.319 ± 0.020 for the best model against a base rate of 10.3 %, about three times the chance level, so even the best Weibo detector flags many normal users. The structure-error score alone has a mean ROC-AUC of 0.392 ± 0.122, below 0.5 and strongly seed dependent, meaning anomalous Weibo users tend to be reconstructed structurally *better* than normal ones; this agrees with the benchmark paper's observation that Weibo outliers are unusually clustered [15] (our own label check, Section 5.1, finds the same direction for clustering). The validation step therefore selected the attribute-only mixing weight (a_sel = 1) in every Weibo run that trained a reconstruction head.

**Influence (RQ2).** All rankings are evaluated on the same held-out test nodes against independent-cascade spread simulated with a separate random stream (Section 3.1). On Cora (p = 0.10), degree reaches a Spearman correlation of 0.752 ± 0.005, betweenness 0.639 ± 0.015 and PageRank 0.467 ± 0.011. The raw received attention of the GAT reaches 0.335 ± 0.014, identical within noise to the non-learned uniform-attention control (0.335 ± 0.013; paired difference 0.000). The reason is visible in the attention statistics (attention-statistics tables in Section 5.3): on Cora the learned attention is almost perfectly uniform, with a normalised entropy of about 0.98 to 1.0. On Weibo (p = 0.003) attention is sharper (entropy about 0.6), yet received attention correlates *negatively* with simulated spread over the whole test set (-0.198 ± 0.055), while degree reaches 0.629 ± 0.003. The negative correlation follows from the definition: because every target splits its attention, a source's received attention is roughly the sum over its neighbours of 1/(their degree), which is large for nodes sitting next to low-degree nodes and not for hubs. At the top of the ranking the picture is better but still behind degree: the NDCG@50 of received attention is 0.624 ± 0.024 on Weibo and 0.775 ± 0.033 on Cora, against 0.993 ± 0.001 and 0.867 ± 0.018 for degree. On Weibo the non-learned uniform-attention control ranks better than the learned attention (Spearman 0.109 ± 0.011, NDCG@50 0.954 ± 0.003) at every cascade probability we tried, so training the attention made this signal worse, not better; on Cora the two are identical at every probability. Degree adjustment, which removes the part of received attention explained by degree, makes the ranking much worse on both datasets (Spearman -0.507 ± 0.030 on Cora, -0.425 ± 0.023 on Weibo): in these graphs influence under the independent-cascade proxy is largely a degree effect, and what remains after removing degree is anti-correlated with it.

The learned influence head behaves very differently depending on the encoder. A plain GCN head is the best learned ranker (0.889 ± 0.005 on Weibo, 0.732 ± 0.020 on Cora) and beats degree on Weibo in every seed (paired +0.260), whereas the GAT head reaches only 0.402 ± 0.053 and 0.648 ± 0.043. A plausible reason is that GCN's degree-normalised aggregation carries degree information directly while the attention softmax removes it; we did not test this explanation separately. The influence head also overfits: its training loss falls to a few hundredths on Cora while held-out correlation stays moderate.

Individual spread and seed-set spread are different quantities. Choosing the ten nodes with the highest individual spread does not give the best ten-node seed set, because those nodes overlap: on Cora (p = 0.10) the oracle top-10 by individual spread reaches an expected set spread of 92.4 ± 8.2, below the 104.1 ± 8.4 reached by the degree-chosen set (influence tables in Section 5.2). All reported seed-set spreads come from a fresh cascade stream with common random numbers across methods.

**Joint versus task-only training (RQ3).** Joint training changes the anomaly ROC-AUC of the GAT reconstruction score by +0.005 on Cora (five seeds, positive in all five, but within the seed spread) and by +0.007 on Weibo (positive in one of three seeds; paired-difference tables in Section 5.2). For influence, the GAT head trained jointly is not better than the influence-only head: on Weibo the influence-only head is higher by 0.151 Spearman (all three seeds), on Cora lower by 0.031 in the joint model (one of five seeds in favour of joint training). We therefore find no evidence that sharing the encoder helps either task at this scale; the effect, if any, is small relative to the variance between seeds and between datasets.

**Does attention explain the model? (RQ4).** Removing the edges with the highest attention at an anomalous test node lowers the node's reconstruction-based anomaly score more than removing the same number of random edges, on both datasets: pooled over seeds the difference is +0.019 on Cora (95 % bootstrap interval excluding zero, Wilcoxon p = 0.0095) and +0.067 on Weibo (interval [+0.017, +0.143], p = 0.003); low-attention removals have the smallest effect. For the influence head the two datasets disagree. On Weibo, where attention is sharper, removing the highest-attention outgoing edges of a high-influence node lowers its predicted influence by +0.298 more than random removal (interval [+0.176, +0.430], p = 0.0001, 63 % of targets). On Cora, where attention is almost uniform, it does not (difference -0.138, interval including zero, p = 0.75). So attention marks edges that matter more to the model where it is non-uniform, which is what one would expect, but the effect sizes are modest and vary between seeds (edge-removal tables in Section 5.3). This is an association with the model's output, not evidence about the real network, and it matches the cautions in the literature [22, 23].

**Failure cases, preprocessing and unresolved issues.** (1) *Data version.* Our first Weibo benchmark used the current upstream file, in which the labels of all validation and test nodes were blanked (Section 5.1); it treated 521 of the 868 true anomalies as normal. We found this through the dataset validation, discarded those runs (kept in `artifacts/superseded/`), pinned the data to the original commit with a checksum and repeated every Weibo experiment. The comparison of the two sets of runs is in Section 5.1: the corrupted labels lowered ROC-AUC by 0.02 to 0.05 and cut average precision by a factor of about three. (2) Weibo features are extremely heavy-tailed; clipping or log-compressing them in a validation-only check gave lower validation ROC-AUC (0.669 and 0.614, against 0.752 with plain standardisation), so plain standardisation was used. Earlier exploratory runs also printed test metrics, and although every choice was made from validation numbers, we cannot exclude an unconscious influence. (3) Weibo training uses random arc subsampling (30 % per epoch) to fit the CPU budget; its effect on the final models was not ablated. (4) The influence targets are simulations; their standard error is stored per node and is small compared with the differences between methods, but the proxy's validity for real cascades is untested. (5) Weibo timings were measured on an otherwise idle machine; the Cora benchmark timings were taken while other work shared the CPU and are about twice the isolated profile, so the isolated profiles in Section 5.3 are the cleaner cost figures for Cora.

## 5. Experimental Validations

### 5.1 Test set (dataset) details and performance

**Dataset authenticity and validation.** Before any result is reported, `scripts/validate_datasets.py` checks that the data are what they claim to be. The summary below groups the individual checks (42 passed, none failed, two findings and one partial agreement; full list: `docs/dataset_validation.md` and the dashboard's *Dataset authenticity* page): SHA-256 of every downloaded file against a pinned upstream commit and against a fresh download, structural validity (finite features, valid edge ids, symmetric clean graph, binary labels), equality with the published node, edge, feature and anomaly counts, label semantics, the integrity of the Cora injection and the properties of the splits.

Table: Summary of the dataset validation (one row per dataset and check family).
Table: 
| dataset | check family | result | evidence |
|---|---|---|---|
| cora | Integrity against pinned upstream (SHA-256, fresh download) | 1/1 pass | 8 files |
| cora | Structural validity | 1/1 pass | 2708 nodes, 1433 features, 10556 directed arcs (= 5278 undirected; the GCN/GAT papers list 5,429 raw citation records), 7 classes |
| cora | Injection integrity | 5/5 pass | cliques complete=True, disjoint/counts=True (70 struct, 70 ctx), contextual copied=True, normal attributes untouched=True, same seed reproduces=True; rate 5.17% |
| cora (injected) | Structural validity | 4/4 pass | 2708 x 1433 |
| cora (injected) | Split integrity | 3/3 pass | 812(42)/542(28)/1354(70) nodes (anomalies) train/val/test |
| reddit | Integrity against pinned upstream (SHA-256, fresh download) | 2/2 pass | local 276432d36219349b..., pinned 276432d36219349b... |
| reddit | Structural validity | 4/4 pass | 10984 x 64 |
| reddit | Equals published statistics | 4/4 pass | 10984 vs 10984 |
| reddit | Label semantics | 1/1 pass | 0.000 vs 0.000 |
| reddit | Split integrity | 3/3 pass | 3295(110)/2197(73)/5492(183) nodes (anomalies) train/val/test |
| weibo | Integrity against pinned upstream (SHA-256, fresh download) | 3/3 pass, 2 finding(s) | pinned (2022): 868 anomalies; upstream main (2024): 347; main anomalies are a subset: True |
| weibo | Structural validity | 4/4 pass | 8405 x 400 |
| weibo | Equals published statistics | 4/4 pass | 8405 vs 8405 |
| weibo | Label semantics | 0/1 pass, 1 partial | clustering direction agrees: True; neighbour-similarity direction agrees: False. Our similarity is the mean cosine between a node and its neighbours on the raw features; BOND does not give its exact definition, |
| weibo | Split integrity | 3/3 pass | 2521(260)/1681(174)/4203(434) nodes (anomalies) train/val/test |

The validation produced one important finding and one open point. The finding: the current upstream Weibo file differs from the original in its labels only (features and edges are identical): its validation- and test-mask nodes were set to normal, so just 347 of 868 anomalies remain and none in the test mask. Only the original 2022 file reproduces the published statistics (8,405 nodes, 407,963 listed edges, 400 features, 868 anomalies), so it is the one used here. The open point: our node-versus-neighbour feature similarity for Weibo (0.445 for anomalies against 0.052 for normal users) does not reproduce the benchmark paper's 0.004 against 0.993, whereas the clustering difference does agree in direction (0.499 against 0.429). The paper does not define its similarity exactly, so we report this as partial agreement and make no claim about the semantic meaning of the Weibo labels beyond what the benchmark paper states (users with suspicious posting behaviour; the posting times themselves are not part of the data). The Cora checks confirm that each of the five injected datasets contains seven complete 10-cliques, 70 contextual nodes whose attributes are copies of other nodes' attributes, untouched normal nodes, and is exactly reproducible from its seed. Licences were not verified (the benchmark repository is MIT-licensed; the terms of the underlying Weibo and Reddit data are not stated in the sources we opened).

Table: Why the validation matters: Weibo anomaly metrics with the authentic labels against the superseded runs that used the corrupted upstream labels (test split, mean ± standard deviation over three seeds).
Table: 
| method | ROC-AUC, authentic labels (868 anomalies) | AP, authentic labels (868 anomalies) | ROC-AUC, corrupted 2024 labels (347 labelled) | AP, corrupted 2024 labels (347 labelled) |
|---|---|---|---|---|
| Isolation Forest (attributes) | 0.458 ± 0.024 | 0.117 ± 0.002 | 0.427 ± 0.017 | 0.041 ± 0.003 |
| GCN reconstruction | 0.791 ± 0.011 | 0.319 ± 0.020 | 0.750 ± 0.016 | 0.112 ± 0.006 |
| GAT reconstruction | 0.771 ± 0.026 | 0.288 ± 0.036 | 0.722 ± 0.050 | 0.101 ± 0.017 |
| GAT attention-only | 0.609 ± 0.049 | 0.253 ± 0.081 | 0.593 ± 0.082 | 0.105 ± 0.050 |
| GAT combined | 0.714 ± 0.011 | 0.294 ± 0.008 | 0.673 ± 0.024 | 0.113 ± 0.014 |
| Uniform-neighbour control | 0.630 ± 0.027 | 0.335 ± 0.038 | 0.615 ± 0.051 | 0.117 ± 0.034 |

The comparison shows the practical cost of a silent label problem. With 60 % of the anomalies labelled as normal, the models look worse (ROC-AUC lower by 0.015 to 0.05) and, much more strongly, average precision falls by a factor of about three, because every correctly flagged hidden anomaly is counted as a false positive. The ranking of methods happens to stay similar, so the qualitative conclusions would have survived, but the absolute numbers, and any comparison with published values, would not have been valid.

Dataset sizes, class balance and splits are in the dataset table of Section 4. Both datasets are evaluated transductively: the encoder sees the attributes and edges of validation and test nodes, but no labels and no influence targets of those nodes. The fraction of anomalies in each split is stratified, so the test split always contains both classes (70 anomalies of 1,354 test nodes on Cora with seed 0, and 434 of 4,203 on Weibo). The following per-seed values show the spread that the aggregate tables hide.

Table: Test ROC-AUC per seed, injected Cora (five seeds).
Table: 
| method (test ROC-AUC per seed) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| Isolation Forest (attributes) | 0.658 | 0.705 | 0.643 | 0.673 | 0.703 |
| Isolation Forest (attributes + structure) | 0.663 | 0.68 | 0.602 | 0.702 | 0.719 |
| GCN reconstruction (DOMINANT-style) | 0.799 | 0.759 | 0.719 | 0.766 | 0.784 |
| GAT reconstruction | 0.808 | 0.758 | 0.728 | 0.771 | 0.792 |
| GATv2 reconstruction | 0.79 | 0.758 | 0.727 | 0.769 | 0.789 |
| GAT attention-only (weighted feature discrepancy) | 0.836 | 0.776 | 0.758 | 0.782 | 0.829 |
| GAT combined (reconstruction + attention-only) | 0.836 | 0.781 | 0.755 | 0.788 | 0.828 |
| GAT joint, reconstruction score | 0.819 | 0.764 | 0.732 | 0.773 | 0.792 |
| GAT joint, attention-only score | 0.835 | 0.778 | 0.763 | 0.781 | 0.831 |
| GAT joint, combined score | 0.843 | 0.788 | 0.762 | 0.79 | 0.831 |
| GCN joint, reconstruction score | 0.811 | 0.754 | 0.757 | 0.778 | 0.781 |
| GATv2 joint, reconstruction score | 0.799 | 0.759 | 0.731 | 0.766 | 0.788 |
| Control: uniform-neighbour feature deviation (no attention) | 0.831 | 0.771 | 0.747 | 0.778 | 0.824 |
| Control: untrained-GAT attention-only | 0.831 | 0.772 | 0.747 | 0.778 | 0.826 |

Table: Test ROC-AUC per seed, Weibo (three seeds).
Table: 
| method (test ROC-AUC per seed) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Isolation Forest (attributes) | 0.468 | 0.43 | 0.475 |
| Isolation Forest (attributes + structure) | 0.461 | 0.472 | 0.439 |
| GCN reconstruction (DOMINANT-style) | 0.802 | 0.779 | 0.791 |
| GAT reconstruction | 0.78 | 0.742 | 0.79 |
| GATv2 reconstruction | 0.786 | 0.762 | 0.77 |
| GAT attention-only (weighted feature discrepancy) | 0.625 | 0.647 | 0.554 |
| GAT combined (reconstruction + attention-only) | 0.71 | 0.705 | 0.726 |
| GAT joint, reconstruction score | 0.823 | 0.723 | 0.787 |
| GAT joint, attention-only score | 0.64 | 0.652 | 0.548 |
| GAT joint, combined score | 0.772 | 0.689 | 0.713 |
| GCN joint, reconstruction score | 0.802 | 0.781 | 0.79 |
| GATv2 joint, reconstruction score | 0.777 | 0.773 | 0.732 |
| Control: uniform-neighbour feature deviation (no attention) | 0.64 | 0.651 | 0.6 |
| Control: untrained-GAT attention-only | 0.565 | 0.663 | 0.613 |

The per-seed tables show that on Cora the ordering of the broad groups is the same in every seed (Isolation Forest lowest, the reconstruction models next, the attention-only, combined and uniform-control scores highest), while differences inside a group change from seed to seed; and that Weibo, with only three seeds, has larger spread for the attention-based scores than for the GCN, which is itself informative: the attention-only score depends on the random initialisation more than a plain reconstruction does.

![ROC and precision-recall curves on injected Cora (test nodes, seed 0).](E:/AttentionGNN/artifacts/results/figures/roc_pr_cora_injected.png)

![ROC and precision-recall curves on Weibo (test nodes, seed 0).](E:/AttentionGNN/artifacts/results/figures/roc_pr_weibo.png)

![Test confusion matrices at the validation-selected thresholds, injected Cora, seed 0.](E:/AttentionGNN/artifacts/results/figures/confusion_cora_injected.png)

![Test confusion matrices at the validation-selected thresholds, Weibo, seed 0.](E:/AttentionGNN/artifacts/results/figures/confusion_weibo.png)

The confusion matrices show the practical consequence of the moderate average precision. At the F1-optimal validation threshold the Weibo GAT reconstruction detector flags about two normal users for every anomalous one it finds (156 true positives and 290 false positives, with 278 of the 434 test anomalies missed), and the attention-only score trades precision for recall (it flags 1,017 users, 258 of them anomalous). On Cora the reconstruction detector is more balanced (21 true positives, 24 false positives, 49 missed). Metrics at a validation-selected threshold depend on how well validation and test distributions match; ROC-AUC and average precision, which do not depend on a threshold, are therefore the main comparison metrics.

### 5.2 Baseline Comparison

Table: Anomaly detection on injected Cora, test split, mean ± standard deviation over five seeds. AP = average precision (step-wise area under the precision-recall curve); P@k = precision among the k highest-scored nodes with k equal to the number of true anomalies.
Table: 
| method | kind | seeds | ROC-AUC | AP | F1 | Precision | Recall | P@k (k=#pos) |
|---|---|---|---|---|---|---|---|---|
| Isolation Forest (attributes) | baseline | 5 | 0.676 ± 0.027 | 0.122 ± 0.007 | 0.160 ± 0.025 | 0.126 ± 0.019 | 0.229 ± 0.075 | 0.140 ± 0.031 |
| Isolation Forest (attributes + structure) | baseline | 5 | 0.673 ± 0.045 | 0.131 ± 0.031 | 0.158 ± 0.041 | 0.111 ± 0.043 | 0.423 ± 0.251 | 0.169 ± 0.019 |
| GCN reconstruction (DOMINANT-style) | baseline | 5 | 0.765 ± 0.030 | 0.344 ± 0.037 | 0.346 ± 0.057 | 0.486 ± 0.073 | 0.283 ± 0.091 | 0.406 ± 0.024 |
| GAT reconstruction | ours | 5 | 0.771 ± 0.031 | 0.362 ± 0.033 | 0.377 ± 0.040 | 0.476 ± 0.037 | 0.317 ± 0.060 | 0.403 ± 0.033 |
| GATv2 reconstruction | ours | 5 | 0.767 ± 0.026 | 0.352 ± 0.031 | 0.390 ± 0.029 | 0.458 ± 0.044 | 0.346 ± 0.054 | 0.394 ± 0.031 |
| GAT attention-only (weighted feature discrepancy) | ours | 5 | 0.796 ± 0.034 | 0.291 ± 0.023 | 0.291 ± 0.061 | 0.323 ± 0.032 | 0.286 ± 0.102 | 0.331 ± 0.027 |
| GAT combined (reconstruction + attention-only) | ours | 5 | 0.798 ± 0.034 | 0.383 ± 0.027 | 0.385 ± 0.017 | 0.573 ± 0.101 | 0.297 ± 0.040 | 0.426 ± 0.036 |
| GAT joint, reconstruction score | ours | 5 | 0.776 ± 0.033 | 0.318 ± 0.054 | 0.384 ± 0.019 | 0.387 ± 0.060 | 0.397 ± 0.074 | 0.383 ± 0.037 |
| GAT joint, attention-only score | ours | 5 | 0.798 ± 0.033 | 0.293 ± 0.018 | 0.306 ± 0.046 | 0.379 ± 0.098 | 0.274 ± 0.069 | 0.343 ± 0.027 |
| GAT joint, combined score | ours | 5 | 0.803 ± 0.033 | 0.378 ± 0.032 | 0.431 ± 0.032 | 0.496 ± 0.113 | 0.409 ± 0.096 | 0.443 ± 0.029 |
| GCN joint, reconstruction score | ours | 5 | 0.776 ± 0.023 | 0.323 ± 0.050 | 0.364 ± 0.031 | 0.366 ± 0.103 | 0.411 ± 0.117 | 0.369 ± 0.031 |
| GATv2 joint, reconstruction score | ours | 5 | 0.769 ± 0.027 | 0.357 ± 0.028 | 0.386 ± 0.032 | 0.476 ± 0.115 | 0.346 ± 0.085 | 0.397 ± 0.033 |
| Control: uniform-neighbour feature deviation (no attention) | control | 5 | 0.790 ± 0.036 | 0.284 ± 0.023 | 0.295 ± 0.043 | 0.313 ± 0.054 | 0.306 ± 0.096 | 0.317 ± 0.027 |
| Control: untrained-GAT attention-only | control | 5 | 0.791 ± 0.036 | 0.286 ± 0.021 | 0.303 ± 0.056 | 0.339 ± 0.038 | 0.294 ± 0.102 | 0.326 ± 0.027 |

Table: Anomaly detection on Weibo (authentic labels), test split, mean ± standard deviation over three seeds.
Table: 
| method | kind | seeds | ROC-AUC | AP | F1 | Precision | Recall | P@k (k=#pos) |
|---|---|---|---|---|---|---|---|---|
| Isolation Forest (attributes) | baseline | 3 | 0.458 ± 0.024 | 0.117 ± 0.002 | 0.152 ± 0.032 | 0.245 ± 0.126 | 0.389 ± 0.527 | 0.147 ± 0.006 |
| Isolation Forest (attributes + structure) | baseline | 3 | 0.458 ± 0.017 | 0.116 ± 0.010 | 0.149 ± 0.032 | 0.270 ± 0.148 | 0.363 ± 0.488 | 0.154 ± 0.020 |
| GCN reconstruction (DOMINANT-style) | baseline | 3 | 0.791 ± 0.011 | 0.319 ± 0.020 | 0.360 ± 0.010 | 0.382 ± 0.040 | 0.346 ± 0.041 | 0.369 ± 0.017 |
| GAT reconstruction | ours | 3 | 0.771 ± 0.026 | 0.288 ± 0.036 | 0.342 ± 0.020 | 0.337 ± 0.012 | 0.347 ± 0.030 | 0.344 ± 0.017 |
| GATv2 reconstruction | ours | 3 | 0.773 ± 0.012 | 0.304 ± 0.021 | 0.368 ± 0.023 | 0.383 ± 0.051 | 0.360 ± 0.040 | 0.361 ± 0.021 |
| GAT attention-only (weighted feature discrepancy) | ours | 3 | 0.609 ± 0.049 | 0.253 ± 0.081 | 0.353 ± 0.025 | 0.308 ± 0.052 | 0.445 ± 0.132 | 0.289 ± 0.093 |
| GAT combined (reconstruction + attention-only) | ours | 3 | 0.714 ± 0.011 | 0.294 ± 0.008 | 0.379 ± 0.014 | 0.352 ± 0.019 | 0.414 ± 0.049 | 0.367 ± 0.009 |
| GAT joint, reconstruction score | ours | 3 | 0.777 ± 0.051 | 0.296 ± 0.063 | 0.356 ± 0.046 | 0.343 ± 0.055 | 0.370 ± 0.036 | 0.353 ± 0.044 |
| GAT joint, attention-only score | ours | 3 | 0.614 ± 0.057 | 0.253 ± 0.077 | 0.361 ± 0.031 | 0.307 ± 0.056 | 0.465 ± 0.120 | 0.301 ± 0.086 |
| GAT joint, combined score | ours | 3 | 0.725 ± 0.043 | 0.304 ± 0.017 | 0.375 ± 0.002 | 0.358 ± 0.020 | 0.396 ± 0.024 | 0.372 ± 0.006 |
| GCN joint, reconstruction score | ours | 3 | 0.791 ± 0.011 | 0.319 ± 0.020 | 0.361 ± 0.010 | 0.381 ± 0.041 | 0.348 ± 0.044 | 0.369 ± 0.017 |
| GATv2 joint, reconstruction score | ours | 3 | 0.761 ± 0.025 | 0.288 ± 0.027 | 0.335 ± 0.024 | 0.410 ± 0.070 | 0.286 ± 0.028 | 0.336 ± 0.032 |
| Control: uniform-neighbour feature deviation (no attention) | control | 3 | 0.630 ± 0.027 | 0.335 ± 0.038 | 0.423 ± 0.026 | 0.423 ± 0.055 | 0.428 ± 0.028 | 0.422 ± 0.028 |
| Control: untrained-GAT attention-only | control | 3 | 0.614 ± 0.049 | 0.275 ± 0.104 | 0.355 ± 0.078 | 0.394 ± 0.151 | 0.347 ± 0.093 | 0.323 ± 0.086 |

![Test ROC-AUC on injected Cora (mean and standard deviation over five seeds). The horizontal axis starts at chance level.](E:/AttentionGNN/artifacts/results/figures/anomaly_auc_cora_injected.png)

![Test ROC-AUC on Weibo (mean and standard deviation over three seeds). The dotted line marks chance level.](E:/AttentionGNN/artifacts/results/figures/anomaly_auc_weibo.png)

*Interpretation.* Both reconstruction baselines (GCN, GAT) clearly beat Isolation Forest on both datasets (paired gap +0.095 on Cora and +0.313 on Weibo, in every seed), so the graph adds information that attributes alone do not carry. Between the GNN encoders the differences are small: on Cora GAT, GATv2 and GCN are within 0.006 of each other; on Weibo the plain GCN is slightly ahead, and the attention-based encoders do not offer an accuracy advantage that would justify their extra cost (GATv2 needs about 1.5 times the training time on Weibo). The attention-only score is the interesting case for the proposal's hypothesis: it is competitive with reconstruction on Cora but not on Weibo, and in neither case does it beat the control that ignores the learned attention. What this does not establish is that attention is useless in general: the learned attention here is nearly uniform on Cora, so the experiment cannot tell whether sharper attention would help.

Table: Influence ranking on injected Cora at p = 0.10, held-out test nodes. Seed-set spread is the expected number of activated nodes for the 10 (20) nodes each method selects, simulated on a fresh cascade stream.
Table: 
| method | seeds | Spearman | Kendall tau | NDCG@50 | Top-50 overlap | Seed-set spread top10 | Seed-set spread top20 |
|---|---|---|---|---|---|---|---|
| Degree | 5 | 0.752 ± 0.005 | 0.602 ± 0.005 | 0.867 ± 0.018 | 0.500 ± 0.057 | 104.1 ± 8.4 | 145.4 ± 7.4 |
| PageRank | 5 | 0.467 ± 0.011 | 0.324 ± 0.008 | 0.813 ± 0.027 | 0.408 ± 0.036 | 102.7 ± 10.6 | 142.6 ± 7.3 |
| Betweenness | 5 | 0.639 ± 0.015 | 0.477 ± 0.012 | 0.820 ± 0.025 | 0.404 ± 0.038 | 100.7 ± 9.4 | 136.2 ± 7.8 |
| Uniform-attention control (non-learned) | 5 | 0.335 ± 0.013 | 0.231 ± 0.010 | 0.777 ± 0.029 | 0.340 ± 0.045 | 101.8 ± 11.8 | 141.1 ± 8.7 |
| GAT received attention (raw), anomaly-only training | 5 | 0.331 ± 0.013 | 0.228 ± 0.009 | 0.776 ± 0.030 | 0.340 ± 0.045 | 102.2 ± 12.4 | 142.3 ± 7.8 |
| GAT received attention (raw), joint | 5 | 0.335 ± 0.014 | 0.230 ± 0.009 | 0.775 ± 0.033 | 0.344 ± 0.046 | 101.2 ± 12.3 | 140.3 ± 9.4 |
| GAT received attention (degree-adjusted), joint | 5 | -0.507 ± 0.030 | -0.357 ± 0.023 | 0.093 ± 0.007 | 0.000 ± 0.000 | 11.8 ± 0.3 | 23.1 ± 0.4 |
| GAT received attention / degree, joint | 5 | -0.548 ± 0.013 | -0.389 ± 0.011 | 0.089 ± 0.006 | 0.000 ± 0.000 | 11.0 ± 0.1 | 22.0 ± 0.1 |
| GATv2 received attention (raw), joint | 5 | 0.343 ± 0.016 | 0.235 ± 0.012 | 0.774 ± 0.029 | 0.336 ± 0.043 | 103.4 ± 9.1 | 141.3 ± 8.9 |
| GAT influence head, joint | 5 | 0.648 ± 0.043 | 0.474 ± 0.035 | 0.465 ± 0.057 | 0.248 ± 0.091 | 40.7 ± 8.4 | 71.8 ± 4.7 |
| GAT influence head, influence-only | 5 | 0.680 ± 0.021 | 0.501 ± 0.016 | 0.513 ± 0.120 | 0.276 ± 0.057 | 49.5 ± 20.3 | 76.2 ± 15.7 |
| GCN influence head, joint | 5 | 0.732 ± 0.020 | 0.566 ± 0.014 | 0.917 ± 0.033 | 0.648 ± 0.054 | 93.1 ± 6.1 | 118.3 ± 7.6 |
| GATv2 influence head, joint | 5 | 0.667 ± 0.029 | 0.488 ± 0.025 | 0.467 ± 0.030 | 0.244 ± 0.079 | 39.4 ± 8.0 | 69.7 ± 4.1 |
| GAT 1-head influence head, joint | 5 | 0.637 ± 0.016 | 0.463 ± 0.012 | 0.472 ± 0.035 | 0.248 ± 0.067 | 44.0 ± 10.4 | 77.8 ± 6.3 |
| Control: untrained-GAT received attention | 5 | 0.336 ± 0.013 | 0.231 ± 0.010 | 0.776 ± 0.028 | 0.340 ± 0.045 | 101.6 ± 11.8 | 140.8 ± 8.3 |
| Reference: random seed set (mean of 10 draws) | 5 | - | - | - | - | 24.9 ± 1.8 | 51.3 ± 3.4 |
| Reference: oracle top-k by true individual spread | 5 | - | - | - | - | 92.4 ± 8.2 | 126.7 ± 7.7 |

Table: Influence ranking on Weibo at p = 0.003, held-out test nodes.
Table: 
| method | seeds | Spearman | Kendall tau | NDCG@50 | Top-50 overlap | Seed-set spread top10 | Seed-set spread top20 |
|---|---|---|---|---|---|---|---|
| Degree | 3 | 0.629 ± 0.003 | 0.471 ± 0.001 | 0.993 ± 0.001 | 0.920 ± 0.000 | 133.7 ± 6.5 | 196.2 ± 11.0 |
| PageRank | 3 | 0.418 ± 0.008 | 0.293 ± 0.005 | 0.986 ± 0.005 | 0.913 ± 0.023 | 133.0 ± 6.8 | 193.5 ± 10.3 |
| Betweenness | 3 | 0.186 ± 0.017 | 0.123 ± 0.011 | 0.777 ± 0.028 | 0.447 ± 0.012 | 126.3 ± 5.2 | 172.4 ± 15.0 |
| Uniform-attention control (non-learned) | 3 | 0.109 ± 0.011 | 0.056 ± 0.007 | 0.954 ± 0.003 | 0.820 ± 0.020 | 119.1 ± 2.4 | 176.3 ± 4.5 |
| GAT received attention (raw), anomaly-only training | 3 | -0.191 ± 0.057 | -0.133 ± 0.034 | 0.597 ± 0.057 | 0.413 ± 0.114 | 81.4 ± 5.1 | 133.5 ± 10.6 |
| GAT received attention (raw), joint | 3 | -0.198 ± 0.055 | -0.137 ± 0.033 | 0.624 ± 0.024 | 0.393 ± 0.127 | 86.5 ± 2.3 | 140.7 ± 13.3 |
| GAT received attention (degree-adjusted), joint | 3 | -0.425 ± 0.023 | -0.292 ± 0.018 | 0.239 ± 0.100 | 0.087 ± 0.076 | 42.1 ± 27.0 | 68.3 ± 34.2 |
| GAT received attention / degree, joint | 3 | -0.468 ± 0.046 | -0.324 ± 0.031 | 0.157 ± 0.045 | 0.060 ± 0.060 | 20.2 ± 4.0 | 33.0 ± 4.8 |
| GATv2 received attention (raw), joint | 3 | -0.211 ± 0.052 | -0.144 ± 0.035 | 0.507 ± 0.051 | 0.273 ± 0.050 | 80.5 ± 17.9 | 119.7 ± 19.4 |
| GAT influence head, joint | 3 | 0.402 ± 0.053 | 0.278 ± 0.038 | 0.333 ± 0.114 | 0.113 ± 0.070 | 38.3 ± 16.4 | 76.1 ± 19.3 |
| GAT influence head, influence-only | 3 | 0.553 ± 0.103 | 0.398 ± 0.080 | 0.441 ± 0.171 | 0.147 ± 0.050 | 51.7 ± 26.0 | 105.0 ± 27.4 |
| GCN influence head, joint | 3 | 0.889 ± 0.005 | 0.720 ± 0.005 | 0.993 ± 0.003 | 0.960 ± 0.040 | 133.0 ± 7.0 | 195.2 ± 9.1 |
| GATv2 influence head, joint | 3 | 0.391 ± 0.077 | 0.271 ± 0.055 | 0.259 ± 0.078 | 0.067 ± 0.046 | 29.6 ± 11.9 | 68.2 ± 24.1 |
| GAT 1-head influence head, joint | 3 | 0.554 ± 0.082 | 0.392 ± 0.070 | 0.328 ± 0.188 | 0.193 ± 0.200 | 41.7 ± 28.3 | 71.9 ± 46.5 |
| Control: untrained-GAT received attention | 3 | -0.070 ± 0.044 | -0.062 ± 0.030 | 0.670 ± 0.055 | 0.407 ± 0.042 | 104.8 ± 10.2 | 158.0 ± 13.7 |
| Reference: random seed set (mean of 10 draws) | 3 | - | - | - | - | 19.6 ± 0.8 | 39.4 ± 1.1 |
| Reference: oracle top-k by true individual spread | 3 | - | - | - | - | 133.2 ± 6.8 | 199.2 ± 9.7 |

![Rank correlation with held-out simulated spread, Cora (p = 0.10).](E:/AttentionGNN/artifacts/results/figures/influence_spearman_cora_injected.png)

![Rank correlation with held-out simulated spread, Weibo (p = 0.003).](E:/AttentionGNN/artifacts/results/figures/influence_spearman_weibo.png)

![Expected spread of the selected 10-node seed set, Cora (p = 0.10).](E:/AttentionGNN/artifacts/results/figures/seedset_spread_cora_injected.png)

![Expected spread of the selected 10-node seed set, Weibo (p = 0.003).](E:/AttentionGNN/artifacts/results/figures/seedset_spread_weibo.png)

![Held-out test nodes on Weibo (seed 0): simulated spread against three scores. Received attention and its degree-adjusted version do not track spread.](E:/AttentionGNN/artifacts/results/figures/influence_scatter_weibo.png)

![Sensitivity of the Spearman correlation to the cascade probability, Cora.](E:/AttentionGNN/artifacts/results/figures/influence_p_sensitivity_cora_injected.png)

![Sensitivity of the Spearman correlation to the cascade probability, Weibo.](E:/AttentionGNN/artifacts/results/figures/influence_p_sensitivity_weibo.png)

*Interpretation.* The influence results separate three kinds of signal. Degree and PageRank are cheap and strong. A learned head on a degree-aware encoder (the GCN) matches or beats them (on Weibo it gains +0.260 Spearman over degree in every seed and its top-10 seed set reaches the oracle spread), so a learned ranker can add value when it is allowed to use the information that matters. Attention-derived scores are weakest: on Cora they equal a trivial non-learned control, on Weibo they are worse than it, and their seed sets are clearly below degree on Weibo. The practical reading is that received attention is not a substitute for a centrality measure under this influence proxy. As the cascade probability grows, degree loses correlation while the GCN head gains (sensitivity figures), which suggests that at higher spreading probabilities structure beyond degree matters more; we did not test this further.

Table: Seed-paired differences between selected methods (same graph, split and cascade streams per seed).
Table: 
| comparison (paired by seed) | seeds | mean difference | per-seed range | seeds with positive difference |
|---|---|---|---|---|
| Anomaly ROC-AUC: GAT attention-only minus uniform-neighbour control | 5 | +0.006 | [+0.004, +0.011] | 5/5 |
| Anomaly ROC-AUC: trained GAT attention-only minus untrained-GAT attention-only | 5 | +0.005 | [+0.003, +0.011] | 5/5 |
| Anomaly ROC-AUC: GAT reconstruction minus GCN reconstruction | 5 | +0.006 | [-0.002, +0.010] | 4/5 |
| Anomaly ROC-AUC: GATv2 reconstruction minus GAT reconstruction | 5 | -0.005 | [-0.018, -0.000] | 0/5 |
| Anomaly ROC-AUC: GAT joint rec. minus GAT anomaly-only rec. | 5 | +0.005 | [+0.000, +0.011] | 5/5 |
| Anomaly ROC-AUC: GAT combined minus GAT reconstruction (anomaly-only models) | 5 | +0.026 | [+0.017, +0.036] | 5/5 |
| Anomaly ROC-AUC: GAT reconstruction minus Isolation Forest (attributes) | 5 | +0.095 | [+0.053, +0.150] | 5/5 |
| Anomaly ROC-AUC: 4-head minus 1-head GAT (joint, reconstruction) | 5 | +0.006 | [-0.001, +0.015] | 4/5 |
| Influence Spearman: GAT received attention minus uniform-attention control | 5 | +0.000 | [-0.014, +0.022] | 3/5 |
| Influence Spearman: GAT received attention minus degree | 5 | -0.417 | [-0.430, -0.400] | 0/5 |
| Influence Spearman: degree-adjusted minus raw received attention | 5 | -0.842 | [-0.880, -0.794] | 0/5 |
| Influence Spearman: GCN head minus GAT head (joint) | 5 | +0.084 | [+0.032, +0.131] | 5/5 |
| Influence Spearman: GAT head joint minus influence-only | 5 | -0.031 | [-0.085, +0.028] | 1/5 |
| Influence Spearman: GCN head minus degree | 5 | -0.020 | [-0.054, +0.001] | 1/5 |
| Influence NDCG@50: GAT received attention minus degree | 5 | -0.092 | [-0.110, -0.072] | 0/5 |

Table: Seed-paired differences, Weibo.
Table: 
| comparison (paired by seed) | seeds | mean difference | per-seed range | seeds with positive difference |
|---|---|---|---|---|
| Anomaly ROC-AUC: GAT attention-only minus uniform-neighbour control | 3 | -0.022 | [-0.046, -0.004] | 0/3 |
| Anomaly ROC-AUC: trained GAT attention-only minus untrained-GAT attention-only | 3 | -0.005 | [-0.059, +0.060] | 1/3 |
| Anomaly ROC-AUC: GAT reconstruction minus GCN reconstruction | 3 | -0.020 | [-0.038, -0.001] | 0/3 |
| Anomaly ROC-AUC: GATv2 reconstruction minus GAT reconstruction | 3 | +0.002 | [-0.021, +0.020] | 2/3 |
| Anomaly ROC-AUC: GAT joint rec. minus GAT anomaly-only rec. | 3 | +0.007 | [-0.019, +0.043] | 1/3 |
| Anomaly ROC-AUC: GAT combined minus GAT reconstruction (anomaly-only models) | 3 | -0.057 | [-0.070, -0.037] | 0/3 |
| Anomaly ROC-AUC: GAT reconstruction minus Isolation Forest (attributes) | 3 | +0.313 | [+0.312, +0.316] | 3/3 |
| Anomaly ROC-AUC: 4-head minus 1-head GAT (joint, reconstruction) | 3 | -0.022 | [-0.059, +0.004] | 1/3 |
| Influence Spearman: GAT received attention minus uniform-attention control | 3 | -0.307 | [-0.348, -0.250] | 0/3 |
| Influence Spearman: GAT received attention minus degree | 3 | -0.827 | [-0.866, -0.762] | 0/3 |
| Influence Spearman: degree-adjusted minus raw received attention | 3 | -0.227 | [-0.309, -0.174] | 0/3 |
| Influence Spearman: GCN head minus GAT head (joint) | 3 | +0.487 | [+0.451, +0.544] | 3/3 |
| Influence Spearman: GAT head joint minus influence-only | 3 | -0.151 | [-0.325, -0.034] | 0/3 |
| Influence Spearman: GCN head minus degree | 3 | +0.260 | [+0.255, +0.266] | 3/3 |
| Influence NDCG@50: GAT received attention minus degree | 3 | -0.369 | [-0.389, -0.342] | 0/3 |

*Interpretation.* Because the seeds share graph, split and simulation streams, paired differences remove most seed-to-seed variation. The sign counts show which comparisons are consistent: the GCN-over-GAT influence head (all seeds on both datasets), received attention below the uniform control on Weibo (all seeds), and degree adjustment hurting (all seeds). Differences within about 0.01 (for example GATv2 against GAT) are not distinguishable from noise with three to five seeds.

**Published numbers for context only.** The benchmark paper reports unsupervised detector ROC-AUC means on its own splits and implementations (injected Cora, DOMINANT 82.7 ± 5.6 and AnomalyDAE 83.4 ± 2.3; Weibo, DOMINANT 85.0 ± 14.6, GCNAE 90.8, AnomalyDAE 91.5, with simple methods such as Radar above 98) [15]. These values are never mixed into the locally measured tables. With the original labels our Weibo label count now equals theirs (868), but the protocol (our 50 % test split, our training budget of 60 epochs with arc subsampling, no tuning per model) and the implementations differ, so we draw only coarse conclusions: our reconstruction baselines on Cora (0.77) are in the range they report for DOMINANT, and on Weibo (0.77 to 0.79) they are below the published 85 to 91; we did not investigate whether training budget, hyperparameters or the arc subsampling explain this gap. We also did not run Radar or ANOMALOUS, so we cannot reproduce the published Weibo ceiling.

### 5.3 Ablation Study

Table: Ablations on injected Cora (mean ± standard deviation over five seeds). Parameter counts are exact; GATv2 uses two weight matrices per layer and therefore about twice the encoder parameters, while one-head and four-head GAT have identical parameter counts.
Table: 
| ablation | configuration | variant | Anomaly ROC-AUC | Anomaly AP | Influence Spearman (head) | Influence Spearman (recv. attention) | Encoder params | Total params | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1. Encoder family (reconstruction only) | GCN | gcn_anom | 0.765 ± 0.030 | 0.344 ± 0.037 | n/a | n/a | 95936 ± 0 | 195420 ± 0 | 5 |
| 1. Encoder family (reconstruction only) | GAT | gat_anom | 0.771 ± 0.031 | 0.362 ± 0.033 | n/a | 0.331 ± 0.013 | 96192 ± 0 | 195804 ± 0 | 5 |
| 1. Encoder family (reconstruction only) | GATv2 | gatv2_anom | 0.767 ± 0.026 | 0.352 ± 0.031 | n/a | 0.333 ± 0.014 | 192128 ± 0 | 295900 ± 0 | 5 |
| 1b. Encoder family (joint training) | GCN | gcn_joint | 0.776 ± 0.023 | 0.323 ± 0.050 | 0.732 ± 0.020 | n/a | 95936 ± 0 | 195420 ± 0 | 5 |
| 1b. Encoder family (joint training) | GAT | gat_joint | 0.776 ± 0.033 | 0.318 ± 0.054 | 0.648 ± 0.043 | 0.335 ± 0.014 | 96192 ± 0 | 195804 ± 0 | 5 |
| 1b. Encoder family (joint training) | GATv2 | gatv2_joint | 0.769 ± 0.027 | 0.357 ± 0.028 | 0.667 ± 0.029 | 0.343 ± 0.016 | 192128 ± 0 | 295900 ± 0 | 5 |
| 2. Heads (joint training) | 1 head x 64 | gat1h_joint | 0.770 ± 0.030 | 0.364 ± 0.028 | 0.637 ± 0.016 | 0.261 ± 0.049 | 96192 ± 0 | 195804 ± 0 | 5 |
| 2. Heads (joint training) | 4 heads x 16 | gat_joint | 0.776 ± 0.033 | 0.318 ± 0.054 | 0.648 ± 0.043 | 0.335 ± 0.014 | 96192 ± 0 | 195804 ± 0 | 5 |
| 4. Training mode (GAT) | anomaly-only | gat_anom | 0.771 ± 0.031 | 0.362 ± 0.033 | n/a | 0.331 ± 0.013 | 96192 ± 0 | 195804 ± 0 | 5 |
| 4. Training mode (GAT) | influence-only | gat_inf | n/a (head untrained) | n/a (head untrained) | 0.680 ± 0.021 | 0.340 ± 0.014 | 96192 ± 0 | 195804 ± 0 | 5 |
| 4. Training mode (GAT) | joint | gat_joint | 0.776 ± 0.033 | 0.318 ± 0.054 | 0.648 ± 0.043 | 0.335 ± 0.014 | 96192 ± 0 | 195804 ± 0 | 5 |
| 3. Anomaly score type | reconstruction-only score [gat_anom] | gat_anom | 0.771 ± 0.031 | 0.362 ± 0.033 | n/a | n/a | 96192 ± 0 | 195804 ± 0 | 5 |
| 3. Anomaly score type | reconstruction-only score [gat_joint] | gat_joint | 0.776 ± 0.033 | 0.318 ± 0.054 | n/a | n/a | 96192 ± 0 | 195804 ± 0 | 5 |
| 3. Anomaly score type | attention-only score [gat_anom] | gat_anom | 0.796 ± 0.034 | 0.291 ± 0.023 | n/a | n/a | 96192 ± 0 | 195804 ± 0 | 5 |
| 3. Anomaly score type | attention-only score [gat_joint] | gat_joint | 0.798 ± 0.033 | 0.293 ± 0.018 | n/a | n/a | 96192 ± 0 | 195804 ± 0 | 5 |
| 3. Anomaly score type | combined score (0.5 / 0.5) [gat_anom] | gat_anom | 0.798 ± 0.034 | 0.383 ± 0.027 | n/a | n/a | 96192 ± 0 | 195804 ± 0 | 5 |
| 3. Anomaly score type | combined score (0.5 / 0.5) [gat_joint] | gat_joint | 0.803 ± 0.033 | 0.378 ± 0.032 | n/a | n/a | 96192 ± 0 | 195804 ± 0 | 5 |
| 3. Anomaly score type | uniform-neighbour deviation (control) [gat_anom] | gat_anom | 0.790 ± 0.036 | 0.284 ± 0.023 | n/a | n/a | 96192 ± 0 | 195804 ± 0 | 5 |
| 3. Anomaly score type | uniform-neighbour deviation (control) [gat_joint] | gat_joint | 0.790 ± 0.036 | 0.284 ± 0.023 | n/a | n/a | 96192 ± 0 | 195804 ± 0 | 5 |
| 5. Degree adjustment (GAT joint) | received attention, raw | gat_joint | n/a | n/a | n/a | 0.335 ± 0.014 | 96192 ± 0 | 195804 ± 0 | 5 |
| 5. Degree adjustment (GAT joint) | degree-adjusted (OLS residual of log R) | gat_joint | n/a | n/a | n/a | -0.507 ± 0.030 | 96192 ± 0 | 195804 ± 0 | 5 |
| 5. Degree adjustment (GAT joint) | received attention / degree | gat_joint | n/a | n/a | n/a | -0.548 ± 0.013 | 96192 ± 0 | 195804 ± 0 | 5 |

Table: Ablations on Weibo (mean ± standard deviation over three seeds).
Table: 
| ablation | configuration | variant | Anomaly ROC-AUC | Anomaly AP | Influence Spearman (head) | Influence Spearman (recv. attention) | Encoder params | Total params | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1. Encoder family (reconstruction only) | GCN | gcn_anom | 0.791 ± 0.011 | 0.319 ± 0.020 | n/a | n/a | 29824 ± 0 | 62163 ± 0 | 3 |
| 1. Encoder family (reconstruction only) | GAT | gat_anom | 0.771 ± 0.026 | 0.288 ± 0.036 | n/a | -0.191 ± 0.057 | 30080 ± 0 | 62547 ± 0 | 3 |
| 1. Encoder family (reconstruction only) | GATv2 | gatv2_anom | 0.773 ± 0.012 | 0.304 ± 0.021 | n/a | -0.215 ± 0.052 | 59904 ± 0 | 96531 ± 0 | 3 |
| 1b. Encoder family (joint training) | GCN | gcn_joint | 0.791 ± 0.011 | 0.319 ± 0.020 | 0.889 ± 0.005 | n/a | 29824 ± 0 | 62163 ± 0 | 3 |
| 1b. Encoder family (joint training) | GAT | gat_joint | 0.777 ± 0.051 | 0.296 ± 0.063 | 0.402 ± 0.053 | -0.198 ± 0.055 | 30080 ± 0 | 62547 ± 0 | 3 |
| 1b. Encoder family (joint training) | GATv2 | gatv2_joint | 0.761 ± 0.025 | 0.288 ± 0.027 | 0.391 ± 0.077 | -0.211 ± 0.052 | 59904 ± 0 | 96531 ± 0 | 3 |
| 2. Heads (joint training) | 1 head x 64 | gat1h_joint | 0.800 ± 0.030 | 0.323 ± 0.032 | 0.554 ± 0.082 | -0.195 ± 0.076 | 30080 ± 0 | 62547 ± 0 | 3 |
| 2. Heads (joint training) | 4 heads x 16 | gat_joint | 0.777 ± 0.051 | 0.296 ± 0.063 | 0.402 ± 0.053 | -0.198 ± 0.055 | 30080 ± 0 | 62547 ± 0 | 3 |
| 4. Training mode (GAT) | anomaly-only | gat_anom | 0.771 ± 0.026 | 0.288 ± 0.036 | n/a | -0.191 ± 0.057 | 30080 ± 0 | 62547 ± 0 | 3 |
| 4. Training mode (GAT) | influence-only | gat_inf | n/a (head untrained) | n/a (head untrained) | 0.553 ± 0.103 | -0.167 ± 0.022 | 30080 ± 0 | 62547 ± 0 | 3 |
| 4. Training mode (GAT) | joint | gat_joint | 0.777 ± 0.051 | 0.296 ± 0.063 | 0.402 ± 0.053 | -0.198 ± 0.055 | 30080 ± 0 | 62547 ± 0 | 3 |
| 3. Anomaly score type | reconstruction-only score [gat_anom] | gat_anom | 0.771 ± 0.026 | 0.288 ± 0.036 | n/a | n/a | 30080 ± 0 | 62547 ± 0 | 3 |
| 3. Anomaly score type | reconstruction-only score [gat_joint] | gat_joint | 0.777 ± 0.051 | 0.296 ± 0.063 | n/a | n/a | 30080 ± 0 | 62547 ± 0 | 3 |
| 3. Anomaly score type | attention-only score [gat_anom] | gat_anom | 0.609 ± 0.049 | 0.253 ± 0.081 | n/a | n/a | 30080 ± 0 | 62547 ± 0 | 3 |
| 3. Anomaly score type | attention-only score [gat_joint] | gat_joint | 0.614 ± 0.057 | 0.253 ± 0.077 | n/a | n/a | 30080 ± 0 | 62547 ± 0 | 3 |
| 3. Anomaly score type | combined score (0.5 / 0.5) [gat_anom] | gat_anom | 0.714 ± 0.011 | 0.294 ± 0.008 | n/a | n/a | 30080 ± 0 | 62547 ± 0 | 3 |
| 3. Anomaly score type | combined score (0.5 / 0.5) [gat_joint] | gat_joint | 0.725 ± 0.043 | 0.304 ± 0.017 | n/a | n/a | 30080 ± 0 | 62547 ± 0 | 3 |
| 3. Anomaly score type | uniform-neighbour deviation (control) [gat_anom] | gat_anom | 0.630 ± 0.027 | 0.335 ± 0.038 | n/a | n/a | 30080 ± 0 | 62547 ± 0 | 3 |
| 3. Anomaly score type | uniform-neighbour deviation (control) [gat_joint] | gat_joint | 0.630 ± 0.027 | 0.335 ± 0.038 | n/a | n/a | 30080 ± 0 | 62547 ± 0 | 3 |
| 5. Degree adjustment (GAT joint) | received attention, raw | gat_joint | n/a | n/a | n/a | -0.198 ± 0.055 | 30080 ± 0 | 62547 ± 0 | 3 |
| 5. Degree adjustment (GAT joint) | degree-adjusted (OLS residual of log R) | gat_joint | n/a | n/a | n/a | -0.425 ± 0.023 | 30080 ± 0 | 62547 ± 0 | 3 |
| 5. Degree adjustment (GAT joint) | received attention / degree | gat_joint | n/a | n/a | n/a | -0.468 ± 0.046 | 30080 ± 0 | 62547 ± 0 | 3 |

![Ablations: anomaly ROC-AUC on injected Cora.](E:/AttentionGNN/artifacts/results/figures/ablation_auc_cora_injected.png)

![Ablations: anomaly ROC-AUC on Weibo.](E:/AttentionGNN/artifacts/results/figures/ablation_auc_weibo.png)

*Interpretation.* Six design choices were varied. (1) *Encoder family:* no encoder is consistently best for anomalies; for influence the GCN head is far better than the attention heads. (2) *Heads:* one head of width 64 and four heads of width 16 have identical parameter counts and similar anomaly scores (on Weibo the single head is nominally ahead, 0.800 against 0.777, but the paired gap is within the spread), so multi-head attention did not pay off here. (3) *Score type:* reconstruction and attention-only differ by dataset, the combination helps on Cora and hurts on Weibo, and the uniform control beats attention-only on Weibo. (4) *Training mode:* joint and anomaly-only training give statistically indistinguishable anomaly scores (the influence-only model has no trained reconstruction head and therefore no anomaly score), and joint training does not improve the influence head. (5) *Degree adjustment* reduces influence ranking quality on both datasets. (6) *Untrained attention* equals trained attention for anomaly scoring, which is the clearest sign that the learned attention contributes little to that score.

Table: Attention statistics (means over seeds): normalised attention entropy (1 = uniform) in layers 1 and 2, and the rank correlation between received attention and degree. Cora.
Table: 
| variant | mean attention entropy ratio L1 | L2 | Spearman(received attention, degree) | mean alpha | uniform-attention mean alpha |
|---|---|---|---|---|---|
| gat_anom | 0.993 | 1.0 | 0.786 | 0.177 | 0.177 |
| gat_joint | 0.98 | 0.992 | 0.785 | 0.177 | 0.177 |
| gatv2_joint | 0.978 | 0.99 | 0.787 | 0.176 | 0.177 |

Table: Attention statistics, Weibo.
Table: 
| variant | mean attention entropy ratio L1 | L2 | Spearman(received attention, degree) | mean alpha | uniform-attention mean alpha |
|---|---|---|---|---|---|
| gat_anom | 0.564 | 0.608 | 0.276 | 0.011 | 0.011 |
| gat_joint | 0.562 | 0.608 | 0.263 | 0.011 | 0.011 |
| gatv2_joint | 0.608 | 0.434 | 0.301 | 0.011 | 0.011 |

Table: Edge-removal test of attention as an explanation, Cora. Effect = output before minus output after removal; positive means the removal lowered the output.
Table: 
| output | seeds | targets (pooled) | mean effect: remove high-attention | mean effect: remove random | mean effect: remove low-attention | high - random (95% bootstrap CI over targets) | high - low (95% CI) | share of targets high > random | Wilcoxon p (high > random) | per-seed mean(high - random) |
|---|---|---|---|---|---|---|---|---|---|---|
| anomaly | 5 | 100 | +0.0165 | -0.0025 | -0.0082 | +0.0190 [+0.0021, +0.0350] | +0.0247 [+0.0013, +0.0478] | 0.61 | 0.0095 | s0: +0.0196, s1: +0.0216, s2: +0.0139, s3: +0.0165, s4: +0.0234 |
| influence | 5 | 100 | +0.1393 | +0.2774 | +0.4592 | -0.1382 [-0.3090, +0.0303] | -0.3199 [-0.5896, -0.0460] | 0.46 | 0.75 | s0: +0.0654, s1: -0.8570, s2: +0.0793, s3: -0.1581, s4: +0.1797 |

Table: Edge-removal test, Weibo.
Table: 
| output | seeds | targets (pooled) | mean effect: remove high-attention | mean effect: remove random | mean effect: remove low-attention | high - random (95% bootstrap CI over targets) | high - low (95% CI) | share of targets high > random | Wilcoxon p (high > random) | per-seed mean(high - random) |
|---|---|---|---|---|---|---|---|---|---|---|
| anomaly | 3 | 60 | +0.0650 | -0.0016 | -0.0177 | +0.0667 [+0.0167, +0.1434] | +0.0828 [+0.0295, +0.1658] | 0.68 | 0.00309 | s0: +0.0336, s1: +0.1536, s2: +0.0128 |
| influence | 3 | 60 | +0.3317 | +0.0339 | +0.0004 | +0.2977 [+0.1757, +0.4303] | +0.3312 [+0.1974, +0.4767] | 0.63 | 0.00012 | s0: +0.6711, s1: +0.2185, s2: +0.0036 |

![Mean drop in output after removing high-attention, random and low-attention edges, Cora.](E:/AttentionGNN/artifacts/results/figures/explanation_cora_injected.png)

![Mean drop in output after removing high-attention, random and low-attention edges, Weibo.](E:/AttentionGNN/artifacts/results/figures/explanation_weibo.png)

*Interpretation.* The attention statistics explain much of the pattern. Where attention is nearly uniform (Cora) it cannot rank neighbours, so received attention equals the uniform control and the edge-removal test finds no influence effect. Where attention is sharper (Weibo, entropy about 0.6) it picks out edges that matter more to the model (the high-minus-random effect is positive for both outputs, with p-values of 0.003 and 0.0001), yet the sharper attention still does not translate into better rankings or better detection than the controls. In other words, attention can be a meaningful *description of what the trained model uses* without being a better *predictor of influence or anomaly* than simpler quantities. The perturbation test measures only the first.

Table: Sensitivity of anomaly ROC-AUC to the injection rate on Cora (anomaly-only models; mean and standard deviation over seeds; the middle row of each model is the main benchmark with five seeds, the other rows use three seeds). The columns `*_count` give the number of seeds; `n/a` means the model has no attention.
Table: 
| variant | injection_rate | rec_auc_mean | rec_auc_std | rec_auc_count | att_only_auc_mean | att_only_auc_std | att_only_auc_count | combined_auc_mean | combined_auc_std | combined_auc_count |
|---|---|---|---|---|---|---|---|---|---|---|
| gat_anom | 0.022 | 0.76 | 0.058 | 3 | 0.786 | 0.08 | 3 | 0.787 | 0.071 | 3 |
| gat_anom | 0.052 | 0.771 | 0.031 | 5 | 0.796 | 0.034 | 5 | 0.798 | 0.034 | 5 |
| gat_anom | 0.103 | 0.799 | 0.016 | 3 | 0.825 | 0.01 | 3 | 0.827 | 0.014 | 3 |
| gcn_anom | 0.022 | 0.758 | 0.058 | 3 | n/a | n/a | 0 | n/a | n/a | 0 |
| gcn_anom | 0.052 | 0.765 | 0.03 | 5 | n/a | n/a | 0 | n/a | n/a | 0 |
| gcn_anom | 0.103 | 0.79 | 0.01 | 3 | n/a | n/a | 0 | n/a | n/a | 0 |

![Sensitivity to the injection rate on Cora (2.2 %, 5.2 % and 10.3 % of nodes anomalous).](E:/AttentionGNN/artifacts/results/figures/sensitivity_injection_rate.png)

Raising the injection rate from 2.2 % to 10.3 % raises the ROC-AUC of every score, and the ordering of methods does not change within the seed spread: reconstruction with a GAT and with a GCN are indistinguishable, and the attention-only and combined scores stay a few hundredths higher. With more injected structural cliques and swapped attributes, anomalies become easier to separate, so absolute numbers on injected data should not be read as expected performance on organic anomalies, where the benchmark paper reports much lower scores for deep detectors [15]. Sensitivity to the cascade probability is shown in the influence figures of Section 5.2: the ranking quality of degree, PageRank and the learned heads changes with p, and relative to the non-learned uniform-attention control, received attention is identical on Cora and lower on Weibo at every p we tried. Degree has the highest correlation at the smallest p, while the GCN head improves as p grows.

Table: Measured training and inference cost (mean ± standard deviation over seeds; wall-clock; process memory is a cumulative upper bound because several runs shared one process). Cora.
Table: 
| variant | seeds | train seconds | anomaly-score inference seconds | centrality seconds | process peak working set, MB (cumulative upper bound) | total params |
|---|---|---|---|---|---|---|
| centrality | 5 | - | - | 25.6 ± 6.9 | 761.4 ± 117.3 | - |
| gat1h_joint | 5 | 18.6 ± 1.6 | 0.2 ± 0.0 | - | 814.5 ± 6.1 | 195804.0 ± 0.0 |
| gat_anom | 5 | 24.4 ± 4.8 | 0.3 ± 0.1 | - | 810.3 ± 10.7 | 195804.0 ± 0.0 |
| gat_inf | 5 | 23.9 ± 4.6 | 0.3 ± 0.1 | - | 813.4 ± 7.4 | 195804.0 ± 0.0 |
| gat_joint | 5 | 24.7 ± 3.6 | 0.2 ± 0.0 | - | 814.1 ± 6.0 | 195804.0 ± 0.0 |
| gat_untrained | 5 | 0.0 ± 0.0 | 0.2 ± 0.0 | - | 814.5 ± 6.1 | 195804.0 ± 0.0 |
| gatv2_anom | 5 | 29.7 ± 3.1 | 0.3 ± 0.1 | - | 811.4 ± 10.6 | 295900.0 ± 0.0 |
| gatv2_joint | 5 | 32.2 ± 5.8 | 0.3 ± 0.0 | - | 814.5 ± 6.1 | 295900.0 ± 0.0 |
| gcn_anom | 5 | 17.7 ± 2.8 | 0.2 ± 0.0 | - | 803.5 ± 23.8 | 195420.0 ± 0.0 |
| gcn_joint | 5 | 18.0 ± 3.5 | 0.2 ± 0.0 | - | 814.1 ± 6.0 | 195420.0 ± 0.0 |
| iforest_attr | 5 | 0.5 ± 0.2 | - | - | 744.7 ± 154.7 | - |
| iforest_attr_struct | 5 | 0.5 ± 0.2 | - | - | 747.9 ± 147.5 | - |

Table: Measured cost, Weibo.
Table: 
| variant | seeds | train seconds | anomaly-score inference seconds | centrality seconds | process peak working set, MB (cumulative upper bound) | total params |
|---|---|---|---|---|---|---|
| centrality | 3 | - | - | 30.6 ± 1.6 | 1323.7 ± 565.9 | - |
| gat1h_joint | 3 | 51.1 ± 7.8 | 3.8 ± 0.4 | - | 1650.4 ± 0.0 | 62547.0 ± 0.0 |
| gat_anom | 3 | 57.5 ± 2.3 | 4.1 ± 0.3 | - | 1556.5 ± 162.6 | 62547.0 ± 0.0 |
| gat_inf | 3 | 57.0 ± 7.1 | 4.4 ± 0.4 | - | 1650.4 ± 0.0 | 62547.0 ± 0.0 |
| gat_joint | 3 | 59.9 ± 7.8 | 5.2 ± 1.9 | - | 1650.4 ± 0.0 | 62547.0 ± 0.0 |
| gat_untrained | 3 | 0.0 ± 0.0 | 4.1 ± 0.4 | - | 1650.4 ± 0.0 | 62547.0 ± 0.0 |
| gatv2_anom | 3 | 85.6 ± 9.9 | 7.1 ± 2.6 | - | 1650.4 ± 0.0 | 96531.0 ± 0.0 |
| gatv2_joint | 3 | 89.4 ± 10.0 | 5.8 ± 1.0 | - | 1650.4 ± 0.0 | 96531.0 ± 0.0 |
| gcn_anom | 3 | 40.5 ± 2.7 | 2.4 ± 0.1 | - | 1495.4 ± 268.5 | 62163.0 ± 0.0 |
| gcn_joint | 3 | 37.9 ± 2.3 | 2.4 ± 0.4 | - | 1650.4 ± 0.0 | 62163.0 ± 0.0 |
| iforest_attr | 3 | 0.3 ± 0.1 | - | - | 1289.1 ± 625.8 | - |
| iforest_attr_struct | 3 | 0.3 ± 0.0 | - | - | 1299.6 ± 607.6 | - |

Table: Isolated single-run profiles (each run in a fresh process, nothing else running; seed 0; the GAT row also includes the edge-removal test, so its memory is the larger of training and that test). Cora.
Table: 
| variant | seed | train seconds | anomaly-score inference seconds | peak working set, MB (isolated process) | total params |
|---|---|---|---|---|---|
| gat_joint | 0 | 11.9 | 0.14 | 767.8 | 195804 |
| gatv2_joint | 0 | 16.5 | 0.16 | 742.6 | 295900 |
| gcn_joint | 0 | 10.4 | 0.09 | 739.8 | 195420 |

Table: Isolated single-run profiles, Weibo.
Table: 
| variant | seed | train seconds | anomaly-score inference seconds | peak working set, MB (isolated process) | total params |
|---|---|---|---|---|---|
| gat_joint | 0 | 51.8 | 4.02 | 1222.1 | 62547 |
| gatv2_joint | 0 | 81.5 | 5.63 | 1525.1 | 96531 |
| gcn_joint | 0 | 36.0 | 2.22 | 1062.6 | 62163 |

The isolated profiles are the cleaner cost measurement. On Cora the isolated training times are about half of the benchmark averages, because the Cora benchmark ran while other work (builds, rendering, tests) shared the CPU; on Weibo the two agree (for example 52 s isolated against a 58 s benchmark average for the GAT) because that benchmark ran on an otherwise idle machine. Peak memory stays near 0.75 GB on Cora and between 1.1 and 1.5 GB on Weibo; GATv2 costs more time and memory than GAT, and GCN is cheapest. These numbers confirm that the full pipeline fits a laptop without a GPU, but they are single runs, and wall-clock times on other machines will differ.

**What the results establish and what they do not.** They establish, for these two graphs and this budget: (a) neighbour feature discrepancy and reconstruction error are useful anomaly signals, and a non-attentive control matches or beats the attention-only score; (b) degree and a plain GCN head are strong influence rankers under the independent-cascade proxy, and learned GAT attention adds nothing beyond a uniform-attention control on Cora and hurts overall ranking on Weibo; (c) high-attention edges matter more than random ones to the trained model's anomaly output on both datasets and to its influence output where attention is sharp; (d) dataset validation is not optional: a changed upstream label file would have biased every Weibo result. They do not establish how attention behaves on other graphs, with measured cascades, with longer training, with other attention regularisers, or on organic anomalies other than the Weibo labels; they do not show that joint training is harmful or useful in general; and they say nothing about causal explanation of the real network. The class imbalance (about 4 to 10 % anomalies), the label-informed selection of the mixing weight and threshold, the transductive setting and the small number of seeds all limit generalisation.


## 6. Conclusion

We implemented and evaluated a shared graph-attention encoder for influence ranking and anomaly detection under a CPU-only budget, with all comparisons run on the same held-out nodes and all numbers recomputed from saved runs. The measured answers to the four research questions are as follows.

**RQ1 (attention-only anomaly scoring).** The attention-only score is a useful detector on injected Cora, but a uniform-weight control and an untrained-attention control perform almost identically (paired gap of about 0.006 ROC-AUC), and on Weibo the attention-only score is weaker than the control. The value comes from the neighbour feature discrepancy, not from learned attention. Combining it with reconstruction helps on Cora and hurts on Weibo, so the combination rule is dataset dependent.

**RQ2 (influence).** Under the independent-cascade proxy, degree and PageRank are hard to beat. Received attention equals a non-learned uniform-attention control on Cora and is worse than that control on Weibo, where it ranks poorly over the whole population; it loses its signal when adjusted for degree. Its top-10 seed sets are competitive with degree on Cora but clearly weaker on Weibo. A plain GCN influence head was the best learned ranker on both datasets.

**RQ3 (joint training).** We found no reliable benefit or harm from joint training for either task; effects are small compared with variation between seeds.

**RQ4 (attention as explanation).** Removing high-attention edges lowers anomaly scores more than random removals on both datasets (small but statistically visible effects), and lowers the influence prediction more on Weibo, where attention is sharp (entropy about 0.6), but not on Cora, where it is nearly uniform. Attention is therefore a modest association with what the trained model uses, not a causal explanation, and it does not make the attention-derived scores better predictors than the controls.

These results are negative for the strong form of the proposal's hypothesis, which is itself a result. The contribution is the integrated, controlled and reproducible evaluation: the pipeline, the controls, the held-out cascade protocol and the perturbation test can be reused on other graphs. Limits: influence is simulated, Cora anomalies are injected, our own first Weibo run used a corrupted upstream label file and had to be repeated (Section 5.1), evaluation is transductive, training budgets are small (Weibo uses arc subsampling), and only three to five seeds were run. Not run: Reddit and the other BOND datasets (the Reddit command is documented and configured), GNNExplainer comparisons, measured cascades, and inductive evaluation.

## 7. Future References

This section lists future work and research directions; the bibliographic references are in Section 8.

- Test sharper or regularised attention, for example GATv2 with an entropy penalty, and graphs where neighbour importance differs strongly between nodes, since the learned attention here was near-uniform on Cora.
- Replace simulated cascades by measured diffusion data, and test whether the GCN advantage for influence persists.
- Run the full BOND collection, including Reddit (where most detectors are near chance), under the same protocol, and add Radar-type baselines that do very well on Weibo [15].
- Study multi-scale neighbourhoods [14] and inductive evaluation on unseen graphs.
- Compare attention-guided perturbation with dedicated explainers [24, 25] and with stability tests of attention under graph edits [32].
- Report the changed Weibo labels to the PyGOD data maintainers (the upstream `main` file blanks validation/test labels) and keep the dataset checks in the continuous tests.
- Add an ablation of the arc-subsampling used for Weibo training and a longer training budget.

## 8. Reference


### References listed in order of first citation

[1] M. E. J. Newman, "Networks: An Introduction," Oxford University Press, 2010. doi:10.1093/acprof:oso/9780199206650.001.0001
[2] L. Page, S. Brin, R. Motwani and T. Winograd, "The PageRank Citation Ranking: Bringing Order to the Web," Stanford InfoLab, 1999. http://ilpubs.stanford.edu:8090/422/
[3] U. Brandes, "A Faster Algorithm for Betweenness Centrality," Journal of Mathematical Sociology, vol. 25, no. 2, pp. 163-177, 2001. doi:10.1080/0022250X.2001.9990249
[4] F. T. Liu, K. M. Ting and Z. H. Zhou, "Isolation Forest," 2008 Eighth IEEE International Conference on Data Mining (ICDM), pp. 413-422, 2008. doi:10.1109/ICDM.2008.17
[5] K. Ding, J. Li, R. Bhanushali and H. Liu, "Deep Anomaly Detection on Attributed Networks," Proceedings of the 2019 SIAM International Conference on Data Mining (SDM), pp. 594-602, 2019. doi:10.1137/1.9781611975673.67
[6] P. Veličković, G. Cucurull, A. Casanova, et al., "Graph Attention Networks," International Conference on Learning Representations (ICLR), 2018. doi:10.48550/arXiv.1710.10903
[7] S. Brody, U. Alon and E. Yahav, "How Attentive are Graph Attention Networks?," International Conference on Learning Representations (ICLR), 2022. doi:10.48550/arXiv.2105.14491
[8] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," International Conference on Learning Representations (ICLR), 2017. doi:10.48550/arXiv.1609.02907
[9] W. Gu, F. Gao, X. Lou and J. Zhang, "Link Prediction via Graph Attention Network," arXiv:1910.04807, 2019. arXiv:1910.04807
[10] W. Gu, F. Gao, X. Lou and J. Zhang, "Discovering Latent Node Information by Graph Attention Network," Scientific Reports, vol. 11, pp. 6967, 2021. doi:10.1038/s41598-021-85826-x
[11] H. Fan, F. Zhang and Z. Li, "AnomalyDAE: Dual Autoencoder for Anomaly Detection on Attributed Networks," arXiv:2002.03665, 2020. doi:10.48550/arXiv.2002.03665
[12] Y. Pei, T. Huang, W. van Ipenburg and M. Pechenizkiy, "ResGCN: Attention-based Deep Residual Modeling for Anomaly Detection on Attributed Networks," Machine Learning, vol. 111, no. 2, pp. 519-541, 2022. doi:10.1007/s10994-021-06044-0
[13] L. Akoglu, H. Tong and D. Koutra, "Graph Based Anomaly Detection and Description: A Survey," Data Mining and Knowledge Discovery, vol. 29, no. 3, pp. 626-688, 2015. doi:10.1007/s10618-014-0365-y
[14] L. Gutiérrez-Gómez, A. Bovet and J. C. Delvenne, "Multi-Scale Anomaly Detection on Attributed Networks," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 01, pp. 678-685, 2020. doi:10.1609/aaai.v34i01.5409
[15] K. Liu, Y. Dou, Y. Zhao, et al., "BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs," Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track, 2022. doi:10.48550/arXiv.2206.10071
[16] K. Liu, Y. Dou, X. Ding, et al., "PyGOD: A Python Library for Graph Outlier Detection," Journal of Machine Learning Research, vol. 25, no. 141, pp. 1-9, 2024. https://jmlr.org/papers/v25/23-0963.html
[17] J. Tang, F. Hua, Z. Gao, et al., "GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection," Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track, 2023. arXiv:2306.12251
[18] D. Kempe, J. Kleinberg and É. Tardos, "Maximizing the Spread of Influence through a Social Network," Proceedings of the 9th International Conference on Knowledge Discovery and Data Mining (KDD '03), pp. 137-146, 2003. doi:10.1145/956750.956769
[19] S. Manchanda, A. Mittal, A. Dhawan, et al., "GCOMB: Learning Budget-Constrained Combinatorial Algorithms over Billion-Sized Graphs," Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 2020. arXiv:1903.03332
[20] C. Ling, J. Jiang, J. Wang, et al., "Deep Graph Representation Learning and Optimization for Influence Maximization," Proceedings of the 40th International Conference on Machine Learning, vol. 202, pp. 21350-21361, 2023. https://proceedings.mlr.press/v202/ling23b.html
[21] T. Chen, S. Yan, J. Guo and W. Wu, "ToupleGDD: A Fine-Designed Solution of Influence Maximization by Deep Reinforcement Learning," IEEE Transactions on Computational Social Systems, 2023. doi:10.1109/TCSS.2023.3272331
[22] S. Jain and B. C. Wallace, "Attention is not Explanation," Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1, pp. 3543-3556, 2019. doi:10.18653/v1/N19-1357
[23] S. Wiegreffe and Y. Pinter, "Attention is not not Explanation," Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 11-20, 2019. doi:10.18653/v1/D19-1002
[24] Z. Ying, D. Bourgeois, J. You, et al., "GNNExplainer: Generating Explanations for Graph Neural Networks," Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 2019. arXiv:1903.03894
[25] C. Agarwal, O. Queen, H. Lakkaraju and M. Zitnik, "Evaluating Explainability for Graph Neural Networks," Scientific Data, vol. 10, pp. 144, 2023. doi:10.1038/s41597-023-01974-x
[26] P. Sen, G. Namata, M. Bilgic, et al., "Collective Classification in Network Data," AI Magazine, vol. 29, no. 3, pp. 93, 2008. doi:10.1609/aimag.v29i3.2157
[27] Z. Yang, W. W. Cohen and R. Salakhutdinov, "Revisiting Semi-Supervised Learning with Graph Embeddings," Proceedings of the 33rd International Conference on Machine Learning, vol. 48, pp. 40-48, 2016. arXiv:1603.08861
[28] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," 3rd International Conference on Learning Representations (ICLR), 2015. doi:10.48550/arXiv.1412.6980
[29] Y. Rong, W. Huang, T. Xu and J. Huang, "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification," International Conference on Learning Representations (ICLR), 2020. doi:10.48550/arXiv.1907.10903
[30] U. Brandes and C. Pich, "Centrality Estimation in Large Networks," International Journal of Bifurcation and Chaos, vol. 17, no. 7, pp. 2303-2318, 2007. doi:10.1142/S0218127407018403
[31] M. Fey and J. E. Lenssen, "Fast Graph Representation Learning with PyTorch Geometric," ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds, 2019. doi:10.48550/arXiv.1903.02428
[32] L. Hu, T. Huang, L. Yu, et al., "Faithful Interpretation for Graph Neural Networks," arXiv:2410.06950, 2024. arXiv:2410.06950

*AI-assistance disclosure:* EDIT_ME: This project was implemented with the help of an AI coding assistant (Claude Code). Check your institution's rules on AI assistance and edit this statement accordingly.