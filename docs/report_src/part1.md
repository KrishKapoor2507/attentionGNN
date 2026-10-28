## i. Abstract

{{include:abstract.md}}

## ii. Keywords

graph attention network; influence ranking; anomaly detection; attributed networks; independent cascade; centrality; interpretability; limited-compute learning

## 1. Introduction

Social, communication and interaction networks are naturally described as graphs whose nodes carry attributes. Two questions are routinely asked of such graphs. Which nodes are structurally influential, in the sense that activity starting at them reaches many others? And which nodes behave unusually given their attributes and their neighbourhood? In practice these questions are answered by separate pipelines: classical centralities such as degree, PageRank and betweenness [@newman2010networks; @page1999pagerank; @brandes2001faster] for influence, and feature-based or reconstruction-based detectors [@liu2008isolation; @ding2019dominant] for anomalies. Centralities treat every neighbour alike and say nothing about why a node matters, while deep detectors usually give a score without an account of which neighbours drove it.

Graph attention networks [@velickovic2018gat] assign a learned weight to every neighbour of a node. The proposal behind this project asks whether those learned weights can serve as a shared signal: the attention a node receives from its neighbours as an influence indicator, and the discrepancy between a node and its attention-weighted neighbourhood as an anomaly indicator. We treat this as a hypothesis to be tested, not as a result to be confirmed, and we test it on a CPU-only budget.

This report documents an implementation and an evaluation, not a new architecture. The encoder is a standard multi-head GAT, with GATv2 [@brody2022gatv2] and GCN [@kipf2017gcn] as controlled alternatives. Earlier work already uses received attention as a node-ranking score for link-prediction-trained encoders [@gu2019deeplinker; @gu2021ganr] and already places attention inside anomaly detectors [@fan2020anomalydae; @pei2022resgcn]. What we contribute is (i) a single shared encoder evaluated on both tasks under one protocol, (ii) a careful treatment of what attention can and cannot rank (normalisation, direction, degree), (iii) comparisons against classical baselines and non-attentive controls on the same held-out nodes, and (iv) a perturbation test of whether high-attention edges matter more to the model than random ones.

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

**Graph encoders.** GCN [@kipf2017gcn] aggregates neighbours with fixed degree-normalised weights, so its weights are not uniform but they are not learned per edge either. GAT [@velickovic2018gat] learns per-edge coefficients with a softmax over each node's incoming neighbours. GATv2 [@brody2022gatv2] shows that the original scoring function is static, meaning the ranking of neighbours does not depend on the query node, and proposes a dynamic variant. Under static attention, a source's received attention is partly a node-wide quantity, which is relevant to how we interpret it for influence.

**Graph anomaly detection.** Akoglu et al. [@akoglu2015survey] organise the field by data type and anomaly kind (structural, contextual, community). DOMINANT [@ding2019dominant] trains a graph autoencoder and scores nodes by attribute and structure reconstruction error. AnomalyDAE [@fan2020anomalydae] adds attention in the structure encoder, and ResGCN [@pei2022resgcn] uses residual-based attention. What counts as anomalous depends on the neighbourhood scale considered [@gutierrez2020multiscale], which warns against a single fixed scale. The BOND benchmark [@liu2022bond] (with the PyGOD library [@liu2024pygod]) separates injected from organic anomalies and reports that detectors that look strong on injected anomalies can be near chance on organic ones; on Weibo, classical non-deep methods already perform very well. Supervised benchmarks such as GADBench [@tang2023gadbench] are not comparable with our unsupervised protocol. The proposal cites a "Zhang et al. (2022), Attention-based Anomaly Detection on Graphs". We searched arXiv, Crossref and OpenAlex and could not verify a paper with that title and author; we therefore do not cite it, and describe the real near-matches above instead (docs/literature/zhang2022_verification.md).

**Influence.** The independent-cascade model of Kempe et al. [@kempe2003influence] defines the expected spread of a seed set and underlies seed-selection research, including GNN-based selectors [@manchanda2020gcomb; @ling2023deepim; @chen2023touplegdd]. Closest to our ranking idea, DeepLinker and GANR [@gu2019deeplinker; @gu2021ganr] rank nodes by the normalised attention their neighbours direct at them, so received attention as a centrality signal is established; our additions are the degree adjustment, the held-out cascade evaluation and the joint use with anomaly scoring.

**Attention and explanation.** Whether attention explains a prediction is contested: Jain and Wallace [@jain2019attention] found weak agreement with gradient-based importance, and Wiegreffe and Pinter [@wiegreffe2019notnot] argued the answer depends on how explanation is tested. For graphs, GNNExplainer [@ying2019gnnexplainer] identifies prediction-relevant subgraphs and features, and later benchmarks report that explainers are fragile on attributed and heterophilous graphs [@agarwal2023graphxai]. We therefore treat attention as an association and test it by perturbation.

**Positioning.** The ingredients are not new. We make no claim of a first unified model or a novel architecture; the contribution is the integrated, controlled empirical study below, and its conclusions are limited to the data and budget used.

## 3. Proposed Work: Attention-Driven Graph Neural Networks for Influence and Anomaly Detection in Complex Networks

### 3.1 Graph Representation and Data Preparation

Every graph is stored as attributes X and an arc list (source j, target i) that is symmetrised, de-duplicated and free of self-loops; self-loops are added inside the message-passing layers. We use two datasets.

**Weibo** (PyGOD/BOND file [@liu2022bond; @liu2024pygod]) is a user-user graph with 400-dimensional user features in which a label marks users with suspicious posting behaviour. These labels are organic, not injected. The source file is directed (346,579 of 407,963 listed arcs have no reverse arc); we symmetrise it. We use the original 2022 file (commit 13e75cd of the PyGOD data repository, SHA-256 pinned in `scripts/download_data.py`), which contains 868 anomalous users (10.3 %), the number given in the benchmark paper. The current upstream file (January 2024, commit "add mask") adds train/validation/test masks and sets the labels of every validation and test node to normal, so only 347 of the 868 anomalies remain labelled; evaluating on it would treat 521 true anomalies as normal. Our first Weibo benchmark used that file. The dataset validation of Section 5.1 exposed the problem; we discarded those runs (kept in `artifacts/superseded/` for transparency) and repeated every Weibo experiment with the original labels.

**Cora** [@sen2008collective; @yang2016planetoid] is a citation graph with 2,708 nodes, 1,433 binary features and seven topic classes. Topic classes are not anomaly labels. We inject synthetic anomalies with a seeded protocol following the scheme used by DOMINANT [@ding2019dominant]: seven disjoint cliques of ten nodes (structural), and 70 nodes whose attributes are replaced by those of the farthest of 50 random candidates (contextual), 140 nodes in total (5.2 %). These labels are synthetic and are never described as observed fraud. The injection seed also determines the split and the model initialisation.

Nodes are split 30/20/50 percent into train, validation and test sets, stratified by label and seeded; the split hash is stored with each run. Normalisation statistics come from the train nodes only. The setting is **transductive**: the encoder sees all node attributes and all edges, including those of validation and test nodes, but never their anomaly labels or influence targets. Weibo has extremely heavy-tailed attributes (standardised values reach 1,814); in a validation-only exploration, clipping them at five standard deviations or compressing them logarithmically removed label signal (validation ROC-AUC of the attribute error, seed 0: 0.669 and 0.614, against 0.752 with plain standardisation), so we use plain standardisation. This check was first run on the corrupted label file and repeated with the original labels, which gave the same ordering.

**Influence proxy.** No measured cascades are available. We use the independent-cascade model [@kempe2003influence] on the symmetrised graph: each directed arc is tested once with probability p when its source becomes active, and the target is the expected number of activated nodes when a single node is the seed. The probability grids lie below the point where cascades become giant (Cora p in {0.05, 0.10, 0.15}; Weibo p in {0.002, 0.003, 0.005}). Spread targets of train, validation and test nodes come from independent random streams with 300/300/1000 repetitions on Cora and 100/100/500 on Weibo; the test-stream estimates are the held-out ground truth. These are simulation outputs, never observed real-world influence.

### 3.2 Attention Encoder, Joint Objectives, Influence Ranking and Anomaly Scoring

**Encoder.** Layer l maps node states z_i^(l) to z_i^(l+1) with H heads of width 16 that are concatenated to width 64. For GAT [@velickovic2018gat], with N(i) the in-neighbours of i including i itself:

$$e_{ij}^{h}=\mathrm{LeakyReLU}_{0.2}\left(\mathbf{a}_{h}^{\top}\,[\,W^{h}z_{i}\,\Vert\,W^{h}z_{j}\,]\right),\qquad \alpha_{ij}^{h}=\frac{\exp(e_{ij}^{h})}{\sum_{k\in N(i)}\exp(e_{ik}^{h})}$$

$$z_{i}^{(l+1)}=\mathrm{ELU}\left(\Vert_{h=1}^{H}\sum_{j\in N(i)}\alpha_{ij}^{h}\,W^{h}z_{j}^{(l)}\right)$$

GATv2 [@brody2022gatv2] replaces the score by a^T LeakyReLU(W [z_i || z_j]). The GCN control [@kipf2017gcn] uses the degree-normalised propagation Z' = ELU(D^(-1/2) (A + I) D^(-1/2) Z W). Two layers give the latent matrix Z (N x 64). Edge convention: the coefficient alpha_ij belongs to the arc j -> i and is normalised over the arcs arriving at i.

**Heads and objectives.** The attribute decoder is one graph convolution of the same family followed by a linear map, giving X_hat in R^(N x d). The structure decoder scores a pair with p_ij = sigmoid(z_i . z_j), evaluated only on sampled pairs: up to 200,000 real arcs per epoch and the same number of uniformly sampled non-edges (ratio 1:1, seed stored), so no dense N x N matrix is formed. An influence head (MLP 64-32-|p|) predicts the standardised log(1 + spread) at each cascade probability. With normalised reconstruction loss and train-node influence targets, the training objective is

$$\mathcal{L}=w_{rec}\left[a\,\mathcal{L}_{attr}+(1-a)\,\mathcal{L}_{struct}\right]+\lambda\,\mathcal{L}_{inf}$$

$$\mathcal{L}_{attr}=\frac{\Vert X-\hat{X}\Vert_{F}^{2}}{N\,d\,\overline{x^{2}}},\qquad \mathcal{L}_{inf}=\frac{1}{|V_{tr}|}\sum_{i\in V_{tr}}\Vert\hat{\sigma}_{i}-\sigma_{i}\Vert^{2}$$

$$\mathcal{L}_{struct}=\mathrm{BCE}\left(\sigma(z_{i}^{\top}z_{j}),\,\mathbf{1}[(j,i)\in E]\right)$$

with a = 0.5, lambda = 1; w_rec = 0 gives influence-only training and lambda = 0 gives anomaly-only training. This joint model, with a shared encoder and task-specific outputs, is our extension beyond the proposal's stronger hypothesis that attention alone suffices; the attention-only scores below test that stronger hypothesis separately. Unit tests verify that each of the three losses sends gradient into the shared encoder. Adam [@kingma2015adam] is used with learning rate 0.005 and weight decay 5e-4. On Weibo the message-passing arcs are randomly subsampled to 30 % in each training epoch (DropEdge-style [@rong2020dropedge]) to fit the CPU budget; inference always uses the full graph.

**Anomaly scores.** All scores are per node and larger means more anomalous. Let z(.) denote robust standardisation with the median and MAD of the train nodes.

$$s_{attr}(i)=\Vert x_{i}-\hat{x}_{i}\Vert_{2},\qquad s_{struct}(i)=\frac{1}{|N(i)|}\sum_{j\in N(i)}(1-p_{ij})^{2}+\frac{1}{K}\sum_{k\in K_{i}}p_{ik}^{2}$$

$$s_{rec}(i)=a_{sel}\,z(s_{attr})+(1-a_{sel})\,z(s_{struct})$$

where K_i are 20 sampled non-neighbours and a_sel is chosen from {0, 0.25, 0.5, 0.75, 1} by validation ROC-AUC. The **attention-only score** is declared in advance as the attention-weighted feature discrepancy to the neighbours,

$$s_{att}(i)=\sum_{j\in N(i)\setminus\{i\}}\tilde{\alpha}_{ij}\,\frac{\Vert x_{i}-x_{j}\Vert_{2}}{\sqrt{d}},\qquad \tilde{\alpha}_{ij}=\frac{\bar{\alpha}_{ij}}{\sum_{k\neq i}\bar{\alpha}_{ik}}$$

where the bar denotes the average over heads and layers; it uses no decoder output. Replacing alpha-tilde by 1/deg(i) gives the non-attentive control **uniform-neighbour deviation**, and an encoder with random weights gives an untrained-attention control. The **combined score** is 0.5 z(s_rec) + 0.5 z(s_att) with fixed weights. The decision threshold is the F1-optimal value on the validation nodes and is frozen before test evaluation. Because validation labels choose a_sel and the threshold, model selection is label-informed even though encoder training is unsupervised.

**Influence ranking.** Because the softmax normalises over the arcs arriving at a target, the coefficients arriving at any node sum to exactly one, so summing them at a target cannot rank anything (a unit test confirms this). We instead define the **received attention** of a source j as the attention its neighbouring targets assign to it,

$$R_{j}=\sum_{i:\,(j\to i)\in E,\;i\neq j}\bar{\alpha}_{ij}$$

with the head and layer average shown above. Isolated nodes receive zero. Since R depends on degree, we compare the raw score, R_j / deg_j, and the **degree-adjusted score**, the residual of log R_j after an ordinary-least-squares fit on log(1 + deg_j) over train nodes. The non-learned control R_uniform_j = sum over targets of 1/(deg_i + 1) shows what uniform attention would give. The influence head output is the second model-based ranking. Baselines are degree, PageRank with damping 0.85 [@page1999pagerank], and betweenness (exact on Cora, a 200-pivot approximation [@brandes2007centrality] on Weibo).

### 3.3 System Architecture

{{fig:architecture.png|System architecture of the implemented pipeline. Orange marks the training objective; green marks stored evidence and user-facing outputs.}}

The pipeline is a Python package (`attngnn`) that runs entirely on the local CPU. Each run, defined by dataset, variant and seed, writes a complete record: configuration, exact command, environment versions, data and split checksums, source-tree hash, log, checkpoint with hash, per-node predictions and machine-readable metrics. A separate builder recomputes every table and figure from those records and writes an index linking each result to its runs. A Streamlit dashboard reads the same records, and can launch new runs as background jobs. The implementation uses PyTorch Geometric [@fey2019pyg].

### 3.4 Functional Block Diagram

{{fig:block_diagram.png|Functional block diagram. Labels are used only in the orange step (validation) and in evaluation; the encoder is trained without anomaly labels.}}

Part A of the diagram is run once per seed; part B uses the frozen model on held-out test nodes, and every compared method is scored on exactly the same nodes.
