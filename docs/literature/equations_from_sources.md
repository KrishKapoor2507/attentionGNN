# Equations defined by source (what the cited papers actually say)

Notation is harmonised here; every equation is tagged with the paper and the place it appears. Where an item is OUR definition (not in any source) it is labelled **[ours]** and must not be attributed to a cited paper.

## 1. GCN propagation (Kipf & Welling 2017, `kipf2017gcn`)

Section 2, Eq. 8 (renormalisation trick): with `A~ = A + I_N`, `D~_ii = sum_j A~_ij`,

`Z = D~^{-1/2} A~ D~^{-1/2} X Theta`, `Theta in R^{C x F}`.

The layer-wise rule used in the two-layer model is `H^{(l+1)} = sigma( D~^{-1/2} A~ D~^{-1/2} H^{(l)} W^{(l)} )`. Neighbour weights are fixed by node degrees and are not learned. Section 7.2 states that the model assumes locality and equal importance of self-connections and edges to neighbours, and handles undirected graphs only.

## 2. GAT attention (Velickovic et al. 2018, `velickovic2018gat`), Section 2.1, Eq. 1-6

- Shared linear map `W in R^{F' x F}`; score `e_ij = a(W h_i, W h_j)` (Eq. 1).
- Masked attention: `e_ij` is computed only for `j in N_i`, where `N_i` is "the first-order neighbors of i (including i)" in all of the paper's experiments. So the self-loop is part of the softmax neighbourhood.
- Normalisation over the incoming neighbourhood of the *target* i (Eq. 2): `alpha_ij = softmax_j(e_ij) = exp(e_ij) / sum_{k in N_i} exp(e_ik)`.
- Single-layer scoring network with weight vector `a in R^{2F'}` and LeakyReLU, negative input slope **0.2** (Eq. 3):
  `alpha_ij = exp( LeakyReLU( a^T [W h_i || W h_j] ) ) / sum_{k in N_i} exp( LeakyReLU( a^T [W h_i || W h_k] ) )`.
- Output (Eq. 4): `h_i' = sigma( sum_{j in N_i} alpha_ij W h_j )`.
- Multi-head, hidden layers (Eq. 5, concatenation): `h_i' = ||_{k=1..K} sigma( sum_{j in N_i} alpha^k_ij W^k h_j )`.
- Multi-head, final layer (Eq. 6, averaging): `h_i' = sigma( (1/K) sum_k sum_{j in N_i} alpha^k_ij W^k h_j )`.
- Transductive settings used K=8 heads x 8 features in layer 1, ELU, one head in layer 2 (Sec. 3.3); dropout 0.6 on layer inputs and on normalised attention coefficients; L2 0.0005.
- Directed graphs: "leave out computing alpha_ij if edge j -> i is not present" (Sec. 2.2).

**Consequence that matters to us.** Because the softmax runs over `j in N_i` for a fixed target i, `sum_j alpha_ij = 1` for every i and every head. Summing coefficients over a node's *own* neighbourhood is therefore the constant 1 and carries no information. Any attention-based importance score must instead aggregate over the *other* index (the nodes that attend to a given source). This is a mathematical consequence of Eq. 2, not a sentence in the paper.

**Received attention [ours, with a published precedent].** For a source node `u`, define the attention it receives, per head and layer, as the sum of the coefficients that its out-neighbours (the targets `i` with `u in N_i`) assign to it:

`R^{(l,k)}(u) = sum_{i : u in N_i} alpha^{(l,k)}_{i u}`, aggregated across heads/layers (mean or sum).

Degree adjustment (our design choice, not from a source): divide by the in-degree-with-self-loop count, or compare against the null value `sum_{i} 1/|N_i|` that uniform attention would give, because `R` grows with the number of targets.
Published precedent for the un-adjusted version: DeepLinker/GANR "Attention Rank" (Gu et al. 2019/2021), Eq. 7 of the arXiv version: sum over heads `k` of the second layer and over `j in N_i` of the normalised coefficient `alpha^k_{j,i}`. In that paper's index convention (`alpha_{i,j}` = weight of j's features to i, Eqs. 1-3), `alpha_{j,i}` is the attention node j pays to node i, so the sum is the attention i receives. Their text and the Sci. Rep. page describe it as the attention that neighbours direct toward the node. The paper does not state a degree correction in the sections read.

## 3. GATv2 (Brody, Alon, Yahav 2022, `brody2022gatv2`), Section 2.2 and 3.3, Eq. 2-7

- GAT scoring (their Eq. 2/6): `e(h_i, h_j) = LeakyReLU( a^T [W h_i || W h_j] )`.
- GATv2 scoring (Eq. 7): `e(h_i, h_j) = a^T LeakyReLU( W [h_i || h_j] )`.
- Softmax and aggregation are unchanged: `alpha_ij = softmax_j(e(h_i,h_j))` over `j in N_i`, `h_i' = sigma( sum_j alpha_ij W h_j )`.
- Static attention (Definition 3.1): for a family of scoring functions there is a key that is always selected regardless of the query. Theorem 1: a GAT layer computes only static attention, i.e. the ranking of neighbours induced by `alpha` is the same for every query node `i`; only the sharpness of the distribution depends on `h_i`. Theorem 2: GATv2 computes dynamic attention.
- Implication for the received-attention score: with GAT, a neighbour's rank inside any neighbourhood is governed by a node-only score `s_j = a_2^T W h_j`, so received attention under GAT is partly a *global node score* plus a degree/softmax-normalisation effect. GATv2 allows query-dependent rankings. Brody et al. report GATv2 is more robust to structural (random edge) noise on ogbn-arxiv and ogbn-mag.

## 4. Independent Cascade (Kempe, Kleinberg, Tardos; `kempe2003influence`, `kempe2015influence`), Section 2.2 of the journal version

Initial active set `A_0`. Time proceeds in discrete steps. When node `v` first becomes active at step `t`, it gets a **single chance** to activate each currently inactive out-neighbour `w`, succeeding with probability `p_{v,w}` independently of the history (multiple newly active in-neighbours of `w` attempt in arbitrary order). If `v` succeeds, `w` is active at step `t+1`. Whether or not `v` succeeds, it never tries `w` again. The process stops when no more activations are possible. Influence of a seed set: `sigma(A) = E[ |A_final| ]` (expected final active-set size).

- Influence maximisation under IC is NP-hard and hard to approximate better than `1 - 1/e` (journal Theorem 2.2); `sigma` is monotone and submodular under IC, so greedy hill-climbing gives a `(1 - 1/e - epsilon)` approximation when `sigma` is estimated by simulation (Theorem 1.1 / Section 4).
- Exact evaluation of `sigma` is #P-complete (proved after the KDD paper; the journal version cites it). Monte-Carlo simulation gives arbitrarily good estimates. Kempe et al.'s experiments used **10,000 simulations per seed set**.
- Edge probabilities `p_{v,w}` are model parameters. The paper does not prescribe a value for real data; a uniform p, 1/in-degree (weighted cascade) or similar is our modelling choice and must be reported as such.
- Note for the project's "held-out" spread proxy: the IC probabilities and seeds used for the ground-truth proxy must be generated independently of the attention weights, otherwise the proxy is circular. This is a design requirement **[ours]**.

## 5. DOMINANT loss (Ding et al. 2019, `ding2019dominant`)

**Status: the paper PDF could not be opened** (SIAM 403; ASU-hosted PDF behind a login), so the symbols below come from the authors' official PyTorch implementation (github.com/kaize0409/GCN_AnomalyDetection_pytorch, `model.py`, `run.py`), not from the paper's equations. Check the paper's own notation before quoting symbols.

Architecture (`model.py`): shared encoder of two GCN layers (`n_feat -> n_hid -> n_hid`, ReLU, dropout); attribute decoder of two GCN layers (`n_hid -> n_hid -> n_feat`); structure decoder of one GCN layer followed by an inner-product `Z Z^T` reconstructing the adjacency.

Per-node errors (`run.py`, `loss_func`):
- attribute error `e^X_i = || x_i - x_hat_i ||_2`
- structure error `e^A_i = || a_i - a_hat_i ||_2` (row i of the target adjacency vs. the reconstruction)
- per-node cost and anomaly score: `score_i = alpha * e^X_i + (1 - alpha) * e^A_i`, default `alpha = 0.8` (CLI default; README does not justify it); training minimises the mean of this cost. Defaults in the repo: 100 epochs, Adam lr 5e-3, hidden 64, dropout 0.3.
- The reconstruction target in the training loop is the adjacency with self-loops given to the model (`adj_label`); the model input is likewise the adjacency passed to the GCN layers (normalisation details sit in `utils.py`: symmetric normalisation `D^{-1/2} A D^{-1/2}` after adding self-loops).
- Evaluation in that repo: ROC-AUC over per-node scores, checked every 10 epochs and at the last epoch (so the repo's reporting is not a held-out-validated protocol; BOND instead reports mean/std/max over a hyper-parameter grid).
- Note the weighting direction: in the code, `alpha` multiplies the **attribute** error. I could not check which term the paper's own equation puts `alpha` on, so do not assume; use the code convention or state yours explicitly.

## 6. GNNExplainer edge-mask objective (Ying et al. 2019), for the perturbation experiments

Maximise mutual information between the model's prediction and the explanatory subgraph; implemented as learning a soft mask `M` (sigmoid-mapped to [0,1]) over the computation-graph adjacency, `A_c (.) sigma(M)`, optimised with cross-entropy to the prediction plus size and element-wise entropy regularisers (Sec. 4). Its ATT baseline uses the average GAT attention across layers as edge importance and, per the authors, cannot give instance-specific explanations because attention weights are the same for predictions across all nodes. Our attention-guided edge-removal test is a *different, deletion-based* check **[ours]**; it is not a re-implementation of GNNExplainer.
