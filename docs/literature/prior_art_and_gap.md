# Prior art and defensible gap

Scope note: found with web search and publisher/arXiv pages on 2026-10-25. Google Scholar, DBLP and Semantic Scholar were unreachable, so this is a good-faith search, not an exhaustive one. "Read" means what was opened: full text only where stated, otherwise the abstract page.

## What is already established

**Attention as an influence signal.** The closest prior work is DeepLinker / GANR (Gu, Gao, Lou, Zhang; arXiv 2019, Scientific Reports 2021). It trains a GAT-style encoder for link prediction, then ranks nodes by the total normalised attention their neighbours direct at them ("Attention Rank", summed over heads of the second layer). It compares against PageRank, closeness and betweenness (VC and APS citation networks) and reports spreading-ability checks with an SIR simulation. Full text of the arXiv version and the PMC page were read. Two limits are visible: the supervision is link prediction, and the sections read describe no degree correction. So "received attention is a usable centrality signal" is not new; our degree-adjusted version, and testing it against an independent-cascade Monte-Carlo proxy, are additions.

**Attention inside anomaly detectors.** AnomalyDAE (Fan, Zhang, Li, ICASSP 2020) puts attention in the structure encoder and scores by attribute and structure reconstruction error; ResGCN (Pei et al., Machine Learning 2022) uses a residual-based attention that down-weights anomalous nodes. Both were read at abstract level (ResGCN also its experimental section). Neither, as far as the abstracts show, scores nodes by the attention they receive or by an attention-weighted feature discrepancy. Not finding it is not proof of absence.

**GNNs for influence maximisation.** GCOMB (NeurIPS 2020), DeepIM (ICML 2023) and ToupleGDD (IEEE TCSS 2023) learn seed selection with GNN encoders and reinforcement or generative objectives, evaluated against IMM/OPIM-type algorithms. Abstract level only. They do not target anomaly detection. Learned influence-ranking regressors (InfGCN, RCNN) exist per search results but could not be opened, so they are UNVERIFIED and not cited.

**Influence used for anomaly detection, not unified.** Wei et al. (IEEE CAI 2025) feed average controllability, a node-influence measure, into GNN anomaly detectors as an edge weight or feature. This treats influence as an input; it does not produce an influence ranking and an anomaly score from one encoder.

**Anomaly detection benchmarks.** BOND (NeurIPS 2022 D&B) shows most unsupervised deep detectors do well on injected anomalies but poorly on organic ones, with Reddit near chance and Weibo solved by classical Radar/ANOMALOUS. Gu and Zou (LoG 2023, PMLR 231) report that message passing can hurt unsupervised node-level detection (abstract only). GADBench (NeurIPS 2023 D&B) is supervised and finds tree ensembles with simple neighbourhood aggregation can beat specialised GNNs; it is not comparable to our unsupervised protocol.

**Attention is a questionable explanation.** In NLP, Jain and Wallace (NAACL 2019) found attention weights often disagree with gradient importance, and Wiegreffe and Pinter (EMNLP 2019) replied that the conclusion depends on how "explanation" is defined and tested; both were read at abstract level. For graphs, the GNNExplainer authors (NeurIPS 2019) note that GAT edge attention is identical across predictions for all nodes, so it cannot give per-instance explanations, and their attention baseline underperformed. Brody et al. prove GAT's attention is "static" (neighbour ranking independent of the query) and that GATv2 is dynamic. Recent graph-specific work reports attention-based interpretations are unstable under added edges or nodes (FGAI, arXiv 2024, abstract only) and that explainers degrade on attributed and heterophilic graphs (GraphXAI, Scientific Data 2023, full text read).

## Defensible gap statement

A defensible claim is narrower than "a new model". Individually, the ingredients exist: attention as centrality, attention in anomaly detectors, GNN seed selection, and edge-perturbation checks of explanations. Our contribution, as designed, is an **integration and an empirical evaluation**: one shared multi-head GAT encoder used for (i) anomaly scoring under three scoring rules and (ii) degree-adjusted received-attention influence ranking, checked against degree, PageRank, betweenness and an independent-cascade spread proxy, with attention-guided edge-removal tests of whether the attention signal is faithful. The new knowledge is what the measurements show: when received attention adds information beyond degree, whether it ranks anomalies, and whether perturbing high-attention edges changes scores more than random edges.

Allowed wording: "we study", "we evaluate", "to our knowledge, within the search described, we found no prior work that combines these in one encoder". Final wording must be conditional on the search being repeated before submission.

## Explicit warnings

1. Do not claim the first unified influence-and-anomaly model, a novel architecture or a novel attention mechanism. The architecture is GAT/GATv2 (plus DOMINANT-style decoders), and Attention Rank predates our influence score.
2. Do not call attention an explanation; call it an importance signal that we test. The test (edge removal) must be described as a sanity check of faithfulness, not proof.
3. Do not cite the proposal's "Zhang et al. (2022)" paper; it was not found (see zhang2022_verification.md).
4. Do not compare numbers with published BOND tables unless data version, labels (Weibo 868 vs the local 347) and protocol match.
5. Under GAT, received attention is partly a node-global score plus degree effect (static attention), so report it against degree and a degree-matched null.
