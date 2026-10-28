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
- Run the full BOND collection, including Reddit (where most detectors are near chance), under the same protocol, and add Radar-type baselines that do very well on Weibo [@liu2022bond].
- Study multi-scale neighbourhoods [@gutierrez2020multiscale] and inductive evaluation on unseen graphs.
- Compare attention-guided perturbation with dedicated explainers [@ying2019gnnexplainer; @agarwal2023graphxai] and with stability tests of attention under graph edits [@hu2024faithfulgat].
- Report the changed Weibo labels to the PyGOD data maintainers (the upstream `main` file blanks validation/test labels) and keep the dataset checks in the continuous tests.
- Add an ablation of the arc-subsampling used for Weibo training and a longer training budget.

## 8. Reference
