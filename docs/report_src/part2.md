## 4. Experimental Analysis and observations

**What was run.** The benchmark contains {{fact:n_runs}} saved runs: twelve methods (three baselines, nine GNN variants) on injected Cora with five seeds and on Weibo with three seeds. The seed fixes the split, the model initialisation, the negative-sampling stream and, for Cora, the injected anomalies and the cascade simulations. All runs used one Intel i7-1165G7 CPU (four cores) with 15.8 GB of memory and no GPU, Python 3.11, PyTorch 2.14.1 (CPU) and PyTorch Geometric 2.8.1. Means and standard deviations in all tables are recomputed from the per-seed values by `scripts/build_results.py` (sample standard deviation); `[n=k]` in the raw CSV files gives the number of seeds, and tables here show the mean ± standard deviation. Three seeds (Weibo) or five seeds (Cora) support only coarse statements about uncertainty, so comparisons below also use differences paired by seed (same graph, split and cascade streams).

Table: Datasets and splits used by the benchmark (anomaly counts per split are for seed 0).
{{table:datasets.csv}}

**Anomaly detection (RQ1).** On injected Cora the reconstruction score of the GAT reaches a test ROC-AUC of {{fact:cora_injected.anom.gat_anom.rec.roc_auc.pm}}, against {{fact:cora_injected.anom.gcn_anom.rec.roc_auc.pm}} for the DOMINANT-style GCN and {{fact:cora_injected.anom.iforest_attr.iforest.roc_auc.pm}} for Isolation Forest. The attention-only score reaches {{fact:cora_injected.anom.gat_anom.att_only.roc_auc.pm}}. The two non-attentive controls are almost as good: uniform-neighbour feature deviation {{fact:cora_injected.anom.gat_anom.feat_dev_uniform.roc_auc.pm}} and the attention of an untrained GAT {{fact:cora_injected.anom.gat_untrained.att_only.roc_auc.pm}}. Paired by seed, the trained attention-only score exceeds the uniform control by only +0.006 ROC-AUC (positive in five of five seeds; see the paired-difference tables in Section 5.2), which is consistent in sign but much smaller than the seed-to-seed spread of about 0.03. We read this as: the useful signal is the feature discrepancy between a node and its neighbours; learned attention contributes at most a negligible amount. The combined score is the best on injected Cora ({{fact:cora_injected.anom.gat_joint.combined.roc_auc.pm}} for the joint model), because reconstruction and neighbour discrepancy capture different anomaly types.

On Weibo the picture differs. The best scores are the GCN reconstruction ({{fact:weibo.anom.gcn_anom.rec.roc_auc.pm}}) and the GATv2 reconstruction ({{fact:weibo.anom.gatv2_anom.rec.roc_auc.pm}}), with the GAT at {{fact:weibo.anom.gat_anom.rec.roc_auc.pm}}; paired by seed the GAT is 0.020 below the GCN (below it in all three seeds). The attention-only score is weak ({{fact:weibo.anom.gat_anom.att_only.roc_auc.pm}}) and lies below the uniform control ({{fact:weibo.anom.gat_anom.feat_dev_uniform.roc_auc.pm}}) in every seed. Averaging attention-only and reconstruction scores lowers the ROC-AUC ({{fact:weibo.anom.gat_anom.combined.roc_auc.pm}}, a paired loss of 0.057). Isolation Forest scores {{fact:weibo.anom.iforest_attr.iforest.roc_auc.pm}}, below chance: the benchmark's anomalous users are not isolated in attribute space, so a detector that looks for rare attribute values is misled. Average precision reaches {{fact:weibo.anom.gcn_anom.rec.average_precision.pm}} for the best model against a base rate of 10.3 %, about three times the chance level, so even the best Weibo detector flags many normal users. The structure-error score alone has a mean ROC-AUC of {{fact:weibo.anom.gat_anom.struct_err.roc_auc.pm}}, below 0.5 and strongly seed dependent, meaning anomalous Weibo users tend to be reconstructed structurally *better* than normal ones; this agrees with the benchmark paper's observation that Weibo outliers are unusually clustered [@liu2022bond] (our own label check, Section 5.1, finds the same direction for clustering). The validation step therefore selected the attribute-only mixing weight (a_sel = 1) in every Weibo run that trained a reconstruction head.

**Influence (RQ2).** All rankings are evaluated on the same held-out test nodes against independent-cascade spread simulated with a separate random stream (Section 3.1). On Cora (p = 0.10), degree reaches a Spearman correlation of {{fact:cora_injected.inf.0.1.centrality.degree.spearman.pm}}, betweenness {{fact:cora_injected.inf.0.1.centrality.betweenness.spearman.pm}} and PageRank {{fact:cora_injected.inf.0.1.centrality.pagerank.spearman.pm}}. The raw received attention of the GAT reaches {{fact:cora_injected.inf.0.1.gat_joint.recv_att.spearman.pm}}, identical within noise to the non-learned uniform-attention control ({{fact:cora_injected.inf.0.1.centrality.uniform_recv_att.spearman.pm}}; paired difference 0.000). The reason is visible in the attention statistics (attention-statistics tables in Section 5.3): on Cora the learned attention is almost perfectly uniform, with a normalised entropy of about 0.98 to 1.0. On Weibo (p = 0.003) attention is sharper (entropy about 0.6), yet received attention correlates *negatively* with simulated spread over the whole test set ({{fact:weibo.inf.0.003.gat_joint.recv_att.spearman.pm}}), while degree reaches {{fact:weibo.inf.0.003.centrality.degree.spearman.pm}}. The negative correlation follows from the definition: because every target splits its attention, a source's received attention is roughly the sum over its neighbours of 1/(their degree), which is large for nodes sitting next to low-degree nodes and not for hubs. At the top of the ranking the picture is better but still behind degree: the NDCG@50 of received attention is {{fact:weibo.inf.0.003.gat_joint.recv_att.ndcg@50.pm}} on Weibo and {{fact:cora_injected.inf.0.1.gat_joint.recv_att.ndcg@50.pm}} on Cora, against {{fact:weibo.inf.0.003.centrality.degree.ndcg@50.pm}} and {{fact:cora_injected.inf.0.1.centrality.degree.ndcg@50.pm}} for degree. On Weibo the non-learned uniform-attention control ranks better than the learned attention (Spearman {{fact:weibo.inf.0.003.centrality.uniform_recv_att.spearman.pm}}, NDCG@50 {{fact:weibo.inf.0.003.centrality.uniform_recv_att.ndcg@50.pm}}) at every cascade probability we tried, so training the attention made this signal worse, not better; on Cora the two are identical at every probability. Degree adjustment, which removes the part of received attention explained by degree, makes the ranking much worse on both datasets (Spearman {{fact:cora_injected.inf.0.1.gat_joint.recv_att_adj.spearman.pm}} on Cora, {{fact:weibo.inf.0.003.gat_joint.recv_att_adj.spearman.pm}} on Weibo): in these graphs influence under the independent-cascade proxy is largely a degree effect, and what remains after removing degree is anti-correlated with it.

The learned influence head behaves very differently depending on the encoder. A plain GCN head is the best learned ranker ({{fact:weibo.inf.0.003.gcn_joint.model_pred.spearman.pm}} on Weibo, {{fact:cora_injected.inf.0.1.gcn_joint.model_pred.spearman.pm}} on Cora) and beats degree on Weibo in every seed (paired +0.260), whereas the GAT head reaches only {{fact:weibo.inf.0.003.gat_joint.model_pred.spearman.pm}} and {{fact:cora_injected.inf.0.1.gat_joint.model_pred.spearman.pm}}. A plausible reason is that GCN's degree-normalised aggregation carries degree information directly while the attention softmax removes it; we did not test this explanation separately. The influence head also overfits: its training loss falls to a few hundredths on Cora while held-out correlation stays moderate.

Individual spread and seed-set spread are different quantities. Choosing the ten nodes with the highest individual spread does not give the best ten-node seed set, because those nodes overlap: on Cora (p = 0.10) the oracle top-10 by individual spread reaches an expected set spread of {{fact:cora_injected.seedset.oracle10}}, below the {{fact:cora_injected.seedset.degree10}} reached by the degree-chosen set (influence tables in Section 5.2). All reported seed-set spreads come from a fresh cascade stream with common random numbers across methods.

**Joint versus task-only training (RQ3).** Joint training changes the anomaly ROC-AUC of the GAT reconstruction score by +0.005 on Cora (five seeds, positive in all five, but within the seed spread) and by +0.007 on Weibo (positive in one of three seeds; paired-difference tables in Section 5.2). For influence, the GAT head trained jointly is not better than the influence-only head: on Weibo the influence-only head is higher by 0.151 Spearman (all three seeds), on Cora lower by 0.031 in the joint model (one of five seeds in favour of joint training). We therefore find no evidence that sharing the encoder helps either task at this scale; the effect, if any, is small relative to the variance between seeds and between datasets.

**Does attention explain the model? (RQ4).** Removing the edges with the highest attention at an anomalous test node lowers the node's reconstruction-based anomaly score more than removing the same number of random edges, on both datasets: pooled over seeds the difference is +0.019 on Cora (95 % bootstrap interval excluding zero, Wilcoxon p = 0.0095) and +0.067 on Weibo (interval [+0.017, +0.143], p = 0.003); low-attention removals have the smallest effect. For the influence head the two datasets disagree. On Weibo, where attention is sharper, removing the highest-attention outgoing edges of a high-influence node lowers its predicted influence by +0.298 more than random removal (interval [+0.176, +0.430], p = 0.0001, 63 % of targets). On Cora, where attention is almost uniform, it does not (difference -0.138, interval including zero, p = 0.75). So attention marks edges that matter more to the model where it is non-uniform, which is what one would expect, but the effect sizes are modest and vary between seeds (edge-removal tables in Section 5.3). This is an association with the model's output, not evidence about the real network, and it matches the cautions in the literature [@jain2019attention; @wiegreffe2019notnot].

**Failure cases, preprocessing and unresolved issues.** (1) *Data version.* Our first Weibo benchmark used the current upstream file, in which the labels of all validation and test nodes were blanked (Section 5.1); it treated 521 of the 868 true anomalies as normal. We found this through the dataset validation, discarded those runs (kept in `artifacts/superseded/`), pinned the data to the original commit with a checksum and repeated every Weibo experiment. The comparison of the two sets of runs is in Section 5.1: the corrupted labels lowered ROC-AUC by 0.02 to 0.05 and cut average precision by a factor of about three. (2) Weibo features are extremely heavy-tailed; clipping or log-compressing them in a validation-only check gave lower validation ROC-AUC (0.669 and 0.614, against 0.752 with plain standardisation), so plain standardisation was used. Earlier exploratory runs also printed test metrics, and although every choice was made from validation numbers, we cannot exclude an unconscious influence. (3) Weibo training uses random arc subsampling (30 % per epoch) to fit the CPU budget; its effect on the final models was not ablated. (4) The influence targets are simulations; their standard error is stored per node and is small compared with the differences between methods, but the proxy's validity for real cascades is untested. (5) Weibo timings were measured on an otherwise idle machine; the Cora benchmark timings were taken while other work shared the CPU and are about twice the isolated profile, so the isolated profiles in Section 5.3 are the cleaner cost figures for Cora.

## 5. Experimental Validations

### 5.1 Test set (dataset) details and performance

**Dataset authenticity and validation.** Before any result is reported, `scripts/validate_datasets.py` checks that the data are what they claim to be. The summary below groups the individual checks (42 passed, none failed, two findings and one partial agreement; full list: `docs/dataset_validation.md` and the dashboard's *Dataset authenticity* page): SHA-256 of every downloaded file against a pinned upstream commit and against a fresh download, structural validity (finite features, valid edge ids, symmetric clean graph, binary labels), equality with the published node, edge, feature and anomaly counts, label semantics, the integrity of the Cora injection and the properties of the splits.

Table: Summary of the dataset validation (one row per dataset and check family).
{{table:dataset_validation_summary.csv}}

The validation produced one important finding and one open point. The finding: the current upstream Weibo file differs from the original in its labels only (features and edges are identical): its validation- and test-mask nodes were set to normal, so just 347 of 868 anomalies remain and none in the test mask. Only the original 2022 file reproduces the published statistics (8,405 nodes, 407,963 listed edges, 400 features, 868 anomalies), so it is the one used here. The open point: our node-versus-neighbour feature similarity for Weibo (0.445 for anomalies against 0.052 for normal users) does not reproduce the benchmark paper's 0.004 against 0.993, whereas the clustering difference does agree in direction (0.499 against 0.429). The paper does not define its similarity exactly, so we report this as partial agreement and make no claim about the semantic meaning of the Weibo labels beyond what the benchmark paper states (users with suspicious posting behaviour; the posting times themselves are not part of the data). The Cora checks confirm that each of the five injected datasets contains seven complete 10-cliques, 70 contextual nodes whose attributes are copies of other nodes' attributes, untouched normal nodes, and is exactly reproducible from its seed. Licences were not verified (the benchmark repository is MIT-licensed; the terms of the underlying Weibo and Reddit data are not stated in the sources we opened).

Table: Why the validation matters: Weibo anomaly metrics with the authentic labels against the superseded runs that used the corrupted upstream labels (test split, mean ± standard deviation over three seeds).
{{table:label_version_weibo.csv}}

The comparison shows the practical cost of a silent label problem. With 60 % of the anomalies labelled as normal, the models look worse (ROC-AUC lower by 0.015 to 0.05) and, much more strongly, average precision falls by a factor of about three, because every correctly flagged hidden anomaly is counted as a false positive. The ranking of methods happens to stay similar, so the qualitative conclusions would have survived, but the absolute numbers, and any comparison with published values, would not have been valid.

Dataset sizes, class balance and splits are in the dataset table of Section 4. Both datasets are evaluated transductively: the encoder sees the attributes and edges of validation and test nodes, but no labels and no influence targets of those nodes. The fraction of anomalies in each split is stratified, so the test split always contains both classes (70 anomalies of 1,354 test nodes on Cora with seed 0, and 434 of 4,203 on Weibo). The following per-seed values show the spread that the aggregate tables hide.

Table: Test ROC-AUC per seed, injected Cora (five seeds).
{{table:anomaly_auc_by_seed_cora_injected.csv}}

Table: Test ROC-AUC per seed, Weibo (three seeds).
{{table:anomaly_auc_by_seed_weibo.csv}}

The per-seed tables show that on Cora the ordering of the broad groups is the same in every seed (Isolation Forest lowest, the reconstruction models next, the attention-only, combined and uniform-control scores highest), while differences inside a group change from seed to seed; and that Weibo, with only three seeds, has larger spread for the attention-based scores than for the GCN, which is itself informative: the attention-only score depends on the random initialisation more than a plain reconstruction does.

{{fig:roc_pr_cora_injected.png|ROC and precision-recall curves on injected Cora (test nodes, seed 0).}}

{{fig:roc_pr_weibo.png|ROC and precision-recall curves on Weibo (test nodes, seed 0).}}

{{fig:confusion_cora_injected.png|Test confusion matrices at the validation-selected thresholds, injected Cora, seed 0.}}

{{fig:confusion_weibo.png|Test confusion matrices at the validation-selected thresholds, Weibo, seed 0.}}

The confusion matrices show the practical consequence of the moderate average precision. At the F1-optimal validation threshold the Weibo GAT reconstruction detector flags about two normal users for every anomalous one it finds (156 true positives and 290 false positives, with 278 of the 434 test anomalies missed), and the attention-only score trades precision for recall (it flags 1,017 users, 258 of them anomalous). On Cora the reconstruction detector is more balanced (21 true positives, 24 false positives, 49 missed). Metrics at a validation-selected threshold depend on how well validation and test distributions match; ROC-AUC and average precision, which do not depend on a threshold, are therefore the main comparison metrics.

### 5.2 Baseline Comparison

Table: Anomaly detection on injected Cora, test split, mean ± standard deviation over five seeds. AP = average precision (step-wise area under the precision-recall curve); P@k = precision among the k highest-scored nodes with k equal to the number of true anomalies.
{{table:anomaly_cora_injected.csv}}

Table: Anomaly detection on Weibo (authentic labels), test split, mean ± standard deviation over three seeds.
{{table:anomaly_weibo.csv}}

{{fig:anomaly_auc_cora_injected.png|Test ROC-AUC on injected Cora (mean and standard deviation over five seeds). The horizontal axis starts at chance level.}}

{{fig:anomaly_auc_weibo.png|Test ROC-AUC on Weibo (mean and standard deviation over three seeds). The dotted line marks chance level.}}

*Interpretation.* Both reconstruction baselines (GCN, GAT) clearly beat Isolation Forest on both datasets (paired gap +0.095 on Cora and +0.313 on Weibo, in every seed), so the graph adds information that attributes alone do not carry. Between the GNN encoders the differences are small: on Cora GAT, GATv2 and GCN are within 0.006 of each other; on Weibo the plain GCN is slightly ahead, and the attention-based encoders do not offer an accuracy advantage that would justify their extra cost (GATv2 needs about 1.5 times the training time on Weibo). The attention-only score is the interesting case for the proposal's hypothesis: it is competitive with reconstruction on Cora but not on Weibo, and in neither case does it beat the control that ignores the learned attention. What this does not establish is that attention is useless in general: the learned attention here is nearly uniform on Cora, so the experiment cannot tell whether sharper attention would help.

Table: Influence ranking on injected Cora at p = 0.10, held-out test nodes. Seed-set spread is the expected number of activated nodes for the 10 (20) nodes each method selects, simulated on a fresh cascade stream.
{{table:influence_cora_injected.csv}}

Table: Influence ranking on Weibo at p = 0.003, held-out test nodes.
{{table:influence_weibo.csv}}

{{fig:influence_spearman_cora_injected.png|Rank correlation with held-out simulated spread, Cora (p = 0.10).}}

{{fig:influence_spearman_weibo.png|Rank correlation with held-out simulated spread, Weibo (p = 0.003).}}

{{fig:seedset_spread_cora_injected.png|Expected spread of the selected 10-node seed set, Cora (p = 0.10).}}

{{fig:seedset_spread_weibo.png|Expected spread of the selected 10-node seed set, Weibo (p = 0.003).}}

{{fig:influence_scatter_weibo.png|Held-out test nodes on Weibo (seed 0): simulated spread against three scores. Received attention and its degree-adjusted version do not track spread.}}

{{fig:influence_p_sensitivity_cora_injected.png|Sensitivity of the Spearman correlation to the cascade probability, Cora.}}

{{fig:influence_p_sensitivity_weibo.png|Sensitivity of the Spearman correlation to the cascade probability, Weibo.}}

*Interpretation.* The influence results separate three kinds of signal. Degree and PageRank are cheap and strong. A learned head on a degree-aware encoder (the GCN) matches or beats them (on Weibo it gains +0.260 Spearman over degree in every seed and its top-10 seed set reaches the oracle spread), so a learned ranker can add value when it is allowed to use the information that matters. Attention-derived scores are weakest: on Cora they equal a trivial non-learned control, on Weibo they are worse than it, and their seed sets are clearly below degree on Weibo. The practical reading is that received attention is not a substitute for a centrality measure under this influence proxy. As the cascade probability grows, degree loses correlation while the GCN head gains (sensitivity figures), which suggests that at higher spreading probabilities structure beyond degree matters more; we did not test this further.

Table: Seed-paired differences between selected methods (same graph, split and cascade streams per seed).
{{table:paired_cora_injected.csv}}

Table: Seed-paired differences, Weibo.
{{table:paired_weibo.csv}}

*Interpretation.* Because the seeds share graph, split and simulation streams, paired differences remove most seed-to-seed variation. The sign counts show which comparisons are consistent: the GCN-over-GAT influence head (all seeds on both datasets), received attention below the uniform control on Weibo (all seeds), and degree adjustment hurting (all seeds). Differences within about 0.01 (for example GATv2 against GAT) are not distinguishable from noise with three to five seeds.

**Published numbers for context only.** The benchmark paper reports unsupervised detector ROC-AUC means on its own splits and implementations (injected Cora, DOMINANT 82.7 ± 5.6 and AnomalyDAE 83.4 ± 2.3; Weibo, DOMINANT 85.0 ± 14.6, GCNAE 90.8, AnomalyDAE 91.5, with simple methods such as Radar above 98) [@liu2022bond]. These values are never mixed into the locally measured tables. With the original labels our Weibo label count now equals theirs (868), but the protocol (our 50 % test split, our training budget of 60 epochs with arc subsampling, no tuning per model) and the implementations differ, so we draw only coarse conclusions: our reconstruction baselines on Cora (0.77) are in the range they report for DOMINANT, and on Weibo (0.77 to 0.79) they are below the published 85 to 91; we did not investigate whether training budget, hyperparameters or the arc subsampling explain this gap. We also did not run Radar or ANOMALOUS, so we cannot reproduce the published Weibo ceiling.

### 5.3 Ablation Study

Table: Ablations on injected Cora (mean ± standard deviation over five seeds). Parameter counts are exact; GATv2 uses two weight matrices per layer and therefore about twice the encoder parameters, while one-head and four-head GAT have identical parameter counts.
{{table:ablation_cora_injected.csv}}

Table: Ablations on Weibo (mean ± standard deviation over three seeds).
{{table:ablation_weibo.csv}}

{{fig:ablation_auc_cora_injected.png|Ablations: anomaly ROC-AUC on injected Cora.}}

{{fig:ablation_auc_weibo.png|Ablations: anomaly ROC-AUC on Weibo.}}

*Interpretation.* Six design choices were varied. (1) *Encoder family:* no encoder is consistently best for anomalies; for influence the GCN head is far better than the attention heads. (2) *Heads:* one head of width 64 and four heads of width 16 have identical parameter counts and similar anomaly scores (on Weibo the single head is nominally ahead, 0.800 against 0.777, but the paired gap is within the spread), so multi-head attention did not pay off here. (3) *Score type:* reconstruction and attention-only differ by dataset, the combination helps on Cora and hurts on Weibo, and the uniform control beats attention-only on Weibo. (4) *Training mode:* joint and anomaly-only training give statistically indistinguishable anomaly scores (the influence-only model has no trained reconstruction head and therefore no anomaly score), and joint training does not improve the influence head. (5) *Degree adjustment* reduces influence ranking quality on both datasets. (6) *Untrained attention* equals trained attention for anomaly scoring, which is the clearest sign that the learned attention contributes little to that score.

Table: Attention statistics (means over seeds): normalised attention entropy (1 = uniform) in layers 1 and 2, and the rank correlation between received attention and degree. Cora.
{{table:attention_stats_cora_injected.csv}}

Table: Attention statistics, Weibo.
{{table:attention_stats_weibo.csv}}

Table: Edge-removal test of attention as an explanation, Cora. Effect = output before minus output after removal; positive means the removal lowered the output.
{{table:explanation_cora_injected.csv}}

Table: Edge-removal test, Weibo.
{{table:explanation_weibo.csv}}

{{fig:explanation_cora_injected.png|Mean drop in output after removing high-attention, random and low-attention edges, Cora.}}

{{fig:explanation_weibo.png|Mean drop in output after removing high-attention, random and low-attention edges, Weibo.}}

*Interpretation.* The attention statistics explain much of the pattern. Where attention is nearly uniform (Cora) it cannot rank neighbours, so received attention equals the uniform control and the edge-removal test finds no influence effect. Where attention is sharper (Weibo, entropy about 0.6) it picks out edges that matter more to the model (the high-minus-random effect is positive for both outputs, with p-values of 0.003 and 0.0001), yet the sharper attention still does not translate into better rankings or better detection than the controls. In other words, attention can be a meaningful *description of what the trained model uses* without being a better *predictor of influence or anomaly* than simpler quantities. The perturbation test measures only the first.

{{include:sensitivity.md}}

Table: Measured training and inference cost (mean ± standard deviation over seeds; wall-clock; process memory is a cumulative upper bound because several runs shared one process). Cora.
{{table:resources_cora_injected.csv}}

Table: Measured cost, Weibo.
{{table:resources_weibo.csv}}

{{include:profile.md}}

**What the results establish and what they do not.** They establish, for these two graphs and this budget: (a) neighbour feature discrepancy and reconstruction error are useful anomaly signals, and a non-attentive control matches or beats the attention-only score; (b) degree and a plain GCN head are strong influence rankers under the independent-cascade proxy, and learned GAT attention adds nothing beyond a uniform-attention control on Cora and hurts overall ranking on Weibo; (c) high-attention edges matter more than random ones to the trained model's anomaly output on both datasets and to its influence output where attention is sharp; (d) dataset validation is not optional: a changed upstream label file would have biased every Weibo result. They do not establish how attention behaves on other graphs, with measured cascades, with longer training, with other attention regularisers, or on organic anomalies other than the Weibo labels; they do not show that joint training is harmful or useful in general; and they say nothing about causal explanation of the real network. The class imbalance (about 4 to 10 % anomalies), the label-informed selection of the mixing weight and threshold, the transductive setting and the small number of seeds all limit generalisation.
