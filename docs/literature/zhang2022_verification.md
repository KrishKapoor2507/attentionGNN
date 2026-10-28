# Verification of "Zhang et al. (2022), Attention-based Anomaly Detection on Graphs"

## Verdict: NOT VERIFIABLE

No paper with this exact title, a Zhang author and the year 2022 was found in any index that could be reached. Do not cite it as written. A citation with no venue is itself a red flag, and none of the near-matches below is the same paper. Do not swap one of them in silently.

Absence of a hit in these tools does not prove the paper does not exist (Google Scholar, DBLP, Semantic Scholar and the ACM/IEEE libraries were not reachable, see the log). If the proposal's author has the PDF, a DOI or a URL, send it and it can be checked in a minute. Until then the safe options are (a) delete the citation, or (b) replace it with one of the verified papers listed under "Verified alternatives".

## Search log (2026-10-25; query -> outcome)

| # | Tool / index | Query | Outcome |
|---|---|---|---|
| 1 | Web search (extended) | "Attention-based Anomaly Detection on Graphs" Zhang 2022 | No exact title. Returned time-series GAT papers (e.g. a 2023 J. Supercomputing paper by Wei Zhang et al. on multivariate time series) and unrelated graph-anomaly surveys. |
| 2 | Web search | "Attention-based anomaly detection on graphs" arXiv | No exact title. Near-matches: AHEAD (arXiv 2208.08200), ResGCN (2009.14738), GRAM (2311.06153), AddGraph (IJCAI 2019), CoBaGAD (HAL). |
| 3 | arXiv API | `ti:"attention-based anomaly detection"` | 2 records, neither about graphs (LATTE, automotive LSTM; a power-systems transformer paper). |
| 4 | arXiv API | `all:"anomaly detection" AND all:graph AND all:attention AND au:Zhang`, newest first | 30 records (2020-2026). None titled like the target. 2022 items: AHEAD (2208.08200), AnomMAN (2201.02822), a multi-scale contrastive GAD paper (2212.00535). |
| 5 | Crossref API | title query "Attention-based Anomaly Detection on Graphs" (15 rows) | No exact title. Closest: "Graph-Based Anomaly Detection via Attention Mechanism" (Yu et al., 2022, LNCS), "Fine-Grained Anomaly Detection on Dynamic Graphs via Attention Alignment" (Chen et al., ICDE 2024). |
| 6 | Crossref API | bibliographic query + "Zhang" filtered to 2021-2023 (20 rows) | All video/time-series/industrial attention papers; none a graph-anomaly paper titled as above. |
| 7 | OpenAlex API | search "attention-based anomaly detection on graphs" (20 rows shown of ~91k) | No exact title (list included GATAE 2020, Yu et al. 2022, AHEAD, DuSAG 2022 ...). |
| 8 | OpenAlex API | title.search filter 2021-2023 (17 rows) | No exact title. Zhang-authored graph+attention items were time-series/sensor or skeleton-video papers. |
| 9 | Web search (extended) | Zhang 2022 graph attention network unsupervised node anomaly attributed networks | Closest Zhang-2022 hit: Zhang F. et al., Deep Dual SVDD for attributed networks (not attention). Attention-based alternative by other authors: ResGCN (Pei et al.). |
| 10 | Semantic Scholar (web + API), DBLP (search API), ACM DL, Springer | various | Blocked (HTTP 403 / 429 / bot-protection page). Not searched. Google Scholar was not queried (no tool access). |

## Near-matches (all real, none is the cited paper)

Verified = an arXiv/publisher page was opened.

| Candidate | Why it looks similar | Why it is NOT the cited paper | Status |
|---|---|---|---|
| Zhang F., Fan H., Wang R., Li Z., Liang T. (2022), "Deep Dual Support Vector Data Description for Anomaly Detection on Attributed Networks", Int. J. Intell. Syst. (arXiv 2109.00138) | A "Zhang et al.", 2022, node anomaly detection on attributed networks | Title differs; method is a dual autoencoder with dual-hypersphere SVDD, abstract mentions no attention | Verified (arXiv abs page) |
| Pei Y., Huang T., van Ipenburg W., Pechenizkiy M. (2022), "ResGCN: Attention-based Deep Residual Modeling for Anomaly Detection on Attributed Networks", Machine Learning 111(2) (arXiv 2009.14738) | "Attention-based", "Anomaly Detection", 2022 | First author Pei, not Zhang; title differs. Attention here is a residual-based attention that down-weights anomalous nodes | arXiv verified; journal details from a secondary record |
| Fan H., Zhang F., Li Z. (2020), "AnomalyDAE", ICASSP 2020 (arXiv 2002.03665) | Attention in the structure encoder; F. Zhang is a middle author | 2020, title differs, first author Fan | Verified (arXiv abs page) |
| Yang S., Zhang B., Feng S., Tan Z., Zheng Q., Zhou J., Luo M. (2022), "AHEAD: A Triple Attention Based Heterogeneous Graph Anomaly Detection Approach", arXiv 2208.08200 (later LNEE 2023) | 2022, attention, graph anomaly detection; B. Zhang is second author | Heterogeneous graphs; first author Yang; title differs | Verified (arXiv abs page) |
| Yu Y., Zha Z., Jin B., Wu G., Dong C. (2022), "Graph-Based Anomaly Detection via Attention Mechanism", LNCS (ICIC 2022), doi 10.1007/978-3-031-13870-6_33 | Closest title in meaning, 2022 | No Zhang author | Record seen only in Crossref/OpenAlex output; paper NOT opened |
| Zhang W., He P., Qin C., Yang F., Liu Y. (2023), "A graph attention network-based model for anomaly detection in multivariate time series", J. Supercomputing, doi 10.1007/s11227-023-05772-5 | Zhang, graph attention, anomaly detection | 2023; time-series sensors, not graph-structured data; title differs | Record seen only in search/API output; NOT opened |
| Yang Y., Wang P., He X., Zou D. (2023/2024), "GRAM: ... Gradient Attention Maps", Neural Networks 178 (2024) 106463 (arXiv 2311.06153) | Attention map for graph anomaly detection | Graph-level setting; gradient-based attention; not Zhang | Verified (arXiv abs page) |

## Verified alternatives to cite instead (attention + graph anomaly detection)

- Fan, Zhang, Li (2020) AnomalyDAE, `fan2020anomalydae`: attention in the structure encoder, reconstruction-error scoring.
- Pei et al. (2022) ResGCN, `pei2022resgcn`: residual-based attention for attributed-network anomaly detection.
- Gu, Gao, Lou, Zhang (2021) GANR, `gu2021ganr` (note: author J. Zhang, link prediction; attention used as a centrality score, not an anomaly score). Do not confuse this with the proposal's citation either.

## Recommendation

Remove "Zhang et al. (2022)" from the reference list unless its source can be produced. If the sentence it supports is "attention has been used for graph anomaly detection", cite AnomalyDAE and ResGCN, which were opened and read.
