Table: Isolated single-run profiles (each run in a fresh process, nothing else running; seed 0; the GAT row also includes the edge-removal test, so its memory is the larger of training and that test). Cora.
{{table:resources_profile_cora_injected.csv}}

Table: Isolated single-run profiles, Weibo.
{{table:resources_profile_weibo.csv}}

The isolated profiles are the cleaner cost measurement. On Cora the isolated training times are about half of the benchmark averages, because the Cora benchmark ran while other work (builds, rendering, tests) shared the CPU; on Weibo the two agree (for example 52 s isolated against a 58 s benchmark average for the GAT) because that benchmark ran on an otherwise idle machine. Peak memory stays near 0.75 GB on Cora and between 1.1 and 1.5 GB on Weibo; GATv2 costs more time and memory than GAT, and GCN is cheapest. These numbers confirm that the full pipeline fits a laptop without a GPU, but they are single runs, and wall-clock times on other machines will differ.
