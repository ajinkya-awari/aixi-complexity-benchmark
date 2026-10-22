# AIXI-Complexity Benchmark (local draft)

Local draft generated from `results/local_run`. Every number below is copied from the run outputs.

## Setup

20 frozen finite-state tasks (5 families x 4 variants), 10 independent data replicates per task, 256 uniform
random public exploration episodes and 100 held-out evaluation episodes per task x replicate. Arms: random,
tabular Q-learning, exact oracle (dynamic programming), AC-CTW-UCT (MC-AIXI-inspired, not faithful MC-AIXI-CTW), and a
local `Qwen/Qwen2.5-0.5B-Instruct` policy-table arm. Task levels are described by a separate six-dimensional proxy
vector; they are not Kolmogorov complexity.

## Suite-level results

Normalized oracle gap = (V_oracle - V_agent) / max(|V_oracle - V_random|, 1e-8), unclipped. Intervals are
hierarchical bootstrap (10,000 resamples of tasks, then replicates), suite-level descriptive uncertainty for this fixed
20-task suite, not inference to all possible tasks.

| Arm | Mean return | Normalized oracle gap | 95% CI low | 95% CI high |
|---|---|---|---|---|
| Random | 6.17 | 1.000 | 1.000 | 1.000 |
| Tabular Q | 10.92 | 0.333 | 0.247 | 0.421 |
| AC-CTW-UCT | 12.14 | 0.199 | 0.076 | 0.346 |
| Local Qwen policy table | 5.48 | 0.999 | 0.610 | 1.351 |
| Local Qwen, digit-constrained (secondary) | 5.02 | 1.166 | 0.917 | 1.409 |
| Exact oracle | 13.16 | 0.000 | 0.000 | 0.000 |

Local Qwen policy-table status over task x replicate records (200 total): evaluated: 86, invalid: 114.
Invalid outputs are excluded from value-based metrics and are never replaced by other behavior.
Secondary digit-constrained Qwen arm status (200 total): evaluated: 200.

## Files

`summary.csv`, `summary_by_replicate.csv`, `bootstrap_cis.csv`, `proxy_analysis.csv` (exploratory, Holm-corrected),
`figures/`. See the evidence manifests for hashes.
