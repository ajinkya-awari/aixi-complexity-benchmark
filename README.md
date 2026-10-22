# AIXI-Complexity Benchmark

A small, fully local benchmark that compares decision-making agents on 20 frozen finite-state controlled tasks whose
difficulty is described by a vector of declared proxies. Nothing here calls a remote model or API.

**Status:** local implementation complete; public release complete with limitations (see below). This is not a claim of
scientific completeness, and nothing here is evidence about AIXI, AGI or universal intelligence.

## What is in the benchmark

- **Tasks:** 20 frozen tasks, 5 families (`reactive`, `contextual`, `delayed`, `branching`, `sparse_sequence`) x 4 variants
  (V1-V4). Each has 2 actions, 2-8 latent states, 2-4 public observations and a horizon in {8, 12, 16, 24, 32}.
  The public history key (last m action/observation pairs plus the current observation, m <= 3) is verified to
  identify the latent state, so every task is Markov in its public key. Task specs and SHA-256 hashes are in
  `task_specs/frozen_finite_tasks.yaml` and `artifacts/manifests/task_hashes.json`.
- **Proxy vector (reported separately, never collapsed to one score):** reachable history-state count, public history
  order m, horizon, branching factor, reward density under the uniform policy, canonical task-spec byte length.
  These are declared proxies, not Kolmogorov complexity.
- **Data protocol:** 10 independent data replicates per task; each has 256 uniform-random public exploration episodes
  and 100 held-out evaluation episodes (frozen seeds). All arms see the same held-out seeds; no learning happens during
  evaluation.
- **Arms:** random; tabular Q-learning (alpha 0.1, gamma 0.99, 20 deterministic shuffled passes); exact oracle (finite-
  horizon dynamic programming); AC-CTW-UCT (action-conditional context tree weighting with 512 UCT simulations,
  planning horizon min(16, remaining), exploration sqrt(2); inspired by MC-AIXI-CTW, not a faithful reproduction); and a
  local policy-table arm using `Qwen/Qwen2.5-0.5B-Instruct` (pinned revision, greedy decoding, strict output validation,
  no fallback for invalid output).
- **Primary metric:** normalized oracle gap = (V_oracle - V_agent) / max(|V_oracle - V_random|, 1e-8), unclipped, computed per
  task x replicate from paired mean returns on the same 100 seeds.

## Results (suite level, from `results/local_run`)

Hierarchical bootstrap (10,000 resamples of tasks, then replicates), 95% intervals. These describe uncertainty for this
fixed 20-task suite only, not inference to all possible tasks.

| Arm | Mean return | Normalized oracle gap | 95% CI |
|---|---|---|---|
| Random | 6.17 | 1.000 | [1.000, 1.000] |
| Tabular Q | 10.92 | 0.333 | [0.247, 0.421] |
| AC-CTW-UCT | 12.14 | 0.199 | [0.076, 0.346] |
| Local Qwen policy table | 5.48 | 0.999 | [0.610, 1.351] |
| Exact oracle | 13.16 | 0.000 | [0.000, 0.000] |

Per-task numbers, per-replicate numbers, paired bootstrap contrasts and the exploratory Holm-corrected Spearman proxy
associations are in `results/local_run/` (`summary.csv`, `summary_by_replicate.csv`, `bootstrap_cis.csv`,
`proxy_analysis.csv`, `figures/`).

## Limitations

- The Qwen arm produced a valid policy table in only 86 of 200 task x replicate records; the other 114 are recorded as
  invalid and are excluded from value-based metrics (never replaced by random or other behavior). Its row above reflects
  the valid records only.
- Proxy associations are exploratory and not confirmatory; no monotonic relationship between proxies and performance is
  claimed.
- The task suite is small, synthetic and fixed; results do not generalize to other task distributions.
- Environment deviation: Python 3.12.13 was used (3.12.10 was unavailable); other pins are exact (see `pyproject.toml`).
- The AC-CTW-UCT agent uses a memoized, frozen percept model after offline fitting on the public exploration corpus.
- Drafts under `draft_outputs/` are unreviewed local drafts, not submitted or sent anywhere.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows layout; use .venv/bin on Linux/macOS
aixi-bench tasks validate
aixi-bench env-check
aixi-bench data verify
aixi-bench model-gate          # downloads the pinned Qwen revision (about 1 GB) into the Hugging Face cache
aixi-bench llm-policy          # slow on CPU
aixi-bench run && aixi-bench run --agents local_llm_policy
aixi-bench metrics && aixi-bench bootstrap && aixi-bench proxy-analysis && aixi-bench verify-run
pytest -q && ruff check .
```

The full evaluation is CPU-heavy (the AC-CTW-UCT arm dominates); expect hours on a laptop.

## Layout

`aixi_bench/` package, `tests/`, `task_specs/` frozen corpus, `artifacts/` (manifests, gate evidence, frozen exploration
corpora), `results/local_run/` outputs, `draft_outputs/` local drafts.

## Attribution

Context Tree Weighting: Willems, Shtarkov and Tjalkens. MC-AIXI-CTW: Veness et al. This project is only inspired by that line of
work.

## License

MIT.
