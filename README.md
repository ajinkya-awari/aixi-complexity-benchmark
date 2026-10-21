# AIXI-Complexity Benchmark

Reproducible benchmark for evaluating decision-making agents on finite-state controlled tasks with declared complexity proxies.

**Status:** HISTORICAL scaffold only. `IMPLEMENTED-UNVERIFIED` / `NOT AUTHORIZED` / `NOT EXECUTED`.

This directory is present on disk but is not an approved source repository and is not release evidence. Gate 0 is not released, separate Project 20 implementation authorization is absent, and this scaffold must not be installed, tested, run, expanded, published, pushed, deployed, or used to claim benchmark readiness during the planning freeze.

No benchmark ran. No task hash was produced. No seed manifest was executed. No transition or reward table was generated. No model or tokenizer was accessed. No model arm was evaluated. No results or metrics exist.

## Quick Start

Blocked during the current planning freeze. The commands below are historical/future command shapes only; they are `PLANNED`, `NOT EXECUTED`, `NOT AUTHORIZED`, and `RELEASE-GATED`.

```bash
# From the repo root
python -m pip install -e .
aixi-bench --help
```

## Agents

- Random policy (baseline)
- Tabular Q-learning
- Exact oracle (value iteration)
- AC-CTW-UCT (MC-AIXI-inspired planner)
- Local LLM policy-table generation (Qwen2.5-0.5B, gated)

## Tasks

The YAML here is an incomplete historical scaffold, not a complete frozen scientific task corpus. Canonical hashes, seed manifests, transition tables, and reward tables are absent.

20 frozen finite-state tasks: 5 families × 4 variants
- `reactive`: Observation directly identifies best action
- `contextual`: Observation aliases latent states; bounded history needed
- `delayed`: Reward depends on delayed action consequences
- `branching`: Multiple action-conditioned transition branches
- `sparse_sequence`: Success requires fixed action/percept sequence

## Documentation

- `DESIGN.md` — Scientific contract and specification
- `FINAL_VULNERABILITY_SCAN.md` — Planning constraints
- `notebooks/KAGGLE_RUNBOOK_aixi_complexity.md` — Kaggle execution plan
- `notebooks/SYNTHETIC_TEST_DESIGNS.md` — Test specifications

## Development

No tests are authorized or present in this scaffold. The command below is a future command shape only and was not executed:
```bash
pytest tests/ -v
```

## Constraints

- **Local-only:** No remote APIs (OpenAI, Anthropic)
- **Reproducible:** Exact dependency pins, frozen seeds
- **Honest:** Proxies are declared, not true Kolmogorov complexity
- **Publication-gated:** Results remain draft until authorized

See `DESIGN.md` for complete contract.
