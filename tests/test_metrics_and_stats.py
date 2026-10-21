import math

import numpy as np
import pandas as pd

from aixi_bench import metrics, stats


def test_normalized_gap_formula_unclipped():
    assert metrics.normalized_oracle_gap(10, 5, 0) == 0.5
    assert metrics.normalized_oracle_gap(10, 10, 0) == 0.0
    assert metrics.normalized_oracle_gap(10, 15, 0) == -0.5  # unclipped, can be negative
    assert metrics.normalized_oracle_gap(10, -10, 0) == 2.0  # unclipped above 1


def test_normalized_gap_na_when_oracle_equals_random():
    assert math.isnan(metrics.normalized_oracle_gap(5.0, 3.0, 5.0))
    assert math.isnan(metrics.normalized_oracle_gap(5.0, 3.0, 5.0 + 5e-9))


def _episodes():
    rows = []
    for rep in range(2):
        for agent, rets in {"oracle": [4, 4], "random": [0, 2], "q_learning": [2, 2]}.items():
            for i, r in enumerate(rets):
                rows.append(dict(task_id="t", rep=rep, agent=agent, episode=i, seed=i, ret=float(r),
                                 success=int(r >= 2), steps=4, runtime_s=0.1, logloss_bits=None))
    return pd.DataFrame(rows)


def test_replicate_table_pairs_v_values():
    t = metrics.replicate_table(_episodes())
    q = t[(t.agent == "q_learning") & (t.rep == 0)].iloc[0]
    assert (q.v_oracle, q.v_random, q.v_agent) == (4.0, 1.0, 2.0)
    assert math.isclose(q.normalized_oracle_gap, (4 - 2) / 3)
    assert q.success_rate == 1.0


def test_bootstrap_uses_exactly_10000_resamples_and_is_deterministic():
    assert stats.N_RESAMPLES == 10_000
    v = np.arange(10, dtype=float)
    a = stats.paired_block_bootstrap(v)
    assert a == stats.paired_block_bootstrap(v)
    assert a[1] < a[0] < a[2]


def test_bootstrap_handles_na_blocks():
    v = np.array([1.0, np.nan, 3.0, np.nan])
    mean, lo, hi = stats.paired_block_bootstrap(v)
    assert mean == 2.0 and 1.0 <= lo <= hi <= 3.0
    assert all(math.isnan(x) for x in stats.paired_block_bootstrap(np.full(4, np.nan)))


def test_hierarchical_bootstrap_wider_than_zero_when_tasks_differ():
    m = np.vstack([np.full(10, 0.0), np.full(10, 1.0)])
    mean, lo, hi = stats.hierarchical_bootstrap(m, n=2000)
    assert mean == 0.5 and lo < 0.5 < hi


def test_holm_matches_reference():
    adj = stats.holm_adjust([0.01, 0.04, 0.03, float("nan")])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and math.isnan(adj[3])


def test_proxy_associations_are_exploratory_and_holm_corrected():
    corpus = {"tasks": [{"task_id": f"t{i}", "proxy_vector": {
        "reachable_history_state_count": i, "public_history_order_m": 1, "horizon": 8 + i,
        "branching_factor": 2.0, "reward_density_uniform_policy": 0.5 - i * 0.01,
        "canonical_task_spec_utf8_bytes": 100 + i}} for i in range(6)]}
    summary = pd.DataFrame([dict(task_id=f"t{i}", agent="q_learning", normalized_oracle_gap=i * 0.1 + (i % 2) * 0.05)
                            for i in range(6)])
    out = stats.proxy_associations(summary, corpus, agents=("q_learning",))
    assert set(out.columns) >= {"spearman_rho", "p_value", "p_holm", "analysis_status"}
    const = out[out.proxy_dimension == "public_history_order_m"].iloc[0]
    assert math.isnan(const.spearman_rho)  # constant dimension: no association reported
    assert (out.p_holm.dropna() >= out.p_value.dropna()).all()
    assert "exploratory" in out.analysis_status.iloc[0]
