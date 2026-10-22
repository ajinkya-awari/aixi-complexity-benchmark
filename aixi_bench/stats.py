"""Bootstrap confidence intervals and exploratory proxy associations.

* Per-task paired intervals: 10,000 resamples of data-replicate blocks within a task.
* Aggregate intervals: hierarchical bootstrap (resample tasks, then replicates within each
  sampled task). This is suite-level descriptive uncertainty for the fixed 20-task suite, not
  inference to all possible finite-state tasks.
* Proxy associations: exploratory Spearman correlations with Holm correction.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", message="Mean of empty slice")

N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261003
CI_LEVEL = 0.95


def _ci(samples):
    lo, hi = np.nanpercentile(samples, [100 * (1 - CI_LEVEL) / 2, 100 * (1 + CI_LEVEL) / 2])
    return float(lo), float(hi)


def paired_block_bootstrap(values: np.ndarray, n=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """values: (n_replicates,) with NaN allowed. Resample replicate blocks; return (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    with np.errstate(all="ignore"):
        means = np.nanmean(values[idx], axis=1)
    lo, hi = _ci(means)
    return float(np.nanmean(values)), lo, hi


def hierarchical_bootstrap(matrix: np.ndarray, n=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """matrix: (n_tasks, n_replicates) with NaN allowed. Returns (mean, lo, hi)."""
    matrix = np.asarray(matrix, dtype=float)
    n_tasks, n_reps = matrix.shape
    rng = np.random.default_rng(seed)
    task_idx = rng.integers(0, n_tasks, size=(n, n_tasks))
    rep_idx = rng.integers(0, n_reps, size=(n, n_tasks, n_reps))
    gathered = matrix[task_idx[:, :, None], rep_idx]
    with np.errstate(all="ignore"):
        task_means = np.nanmean(gathered, axis=2)
        agg = np.nanmean(task_means, axis=1)
    with np.errstate(all="ignore"):
        point = np.nanmean(np.nanmean(matrix, axis=1))
    lo, hi = _ci(agg)
    return float(point), lo, hi


METRICS = ["normalized_oracle_gap", "v_agent"]


def _matrix(rep_table, agent, metric, tasks, reps):
    sub = rep_table[rep_table.agent == agent].set_index(["task_id", "rep"])[metric]
    m = np.full((len(tasks), len(reps)), np.nan)
    for i, t in enumerate(tasks):
        for j, r in enumerate(reps):
            if (t, r) in sub.index:
                m[i, j] = sub.loc[(t, r)]
    return m


def bootstrap_table(rep_table: pd.DataFrame, n=N_RESAMPLES, seed=BOOTSTRAP_SEED,
                    pair_agents=("q_learning", "ac_ctw_uct", "local_llm_policy", "local_llm_constrained")):
    tasks = sorted(rep_table.task_id.unique())
    reps = sorted(rep_table.rep.unique())
    agents = sorted(rep_table.agent.unique())
    rows = []
    for metric in METRICS:
        mats = {a: _matrix(rep_table, a, metric, tasks, reps) for a in agents}
        for a in agents:
            for i, t in enumerate(tasks):
                mean, lo, hi = paired_block_bootstrap(mats[a][i], n, seed)
                rows.append(dict(scope="task", task_id=t, contrast=a, metric=metric, mean=mean,
                                 ci_lo=lo, ci_hi=hi, resamples=n))
            mean, lo, hi = hierarchical_bootstrap(mats[a], n, seed)
            rows.append(dict(scope="suite_hierarchical", task_id="ALL", contrast=a, metric=metric, mean=mean,
                             ci_lo=lo, ci_hi=hi, resamples=n))
        for a in pair_agents:
            for b in pair_agents:
                if a >= b or a not in mats or b not in mats:
                    continue
                diff = mats[a] - mats[b]
                for i, t in enumerate(tasks):
                    mean, lo, hi = paired_block_bootstrap(diff[i], n, seed)
                    rows.append(dict(scope="task", task_id=t, contrast=f"{a} - {b}", metric=metric, mean=mean,
                                     ci_lo=lo, ci_hi=hi, resamples=n))
                mean, lo, hi = hierarchical_bootstrap(diff, n, seed)
                rows.append(dict(scope="suite_hierarchical", task_id="ALL", contrast=f"{a} - {b}", metric=metric,
                                 mean=mean, ci_lo=lo, ci_hi=hi, resamples=n))
    return pd.DataFrame(rows)


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values (NaN inputs stay NaN and are not counted)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


PROXY_DIMS = ["reachable_history_state_count", "public_history_order_m", "horizon", "branching_factor",
              "reward_density_uniform_policy", "canonical_task_spec_utf8_bytes"]


def proxy_associations(summary: pd.DataFrame, corpus: dict, agents=("random", "q_learning", "ac_ctw_uct",
                                                                      "local_llm_policy", "local_llm_constrained")):
    """Exploratory Spearman associations between each proxy dimension and the task-level
    normalized oracle gap, Holm-corrected across all tests. Never a confirmatory analysis."""
    proxies = pd.DataFrame([{**t["proxy_vector"], "task_id": t["task_id"]} for t in corpus["tasks"]])
    rows = []
    for agent in agents:
        s = summary[summary.agent == agent][["task_id", "normalized_oracle_gap"]].dropna()
        merged = s.merge(proxies, on="task_id")
        for dim in PROXY_DIMS:
            if len(merged) < 4 or merged[dim].nunique() < 2 or merged.normalized_oracle_gap.nunique() < 2:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = spearmanr(merged[dim], merged.normalized_oracle_gap)
            rows.append(dict(agent=agent, proxy_dimension=dim, n_tasks=len(merged), spearman_rho=rho,
                             p_value=p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out.p_value.values)
    out["analysis_status"] = "exploratory; not confirmatory; no monotonic conclusion implied"
    return out
