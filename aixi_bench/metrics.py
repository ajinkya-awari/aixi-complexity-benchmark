"""Metric definitions. Primary metric: normalized oracle gap.

    normalized_oracle_gap = (V_oracle - V_agent) / max(|V_oracle - V_random|, 1e-8)

V_agent, V_oracle and V_random are paired empirical mean returns over the same 100 held-out
seeds of one task x data-replicate. The gap is reported unclipped; when
|V_oracle - V_random| < 1e-8 it is NA (NaN) and raw return is retained.
"""
import pandas as pd

EPSILON = 1e-8


def normalized_oracle_gap(v_oracle, v_agent, v_random):
    denom = abs(v_oracle - v_random)
    if denom < EPSILON:
        return float("nan")
    return (v_oracle - v_agent) / max(denom, EPSILON)


def replicate_table(episodes: pd.DataFrame, steps: pd.DataFrame = None) -> pd.DataFrame:
    """One row per (task, replicate, agent) with paired V values and the primary metric."""
    g = episodes.groupby(["task_id", "rep", "agent"])
    rows = g.agg(v_agent=("ret", "mean"), success_rate=("success", "mean"),
                 runtime_per_task_s=("runtime_s", "mean"), mean_logloss_bits=("logloss_bits", "mean"),
                 n_episodes=("ret", "size")).reset_index()
    pivot = rows.pivot_table(index=["task_id", "rep"], columns="agent", values="v_agent")
    rows["v_oracle"] = [pivot.loc[(t, r), "oracle"] for t, r in zip(rows.task_id, rows.rep)]
    rows["v_random"] = [pivot.loc[(t, r), "random"] for t, r in zip(rows.task_id, rows.rep)]
    rows["normalized_oracle_gap"] = [normalized_oracle_gap(o, a, r) for o, a, r in
                                     zip(rows.v_oracle, rows.v_agent, rows.v_random)]
    if steps is not None and len(steps):
        d = steps.groupby(["task_id", "rep", "agent"])["decision_runtime_s"].mean().rename(
            "runtime_per_decision_s").reset_index()
        rows = rows.merge(d, on=["task_id", "rep", "agent"], how="left")
    return rows


def summary_by_task(rep_table: pd.DataFrame, status: pd.DataFrame, agents) -> pd.DataFrame:
    """Task x agent summary (mean over data replicates) including the invalid-output rate."""
    out = []
    for (task, agent), grp in rep_table.groupby(["task_id", "agent"]):
        st = status[(status.task_id == task) & (status.agent == agent)]
        invalid = float((st.status == "invalid").mean()) if len(st) else float("nan")
        gaps = grp.normalized_oracle_gap
        out.append(dict(
            task_id=task, agent=agent, n_replicates=len(grp),
            mean_return=grp.v_agent.mean(),
            normalized_oracle_gap=float(gaps.mean()) if gaps.notna().any() else float("nan"),
            success_rate=grp.success_rate.mean(),
            invalid_output_rate=invalid if agent.startswith("local_llm") else 0.0,
            runtime_per_task_s=grp.runtime_per_task_s.mean(),
            runtime_per_decision_s=grp.get("runtime_per_decision_s", pd.Series(dtype=float)).mean(),
            mean_logloss_bits=grp.mean_logloss_bits.mean() if grp.mean_logloss_bits.notna().any() else float("nan"),
        ))
    for arm in ("local_llm_policy", "local_llm_constrained"):
        missing = status[(status.agent == arm) & (status.status != "evaluated")]
        for task, grp in missing.groupby("task_id"):
            if any(r["task_id"] == task and r["agent"] == arm for r in out):
                continue
            total = len(status[(status.agent == arm) & (status.task_id == task)])
            out.append(dict(task_id=task, agent=arm, n_replicates=0, mean_return=float("nan"),
                            normalized_oracle_gap=float("nan"), success_rate=float("nan"),
                            invalid_output_rate=float((grp.status == "invalid").sum() / total),
                            runtime_per_task_s=float("nan"), runtime_per_decision_s=float("nan"),
                            mean_logloss_bits=float("nan")))
    return pd.DataFrame(out).sort_values(["task_id", "agent"]).reset_index(drop=True)
