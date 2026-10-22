"""Plots, local-only drafts, and run verification. Every number comes from real run outputs."""
import json
from pathlib import Path

import pandas as pd
import yaml

AGENTS = ["random", "q_learning", "ac_ctw_uct", "local_llm_policy", "local_llm_constrained", "oracle"]
LABELS = {"random": "Random", "q_learning": "Tabular Q", "ac_ctw_uct": "AC-CTW-UCT",
          "local_llm_policy": "Local Qwen policy table",
          "local_llm_constrained": "Local Qwen, digit-constrained (secondary)", "oracle": "Exact oracle"}


def _fmt(x, nd=3):
    return "NA" if pd.isna(x) else f"{x:.{nd}f}"


def make_plots(run, tasks_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    summary = pd.read_csv(run / "summary.csv")
    corpus = yaml.safe_load(Path(tasks_path).read_text(encoding="utf-8"))
    order = [t["task_id"] for t in corpus["tasks"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_agents = [a for a in AGENTS if a not in ("oracle",) and a in set(summary.agent)]
    width = 0.8 / len(plot_agents)
    for i, agent in enumerate(plot_agents):
        sub = summary[summary.agent == agent].set_index("task_id").reindex(order)
        ax.bar([x + i * width for x in range(len(order))], sub.normalized_oracle_gap, width, label=LABELS[agent])
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(order))])
    ax.set_xticklabels(order, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Normalized oracle gap (lower = closer to oracle)")
    ax.set_title("Mean normalized oracle gap per task (mean over 10 data replicates)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "gap_by_task.png", dpi=150)
    plt.close(fig)

    proxies = pd.DataFrame([{**t["proxy_vector"], "task_id": t["task_id"]} for t in corpus["tasks"]])
    dims = ["reachable_history_state_count", "horizon", "reward_density_uniform_policy", "branching_factor"]
    fig, axes = plt.subplots(1, len(dims), figsize=(16, 4))
    for ax, dim in zip(axes, dims):
        for agent in plot_agents:
            s = summary[summary.agent == agent].merge(proxies, on="task_id")
            ax.scatter(s[dim], s.normalized_oracle_gap, label=LABELS[agent], s=18)
        ax.set_xlabel(dim)
        ax.set_ylabel("normalized oracle gap")
    axes[0].legend(fontsize=7)
    fig.suptitle("Exploratory view: gap vs proxy dimensions (proxies are not Kolmogorov complexity)")
    fig.tight_layout()
    fig.savefig(fig_dir / "gap_vs_proxies.png", dpi=150)
    plt.close(fig)
    print("wrote figures to", fig_dir)


def _suite_table(run):
    boot = pd.read_csv(Path(run) / "bootstrap_cis.csv")
    suite = boot[(boot.scope == "suite_hierarchical") & (boot.metric == "normalized_oracle_gap")
                 & ~boot.contrast.str.contains(" - ")]
    ret = boot[(boot.scope == "suite_hierarchical") & (boot.metric == "v_agent") & ~boot.contrast.str.contains(" - ")]
    return suite.set_index("contrast"), ret.set_index("contrast")


def verify_run(run, expect_tasks=20, expect_reps=10, expect_eval=100, expect_resamples=10000):
    run = Path(run)
    problems = []
    eps = pd.read_parquet(run / "episodes.parquet")
    status = pd.read_csv(run / "agent_status.csv")
    if eps.task_id.nunique() != expect_tasks:
        problems.append(f"tasks {eps.task_id.nunique()} != {expect_tasks}")
    counts = eps.groupby(["task_id", "rep", "agent"]).size()
    if (counts != expect_eval).any():
        problems.append("some task x replicate x agent cells lack exactly 100 held-out episodes")
    for agent in ("random", "oracle", "q_learning", "ac_ctw_uct"):
        n = counts.xs(agent, level="agent").shape[0] if agent in set(eps.agent) else 0
        if n != expect_tasks * expect_reps:
            problems.append(f"{agent}: {n} cells != {expect_tasks * expect_reps}")
    seed_sets = eps.groupby(["task_id", "rep", "agent"]).seed.apply(lambda s: tuple(sorted(s)))
    for (task, rep), grp in seed_sets.groupby(level=[0, 1]):
        if grp.nunique() != 1:
            problems.append(f"{task}/{rep}: held-out seeds differ across agents")
    llm = status[status.agent == "local_llm_policy"]
    if len(llm) and not set(llm.status) <= {"evaluated", "invalid", "skipped"}:
        problems.append("unexpected LLM status values")
    manifest = json.loads((run / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    if manifest["resamples"] != expect_resamples:
        problems.append("bootstrap resamples != 10000")
    required = ("summary.csv", "bootstrap_cis.csv", "proxy_analysis.csv", "raw_traces.parquet",
                "metric_definitions.json")
    for f in required:
        if not (run / f).exists():
            problems.append(f"missing {f}")
    return problems


def write_drafts(run, tasks_path, out_dir):
    from .drafts import write_drafts as _write
    _write(run, tasks_path, out_dir)
