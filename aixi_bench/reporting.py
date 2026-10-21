"""Plots, local-only drafts, and run verification. Every number comes from real run outputs."""
import json
from pathlib import Path

import pandas as pd
import yaml

AGENTS = ["random", "q_learning", "ac_ctw_uct", "local_llm_policy", "oracle"]
LABELS = {"random": "Random", "q_learning": "Tabular Q", "ac_ctw_uct": "AC-CTW-UCT",
          "local_llm_policy": "Local Qwen policy table", "oracle": "Exact oracle"}


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


def write_drafts(run, tasks_path, out_dir):
    run, out_dir = Path(run), Path(out_dir)
    (out_dir / "section4_draft").mkdir(parents=True, exist_ok=True)
    (out_dir / "outreach").mkdir(parents=True, exist_ok=True)
    status = pd.read_csv(run / "agent_status.csv")
    gap, ret = _suite_table(run)
    llm = status[status.agent == "local_llm_policy"]
    llm_counts = llm.status.value_counts().to_dict()
    rows = []
    for a in AGENTS:
        if a in gap.index:
            g = gap.loc[a]
            rows.append((a, _fmt(ret.loc[a, "mean"], 2), _fmt(g["mean"]), _fmt(g.ci_lo), _fmt(g.ci_hi)))
    md_table = "| Arm | Mean return | Normalized oracle gap | 95% CI low | 95% CI high |\n|---|---|---|---|---|\n"
    md_table += "\n".join(f"| {LABELS[a]} | {r} | {g} | {lo} | {hi} |" for a, r, g, lo, hi in rows)
    llm_line = ", ".join(f"{k}: {v}" for k, v in sorted(llm_counts.items())) or "no records"

    readme = f"""# AIXI-Complexity Benchmark (local draft)

Local draft generated from `{run.as_posix()}`. Every number below is copied from the run outputs.

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

{md_table}

Local Qwen policy-table status over task x replicate records ({len(llm)} total): {llm_line}.
Invalid outputs are excluded from value-based metrics and are never replaced by other behavior.

## Files

`summary.csv`, `summary_by_replicate.csv`, `bootstrap_cis.csv`, `proxy_analysis.csv` (exploratory, Holm-corrected),
`figures/`. See the evidence manifests for hashes.
"""
    (out_dir / "README_draft.md").write_text(readme, encoding="utf-8")

    tex_rows = "\n".join(f"{LABELS[a]} & {r} & {g} & [{lo}, {hi}] \\\\" for a, r, g, lo, hi in rows)
    tex = r"""% Local draft of Section 4 (Results). Numbers are copied from the run outputs.
\section{Results}
We evaluate five arms on 20 frozen finite-state tasks with 10 data replicates each (256 exploration and 100 held-out
episodes per task and replicate). The primary metric is the normalized oracle gap
$(V_{oracle}-V_{agent})/\max(|V_{oracle}-V_{random}|,10^{-8})$, reported unclipped.
Table~\ref{tab:suite} gives suite-level means with 95\% hierarchical bootstrap intervals; these describe uncertainty
for this fixed task suite only. Task levels are summarized by a six-dimensional proxy vector and are not Kolmogorov
complexity; proxy associations are exploratory and Holm-corrected.
\begin{table}[h]
\centering
\begin{tabular}{lrrr}
Arm & Mean return & Normalized gap & 95\% CI \\
\hline
""" + tex_rows + r"""
\end{tabular}
\caption{Suite-level results.}
\label{tab:suite}
\end{table}
"""
    (out_dir / "section4_draft" / "aixi_complexity_section4.tex").write_text(tex, encoding="utf-8")

    outreach = """Subject: Finite-state benchmark of AIXI-style planning and simple learners (draft)

Hello,

I built a small local benchmark that compares a tabular Q-learner, an AC-CTW-UCT agent (inspired by MC-AIXI-CTW, not a
faithful reproduction), an exact dynamic-programming oracle, a random policy and a local 0.5B policy-table model on 20
frozen finite-state tasks. Task difficulty is described by a separate proxy vector, not by Kolmogorov complexity.

Suite-level results with intervals are in `README_draft.md` of the project folder. I would value your view on whether
the task families and the proxy vector are a sensible way to organize this kind of comparison.

Best regards,
Ajinkya

Draft only. Recipient and contact details intentionally left unset.
"""
    (out_dir / "outreach" / "aixi_labs_draft.md").write_text(outreach, encoding="utf-8")
    print("wrote drafts to", out_dir)


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
