"""Local drafts (README, Section 4, outreach) and the arXiv upload bundle.

Every number is read from the run outputs; nothing is typed by hand. Nothing here submits,
sends, or publishes anything.
"""
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ORDER = ["random", "q_learning", "ac_ctw_uct", "local_llm_policy", "local_llm_constrained", "oracle"]
LABELS = {
    "random": "Random",
    "q_learning": "Tabular Q-learning",
    "ac_ctw_uct": "AC-CTW-UCT",
    "local_llm_policy": "Local Qwen policy table (primary)",
    "local_llm_constrained": "Local Qwen, digit-constrained (post-hoc secondary)",
    "oracle": "Exact oracle",
}
REPO_URL = "https://github.com/ajinkya-awari/aixi-complexity-benchmark"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def fmt(x, nd=3):
    return "NA" if pd.isna(x) else ("%." + str(nd) + "f") % x


def tex_esc(s):
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def facts(run):
    """Collect every reported quantity from the run directory."""
    run = Path(run)
    summary = pd.read_csv(run / "summary.csv")
    boot = pd.read_csv(run / "bootstrap_cis.csv")
    proxy = pd.read_csv(run / "proxy_analysis.csv")
    status = pd.read_csv(run / "agent_status.csv")
    suite = boot[boot.scope == "suite_hierarchical"]
    rows = []
    for a in ORDER:
        g = suite[(suite.contrast == a) & (suite.metric == "normalized_oracle_gap")].iloc[0]
        r = suite[(suite.contrast == a) & (suite.metric == "v_agent")].iloc[0]
        s = summary[summary.agent == a]
        rows.append(dict(agent=a, ret=r["mean"], gap=g["mean"], lo=g.ci_lo, hi=g.ci_hi,
                         tasks=int((s.n_replicates > 0).sum()), records=int(s.n_replicates.sum())))
    contrasts = []
    for name in ("ac_ctw_uct - q_learning", "local_llm_constrained - q_learning"):
        c = suite[(suite.contrast == name) & (suite.metric == "normalized_oracle_gap")].iloc[0]
        contrasts.append(dict(name=name, mean=c["mean"], lo=c.ci_lo, hi=c.ci_hi))
    valid_p = proxy.dropna(subset=["p_value"])
    llm = status[status.agent == "local_llm_policy"].status.value_counts().to_dict()
    con = status[status.agent == "local_llm_constrained"].status.value_counts().to_dict()
    return dict(rows=rows, contrasts=contrasts, proxy=proxy, n_tests=len(valid_p),
                min_p_holm=float(valid_p.p_holm.min()), min_p=float(valid_p.p_value.min()),
                llm=llm, con=con, summary=summary)


def _git_head(root):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def md_table(f):
    out = ["| Arm | Tasks with data | Records | Mean return | Normalized oracle gap | 95% CI |",
           "|---|---:|---:|---:|---:|---|"]
    for r in f["rows"]:
        out.append("| " + LABELS[r["agent"]] + " | " + str(r["tasks"]) + " of 20 | " + str(r["records"]) + " of 200 | "
                   + fmt(r["ret"], 2) + " | " + fmt(r["gap"]) + " | [" + fmt(r["lo"]) + ", " + fmt(r["hi"]) + "] |")
    return "\n".join(out)


COVERAGE_NOTE = (
    "The primary Qwen arm has a valid policy table in only 86 of 200 task x replicate records, covering 10 of 20 tasks; "
    "its row averages only those records, whereas the other arms average all 200. Rows are therefore not directly "
    "comparable. The digit-constrained arm was added after the primary result was known (post-hoc, secondary, "
    "exploratory). It decodes exactly one token per history key restricted to the digits 0 and 1, so all 200 tables are "
    "valid by construction. On the 86 records where the primary arm was valid, the constrained tables are identical to "
    "the primary tables (checked record by record); the two arms differ only on the 114 records where the primary output "
    "was invalid.")


def write_readme_draft(f, run, out_dir):
    llm = ", ".join(str(k) + ": " + str(v) for k, v in sorted(f["llm"].items()))
    con = ", ".join(str(k) + ": " + str(v) for k, v in sorted(f["con"].items()))
    text = (
        "# AIXI-Complexity Benchmark (local draft)\n\n"
        "Local draft generated from `" + Path(run).as_posix() + "`. Every number is copied from the run outputs.\n\n"
        "## Setup\n\n"
        "20 frozen finite-state tasks (5 families x 4 variants), 10 data replicates per task, 256 uniform random public "
        "exploration episodes and 100 held-out episodes per task x replicate. Arms: random, tabular Q-learning, exact "
        "oracle (dynamic programming), AC-CTW-UCT (MC-AIXI-inspired, not a faithful MC-AIXI-CTW), a local "
        "`Qwen/Qwen2.5-0.5B-Instruct` policy-table arm (primary), and a post-hoc secondary digit-constrained variant of "
        "that arm. Task levels are described by a separate six-dimensional proxy vector, not Kolmogorov complexity.\n\n"
        "## Suite-level results\n\n"
        "Normalized oracle gap = (V_oracle - V_agent) / max(|V_oracle - V_random|, 1e-8), unclipped; lower is closer to "
        "the oracle, and a value above 1 means below the random policy on average. Intervals: hierarchical bootstrap "
        "(10,000 resamples of tasks, then replicates), descriptive uncertainty for this fixed 20-task suite, not "
        "inference to all possible tasks.\n\n" + md_table(f) + "\n\n" + COVERAGE_NOTE + "\n\n"
        "Primary Qwen record status: " + llm + ". Constrained arm record status: " + con + ".\n\n"
        "Paired contrast (difference in gap, negative means a lower gap for the first arm): AC-CTW-UCT minus tabular "
        "Q: " + fmt(f["contrasts"][0]["mean"]) + " [" + fmt(f["contrasts"][0]["lo"]) + ", " + fmt(f["contrasts"][0]["hi"])
        + "]; the interval includes zero. Proxy associations are exploratory: across " + str(f["n_tests"])
        + " Spearman tests with finite estimates the smallest Holm-adjusted p-value is " + fmt(f["min_p_holm"])
        + ".\n\n## Files\n\n`summary.csv`, `summary_by_replicate.csv`, `bootstrap_cis.csv`, `proxy_analysis.csv`, "
        "`figures/`.\n")
    (Path(out_dir) / "README_draft.md").write_text(text, encoding="utf-8")


def suite_table_tex(f):
    lines = ["\\begin{tabular}{lrrrl}", "\\toprule",
             "Arm & Tasks & Records & Mean return & Gap [95\\% CI] \\\\", "\\midrule"]
    for r in f["rows"]:
        lines.append(tex_esc(LABELS[r["agent"]]) + " & " + str(r["tasks"]) + " & " + str(r["records"]) + " & "
                     + fmt(r["ret"], 2) + " & " + fmt(r["gap"]) + " [" + fmt(r["lo"]) + ", " + fmt(r["hi"]) + "] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def contrast_table_tex(f):
    names = {"ac_ctw_uct - q_learning": "AC-CTW-UCT $-$ tabular Q",
             "local_llm_constrained - q_learning": "Constrained Qwen (post-hoc) $-$ tabular Q"}
    lines = ["\\begin{tabular}{lr}", "\\toprule", "Contrast (gap difference) & Mean [95\\% CI] \\\\", "\\midrule"]
    for c in f["contrasts"]:
        lines.append(names[c["name"]] + " & " + fmt(c["mean"]) + " [" + fmt(c["lo"]) + ", " + fmt(c["hi"]) + "] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def proxy_table_tex(f):
    p = f["proxy"].dropna(subset=["p_value"])
    names = {"q_learning": "Tabular Q", "ac_ctw_uct": "AC-CTW-UCT", "local_llm_policy": "Qwen (primary)",
             "local_llm_constrained": "Qwen (post-hoc)"}
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule", "Arm & Proxy dimension & $n$ & $\\rho$ & $p$ & $p_{\\mathrm{Holm}}$ \\\\",
             "\\midrule"]
    for _, r in p.iterrows():
        lines.append(names[r.agent] + " & " + tex_esc(r.proxy_dimension) + " & " + str(int(r.n_tasks)) + " & "
                     + fmt(r.spearman_rho) + " & " + fmt(r.p_value) + " & " + fmt(r.p_holm) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def corpus_table_tex(tasks_path):
    corpus = yaml.safe_load(Path(tasks_path).read_text(encoding="utf-8"))
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "Task & $m$ & Latent & Obs. & $H$ & Keys & Branch. & Rew. dens. \\\\", "\\midrule"]
    for t in corpus["tasks"]:
        pv = t["proxy_vector"]
        lines.append(tex_esc(t["task_id"]) + " & " + str(t["public_history_order_m"]) + " & "
                     + str(t["num_latent_states"]) + " & " + str(t["num_public_observations"]) + " & "
                     + str(t["horizon"]) + " & " + str(pv["reachable_history_state_count"]) + " & "
                     + fmt(pv["branching_factor"], 2) + " & " + fmt(pv["reward_density_uniform_policy"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_section4(f, out_dir):
    tex = (
        "% Local draft of Section 4 (Results); numbers are read from the run outputs.\n"
        "\\section{Results}\n"
        "Table~\\ref{tab:suite} reports suite-level means with 95\\% hierarchical bootstrap intervals, for this fixed "
        "20-task suite only. The primary Qwen arm has a valid policy table in 86 of 200 records (10 of 20 tasks) and its row "
        "averages only those records; rows are therefore not directly comparable. The digit-constrained arm is a "
        "post-hoc, secondary, exploratory analysis added after the primary result was known.\n"
        "\\begin{table}[h]\n\\centering\\small\n\\input{tables/suite_results}\n"
        "\\caption{Suite-level normalized oracle gap (lower is closer to the oracle; above 1 is worse than random on average).}"
        "\n\\label{tab:suite}\n\\end{table}\n")
    (Path(out_dir) / "section4_draft").mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "section4_draft" / "aixi_complexity_section4.tex").write_text(tex, encoding="utf-8")


def write_outreach(f, out_dir):
    q = next(r for r in f["rows"] if r["agent"] == "q_learning")
    a = next(r for r in f["rows"] if r["agent"] == "ac_ctw_uct")
    text = (
        "Subject: Finite-state benchmark of AIXI-style planning and simple learners (draft)\n\n"
        "Hello,\n\n"
        "I built a small local benchmark that compares a tabular Q-learner, an AC-CTW-UCT agent (inspired by "
        "MC-AIXI-CTW, not a faithful reproduction), an exact dynamic-programming oracle, a random policy and a local "
        "0.5B policy-table model on 20 frozen finite-state tasks. Task difficulty is described by a separate proxy "
        "vector, not by Kolmogorov complexity.\n\n"
        "Suite-level normalized oracle gaps (lower is closer to the oracle; 95% hierarchical bootstrap intervals, "
        "descriptive for this fixed suite): tabular Q " + fmt(q["gap"]) + " [" + fmt(q["lo"]) + ", " + fmt(q["hi"])
        + "], AC-CTW-UCT " + fmt(a["gap"]) + " [" + fmt(a["lo"]) + ", " + fmt(a["hi"]) + "]; their paired difference "
        "interval includes zero. The small local language model produced a valid policy table in only 86 of 200 "
        "records, so I do not draw conclusions about it.\n\n"
        "Code and results: " + REPO_URL + "\n\n"
        "I would value your view on whether the task families and the proxy vector are a sensible way to organize this "
        "kind of comparison.\n\nBest regards,\nAjinkya\n\n"
        "Draft only. Recipient and contact details intentionally left unset.\n")
    (Path(out_dir) / "outreach").mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "outreach" / "aixi_labs_draft.md").write_text(text, encoding="utf-8")


def write_drafts(run, tasks_path, out_dir):
    f = facts(run)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_readme_draft(f, run, out_dir)
    write_section4(f, out_dir)
    (out_dir / "section4_draft" / "tables").mkdir(exist_ok=True)
    (out_dir / "section4_draft" / "tables" / "suite_results.tex").write_text(suite_table_tex(f), encoding="utf-8")
    write_outreach(f, out_dir)
    print("wrote drafts to", out_dir)


BIB = r"""@article{willems1995ctw,
  author = {Willems, Frans M. J. and Shtarkov, Yuri M. and Tjalkens, Tjalling J.},
  title = {The context-tree weighting method: basic properties},
  journal = {IEEE Transactions on Information Theory},
  volume = {41}, number = {3}, pages = {653--664}, year = {1995}
}
@article{veness2011mcaixi,
  author = {Veness, Joel and Ng, Kee Siong and Hutter, Marcus and Uther, William and Silver, David},
  title = {A {Monte-Carlo} {AIXI} approximation},
  journal = {Journal of Artificial Intelligence Research},
  volume = {40}, pages = {95--142}, year = {2011}
}
@book{hutter2005uai,
  author = {Hutter, Marcus},
  title = {Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability},
  publisher = {Springer}, year = {2005}
}
@article{krichevsky1981kt,
  author = {Krichevsky, Raphail E. and Trofimov, Victor K.},
  title = {The performance of universal encoding},
  journal = {IEEE Transactions on Information Theory},
  volume = {27}, number = {2}, pages = {199--207}, year = {1981}
}
@inproceedings{kocsis2006uct,
  author = {Kocsis, Levente and Szepesv{\'a}ri, Csaba},
  title = {Bandit based {Monte-Carlo} planning},
  booktitle = {European Conference on Machine Learning (ECML)},
  series = {Lecture Notes in Computer Science}, volume = {4212}, pages = {282--293}, year = {2006}
}
@article{watkins1992q,
  author = {Watkins, Christopher J. C. H. and Dayan, Peter},
  title = {Q-learning},
  journal = {Machine Learning},
  volume = {8}, pages = {279--292}, year = {1992}
}
@article{towers2024gymnasium,
  author = {Towers, Mark and others},
  title = {Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  journal = {arXiv preprint arXiv:2407.17032}, year = {2024}
}
@article{qwen2024qwen25,
  author = {{Qwen Team}},
  title = {Qwen2.5 Technical Report},
  journal = {arXiv preprint arXiv:2412.15115}, year = {2024}
}
@article{holm1979,
  author = {Holm, Sture},
  title = {A simple sequentially rejective multiple test procedure},
  journal = {Scandinavian Journal of Statistics},
  volume = {6}, number = {2}, pages = {65--70}, year = {1979}
}
@article{efron1979bootstrap,
  author = {Efron, Bradley},
  title = {Bootstrap methods: another look at the jackknife},
  journal = {The Annals of Statistics},
  volume = {7}, number = {1}, pages = {1--26}, year = {1979}
}
"""

BBL = r"""\begin{thebibliography}{10}
\bibitem{willems1995ctw} F.~M.~J. Willems, Y.~M. Shtarkov, and T.~J. Tjalkens. The context-tree weighting method: basic properties. \emph{IEEE Trans. Inf. Theory}, 41(3):653--664, 1995.
\bibitem{veness2011mcaixi} J.~Veness, K.~S. Ng, M.~Hutter, W.~Uther, and D.~Silver. A Monte-Carlo AIXI approximation. \emph{J. Artif. Intell. Res.}, 40:95--142, 2011.
\bibitem{hutter2005uai} M.~Hutter. \emph{Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability}. Springer, 2005.
\bibitem{krichevsky1981kt} R.~E. Krichevsky and V.~K. Trofimov. The performance of universal encoding. \emph{IEEE Trans. Inf. Theory}, 27(2):199--207, 1981.
\bibitem{kocsis2006uct} L.~Kocsis and C.~Szepesv{\'a}ri. Bandit based Monte-Carlo planning. In \emph{ECML}, LNCS 4212, pages 282--293, 2006.
\bibitem{watkins1992q} C.~J. C.~H. Watkins and P.~Dayan. Q-learning. \emph{Machine Learning}, 8:279--292, 1992.
\bibitem{towers2024gymnasium} M.~Towers et~al. Gymnasium: A standard interface for reinforcement learning environments. arXiv:2407.17032, 2024.
\bibitem{qwen2024qwen25} Qwen Team. Qwen2.5 technical report. arXiv:2412.15115, 2024.
\bibitem{holm1979} S.~Holm. A simple sequentially rejective multiple test procedure. \emph{Scand. J. Statist.}, 6(2):65--70, 1979.
\bibitem{efron1979bootstrap} B.~Efron. Bootstrap methods: another look at the jackknife. \emph{Ann. Statist.}, 7(1):1--26, 1979.
\end{thebibliography}
"""

MAIN_TEX = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}

\title{AIXI-Complexity: A Local Benchmark of Decision-Making Agents on Frozen Finite-State Tasks with Declared Proxy Vectors}
\author{Ajinkya Awari}
\date{}

\begin{document}
\maketitle

\begin{abstract}
We describe a small, fully local benchmark that compares decision-making agents on 20 frozen finite-state controlled tasks
(5 families $\times$ 4 variants). Task difficulty is summarized by a separate six-dimensional proxy vector rather than by any
claim about Kolmogorov complexity. We compare a random policy, tabular Q-learning, an AC-CTW-UCT planner inspired by MC-AIXI-CTW,
a local 0.5B-parameter language model that emits a policy table, and an exact dynamic-programming oracle, using 10 data
replicates per task and identical held-out seeds. The primary metric is the normalized oracle gap. At suite level, tabular
Q-learning has a mean gap of @@Q_GAP@@ and AC-CTW-UCT of @@A_GAP@@ (95\% hierarchical bootstrap intervals @@Q_CI@@ and @@A_CI@@);
the interval of their paired difference includes zero. The language-model arm produced a valid policy table in only 86 of 200
records, so no conclusion is drawn about it; a post-hoc, exploratory variant with constrained decoding is reported separately.
Exploratory proxy associations are not significant after Holm correction. The benchmark is small, synthetic and fixed, and
makes no claim about AIXI, AGI or universal intelligence.
\end{abstract}

\section{Introduction}
Universal-agent models such as AIXI~\cite{hutter2005uai} are incomputable, and practical approximations such as
MC-AIXI-CTW~\cite{veness2011mcaixi} are usually evaluated on a handful of games. We build a small benchmark where every
environment is a frozen finite-state task with a public history key that is, by construction, sufficient for the latent
state, so that an exact oracle exists and agents can be compared on a common, normalized scale. The contribution is the
benchmark and its evaluation protocol, not a claim of state-of-the-art performance.

\section{Benchmark}
\paragraph{Tasks.} The corpus has five families (reactive, contextual, delayed, branching, sparse sequence), each with four
variants V1--V4 generated deterministically from one frozen seed and hashed with SHA-256. Every task has two actions, 2--8 latent
states, 2--4 public observations, rewards in $\{0,1\}$ and a horizon in $\{8,12,16,24,32\}$ (Table~\ref{tab:corpus}). Environments
are custom Gymnasium~\cite{towers2024gymnasium} environments and pass the library environment checker.

\paragraph{Public history key.} A key is the last $m$ action--observation pairs plus the current observation, with $m\le 3$;
rewards and time are not part of the key. For each task we verify exhaustively that every reachable key identifies exactly one
latent state, so each task is Markov in its key. The generator records the minimal sufficient $m$.

\paragraph{Proxy vector.} Difficulty is described by six separate quantities: reachable history-state count, order $m$, horizon,
branching factor (mean number of distinct next percepts per reachable key--action pair), reward density under the uniform policy,
and canonical task-specification byte length. The proxies are covariates, not a scalar score and not Kolmogorov complexity.

\paragraph{Data and evaluation.} Each task has 10 independent data replicates; each replicate has 256 uniform-random public
exploration episodes and 100 held-out episodes defined by frozen seeds. All arms are evaluated on the same held-out seeds, and no
learner is updated during evaluation.

\section{Agents}
\paragraph{Random and oracle.} The random policy is uniform. The oracle performs exact finite-horizon dynamic programming using the
hidden task tables inside the evaluator only; because keys identify latent states this equals dynamic programming over the public
key and time step.

\paragraph{Tabular Q-learning~\cite{watkins1992q}.} Offline on the public corpus with $\alpha=0.1$, $\gamma=0.99$, 20 deterministic
shuffled passes and lowest-action tie breaking. Keys never seen in the corpus use a declared default action 0, as for the language-model arm.

\paragraph{AC-CTW-UCT.} An action-conditional context tree weighting~\cite{willems1995ctw} model with KT estimators~\cite{krichevsky1981kt}
at every node predicts the percept (next observation, reward). Each interaction cycle is encoded as action bit (side information), a
fixed-width binary observation and one reward bit; only percept bits are predicted. The model is fitted offline on the public corpus and then
frozen. Planning uses UCT~\cite{kocsis2006uct} with 512 simulations per decision, planning horizon $\min(16,\text{remaining steps})$,
exploration constant $\sqrt{2}$ and seeded tie breaking; rollouts never mutate the model. This is inspired by MC-AIXI-CTW but is not a
faithful reproduction of it.

\paragraph{Local language-model arm.} \texttt{Qwen/Qwen2.5-0.5B-Instruct}~\cite{qwen2024qwen25} at revision
\texttt{@@REV@@}, run locally on CPU with greedy decoding. The prompt lists, for each public key observed in the exploration corpus,
the empirical action counts and mean rewards; the model must reply with one digit per key. The reply is valid only if it is exactly
that many digits in $\{0,1\}$; invalid replies are recorded as invalid and never replaced. The cap on new tokens is the number of keys plus 8.
\emph{Post-hoc secondary arm:} after seeing the primary result we added a variant that restricts decoding to the tokens 0 and 1 for exactly one
token per key, which makes every table valid by construction. On the 86 records where the primary reply was valid the two arms produce identical tables.

\section{Metrics and statistics}
The primary metric is the normalized oracle gap $(V_{\mathrm{oracle}}-V_{\mathrm{agent}})/\max(|V_{\mathrm{oracle}}-V_{\mathrm{random}}|,10^{-8})$,
reported unclipped, where each $V$ is the paired mean return over the same 100 held-out seeds of one task and replicate; if the denominator is below
$10^{-8}$ the gap is not available. A value above 1 means below the random policy on average. Per-task intervals resample replicate blocks within a task
10,000 times~\cite{efron1979bootstrap}; suite-level intervals use a hierarchical bootstrap (tasks, then replicates), and describe uncertainty for this
fixed suite, not for all possible tasks. Proxy associations are exploratory Spearman correlations with Holm correction~\cite{holm1979}.

\section{Results}
Table~\ref{tab:suite} gives suite-level results. The primary Qwen arm has a valid policy table in 86 of 200 records covering 10 of 20 tasks, and its row
averages only those records, while other rows use all 200; rows are therefore not directly comparable. Table~\ref{tab:contrast} gives paired contrasts; the
interval for AC-CTW-UCT minus tabular Q includes zero. Per-task gaps are shown in Figure~\ref{fig:gap}. Across the @@NTESTS@@ Spearman tests with finite
estimates, the smallest Holm-adjusted $p$-value is @@MINP@@ (Table~\ref{tab:proxy}); we draw no conclusion about monotonic relations.

\begin{table}[h]\centering\small
\input{tables/suite_results}
\caption{Suite-level normalized oracle gap (lower is closer to the oracle). ``Tasks'' and ``Records'' count the tasks and task$\times$replicate records with data.}
\label{tab:suite}
\end{table}

\begin{table}[h]\centering\small
\input{tables/paired_contrasts}
\caption{Paired suite-level contrasts of the normalized oracle gap with 95\% hierarchical bootstrap intervals (negative favours the first arm).}
\label{tab:contrast}
\end{table}

\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{figures/gap_by_task.png}
\caption{Mean normalized oracle gap per task, averaged over 10 data replicates.}
\label{fig:gap}
\end{figure}

\begin{table}[h]\centering\scriptsize
\input{tables/proxy_summary}
\caption{Exploratory Spearman associations between proxy dimensions and task-level gap, Holm-corrected over all tests with finite estimates.}
\label{tab:proxy}
\end{table}

\section{Limitations}
The suite is small, synthetic and fixed; results do not generalize to other task distributions. The primary language-model arm is invalid on 114 of 200 records,
and the secondary arm was designed after the primary result was seen. Proxy associations are exploratory and underpowered with 20 tasks. AC-CTW-UCT uses a
frozen model fitted on 256 episodes and is not a faithful MC-AIXI-CTW. Python 3.12.13 was used because 3.12.10 was unavailable. Nothing here bears on AIXI, AGI or
universal intelligence.

\section{Reproducibility}
Code, frozen tasks, manifests and results: \url{@@REPO@@} (commit @@COMMIT@@ at bundle creation). All computation is local and on CPU.

\appendix
\section{Task corpus}
\begin{table}[h]\centering\scriptsize
\input{tables/task_corpus}
\caption{Frozen tasks with selected proxy dimensions ($m$: public history order, $H$: horizon, Keys: reachable history keys).}
\label{tab:corpus}
\end{table}

\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""


def bundle_readme(f):
    return (
        "# arXiv upload bundle (local, not submitted)\n\n"
        "Status: LOCAL DRAFT. Nothing in this folder has been submitted to arXiv or sent anywhere. LaTeX was NOT compiled in the environment "
        "that produced it (no TeX installation); compile it yourself (for example on Overleaf) and fix any errors before uploading.\n\n"
        "## Files\n\n"
        "| File | Purpose |\n|---|---|\n"
        "| `main.tex` | Paper source (article class, 11pt). Numbers in the abstract and Results text are filled from `results/local_run`. |\n"
        "| `main.bbl` | Hand-written compiled bibliography so arXiv does not need to run BibTeX; keep it in sync with `references.bib`. |\n"
        "| `references.bib` | Bibliography source (10 entries). Verify each entry against the original publication before submission. |\n"
        "| `tables/suite_results.tex` | Suite-level results table (generated from `bootstrap_cis.csv` and `summary.csv`). |\n"
        "| `tables/paired_contrasts.tex` | Paired suite-level contrasts (generated). |\n"
        "| `tables/proxy_summary.tex` | Exploratory Holm-corrected Spearman table (generated from `proxy_analysis.csv`). |\n"
        "| `tables/task_corpus.tex` | The 20 frozen tasks with proxy dimensions (generated from the frozen YAML). |\n"
        "| `figures/gap_by_task.png` | Per-task gap figure (copied from `results/local_run/figures`). |\n"
        "| `MANIFEST.sha256` | SHA-256 of every file in the bundle. |\n\n"
        "## Before you upload\n\n"
        "1. Compile `main.tex` twice (pdfLaTeX) and read the PDF.\n"
        "2. Verify the bibliography entries (authors, volumes, pages, arXiv ids) against the sources; they were written from memory of the standard references.\n"
        "3. Add your affiliation and contact line if you want them; the author block currently has only the name.\n"
        "4. Choose arXiv categories and a license yourself (cs.AI or cs.LG are plausible, not decided). First-time submitters may need an endorsement.\n"
        "5. The paper is a benchmark description with limited evidence: the primary language-model arm was invalid on 114 of 200 records, and the secondary arm is post-hoc.\n\n"
        "A zip of this folder, `../aixi_complexity_arxiv_bundle.zip`, is ready for upload once you are satisfied.\n")


def overlap_identical(llm_dir, constrained_dir):
    """True if the constrained tables equal the primary tables on every record where the primary is valid."""
    n = 0
    for p in Path(llm_dir).glob("*/rep_*.json"):
        a = json.loads(p.read_text(encoding="utf-8"))
        if a.get("status") != "valid":
            continue
        b = json.loads((Path(constrained_dir) / p.parent.name / p.name).read_text(encoding="utf-8"))
        if a["actions"] != b["actions"]:
            return False
        n += 1
    return n > 0


def check_text_claims(f, llm_dir, constrained_dir):
    """The fixed prose cites counts and statements; refuse to build if the data no longer match."""
    prim = next(r for r in f["rows"] if r["agent"] == "local_llm_policy")
    sec = next(r for r in f["rows"] if r["agent"] == "local_llm_constrained")
    ok = (prim["records"] == 86 and prim["tasks"] == 10 and sec["records"] == 200 and f["min_p_holm"] >= 0.05
          and f["contrasts"][0]["lo"] < 0 < f["contrasts"][0]["hi"] and overlap_identical(llm_dir, constrained_dir))
    if not ok:
        raise RuntimeError("results changed: update the fixed prose in drafts.py before building the bundle")


def write_arxiv_bundle(run, tasks_path, out_dir, repo_root=".", llm_dir="artifacts/models/local_llm_policy",
                       constrained_dir="artifacts/models/local_llm_constrained"):
    run, out_dir = Path(run), Path(out_dir)
    f = facts(run)
    check_text_claims(f, llm_dir, constrained_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "tables").mkdir(parents=True)
    (out_dir / "figures").mkdir()
    (out_dir / "tables" / "suite_results.tex").write_text(suite_table_tex(f), encoding="utf-8")
    (out_dir / "tables" / "paired_contrasts.tex").write_text(contrast_table_tex(f), encoding="utf-8")
    (out_dir / "tables" / "proxy_summary.tex").write_text(proxy_table_tex(f), encoding="utf-8")
    (out_dir / "tables" / "task_corpus.tex").write_text(corpus_table_tex(tasks_path), encoding="utf-8")
    shutil.copy(run / "figures" / "gap_by_task.png", out_dir / "figures" / "gap_by_task.png")
    q = next(r for r in f["rows"] if r["agent"] == "q_learning")
    a = next(r for r in f["rows"] if r["agent"] == "ac_ctw_uct")
    tex = (MAIN_TEX.replace("@@Q_GAP@@", fmt(q["gap"])).replace("@@A_GAP@@", fmt(a["gap"]))
           .replace("@@Q_CI@@", "[" + fmt(q["lo"]) + ", " + fmt(q["hi"]) + "]")
           .replace("@@A_CI@@", "[" + fmt(a["lo"]) + ", " + fmt(a["hi"]) + "]")
           .replace("@@REV@@", MODEL_REVISION).replace("@@NTESTS@@", str(f["n_tests"]))
           .replace("@@MINP@@", fmt(f["min_p_holm"])).replace("@@REPO@@", REPO_URL)
           .replace("@@COMMIT@@", _git_head(repo_root)))
    (out_dir / "main.tex").write_text(tex, encoding="utf-8")
    (out_dir / "references.bib").write_text(BIB, encoding="utf-8")
    (out_dir / "main.bbl").write_text(BBL, encoding="utf-8")
    (out_dir / "README.md").write_text(bundle_readme(f), encoding="utf-8")
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    manifest = "\n".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(out_dir).as_posix()
                         for p in files if p.name != "MANIFEST.sha256")
    (out_dir / "MANIFEST.sha256").write_text(manifest + "\n", encoding="utf-8")
    zpath = out_dir.parent / "aixi_complexity_arxiv_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out_dir).as_posix())
    print("wrote bundle", out_dir, "and", zpath)
    return json.dumps({"files": len(files) + 1})
