"""aixi-bench command line interface."""
import argparse
import json
import sys
from pathlib import Path

import yaml

from . import __version__


def _load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")


def cmd_tasks(args):
    from .generator import generate_corpus
    from .hashing import canonical_json, sha256_hex
    from .tasks import load_corpus, validate_corpus
    if args.action == "generate":
        corpus = generate_corpus(args.generator_seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(corpus, sort_keys=False, width=120), encoding="utf-8")
        print("wrote", args.out, "tasks:", len(corpus["tasks"]))
        return 0
    corpus = _load_yaml(args.tasks)
    if args.action == "validate":
        problems = validate_corpus(corpus)
        load_corpus(corpus)  # also proves public-key sufficiency for every task
        for p in problems:
            print("PROBLEM:", p)
        print("tasks:", len(corpus["tasks"]), "problems:", len(problems))
        return 1 if problems else 0
    if args.action == "hash":
        out = {t["task_id"]: t["sha256"] for t in corpus["tasks"]}
        out["_corpus_sha256"] = sha256_hex(canonical_json(corpus).encode("utf-8"))
        _write_json(args.out, out)
        print("wrote", args.out)
        return 0


def cmd_data(args):
    from .data import freeze_data, verify_data
    from .tasks import load_corpus
    tasks = load_corpus(args.tasks)
    if args.action == "freeze":
        hashes = freeze_data(tasks, args.out)
        print(json.dumps(hashes, indent=1))
        return 0
    problems = verify_data(args.data, tasks)
    for p in problems:
        print("PROBLEM:", p)
    print("data problems:", len(problems))
    return 1 if problems else 0


def cmd_env(args):
    from gymnasium.utils.env_checker import check_env

    from .env import FiniteTaskEnv
    from .tasks import load_corpus
    results = {}
    for task in load_corpus(args.tasks):
        check_env(FiniteTaskEnv(task), skip_render_check=True)
        results[task.task_id] = "pass"
    if args.out:
        _write_json(args.out, results)
    print("check_env passed for", len(results), "tasks")
    return 0


def cmd_model_gate(args):
    from .modelgate import run_gate
    gate = run_gate(args.out)
    print(json.dumps(gate, indent=1))
    return 0 if gate["status"] == "pass" else 2


def cmd_llm_policy(args):
    from .data import corpus_path, load_corpus_file
    from .llmpolicy import QwenPolicyGenerator, generate_policy_record
    from .tasks import load_corpus
    gate = json.loads(Path(args.model_gate).read_text(encoding="utf-8"))
    tok = json.loads((Path(args.model_gate).parent / "qwen_tokenizer_manifest.json").read_text(encoding="utf-8"))
    tasks = load_corpus(args.tasks)
    out_dir = Path(args.out)
    if gate["status"] != "pass":
        print("model gate not passed: LLM arm skipped (no substitution)")
        for t in tasks:
            for rep in range(10):
                record = {"status": "skipped", "reason": gate["reason"]}
                _write_json(out_dir / t.task_id / f"rep_{rep:02d}.json", record)
        return 0
    gen = QwenPolicyGenerator(gate)
    n = 0
    jobs = [(t, rep) for t in tasks for rep in range(10)]
    if args.reverse:
        jobs.reverse()
    for t, rep in jobs:
        path = out_dir / t.task_id / f"rep_{rep:02d}.json"
        if path.exists():
            continue
        corpus = load_corpus_file(corpus_path(args.data, t.task_id, rep))
        _write_json(path, generate_policy_record(gen, corpus, t.m, tok["tokenizer_files"],
                                                 constrained=args.constrained))
        n += 1
        print("generated", t.task_id, rep, flush=True)
    return 0


def cmd_run(args):
    from .evalrun import run_all
    corpus = _load_yaml(args.tasks)
    ids = args.task_ids.split(",") if args.task_ids else None
    reps = args.replicates
    agents = args.agents.split(",")
    run_all(corpus, args.data, args.llm_dir, args.out, constrained_dir=args.constrained_dir, replicates=reps,
            agents_enabled=agents, processes=args.processes, task_ids=ids)
    return 0


def _load_run(run_dir):
    from .evalrun import collect
    return collect(run_dir)


def cmd_metrics(args):
    from .metrics import replicate_table, summary_by_task
    eps, steps, status = _load_run(args.run)
    rep = replicate_table(eps, steps)
    out = Path(args.run)
    rep.to_csv(out / "summary_by_replicate.csv", index=False)
    summary = summary_by_task(rep, status, None)
    summary.to_csv(out / "summary.csv", index=False)
    status.to_csv(out / "agent_status.csv", index=False)
    steps.to_parquet(out / "raw_traces.parquet", index=False)
    eps.to_parquet(out / "episodes.parquet", index=False)
    _write_json(out / "metric_definitions.json", {
        "normalized_oracle_gap": "(V_oracle - V_agent) / max(abs(V_oracle - V_random), 1e-8)",
        "reported_unclipped": True, "na_rule": "NA when abs(V_oracle - V_random) < 1e-8; raw return retained",
        "V_definition": "paired empirical mean return over the same 100 held-out seeds of one task x replicate"})
    print("wrote summary.csv rows:", len(summary))
    return 0


def cmd_bootstrap(args):
    import pandas as pd

    from .stats import bootstrap_table
    rep = pd.read_csv(Path(args.run) / "summary_by_replicate.csv")
    table = bootstrap_table(rep)
    table.to_csv(Path(args.run) / "bootstrap_cis.csv", index=False)
    _write_json(Path(args.run) / "bootstrap_manifest.json", {
        "resamples": 10000, "seed": 20261003, "ci": 0.95, "paired_block": "replicate within task",
        "hierarchical": "task then replicate",
        "aggregate_label": ("suite-level descriptive uncertainty for the fixed 20-task suite; "
                            "not inference to all possible tasks")})
    print("wrote bootstrap_cis.csv rows:", len(table))
    return 0


def cmd_proxy(args):
    import pandas as pd

    from .stats import proxy_associations
    summary = pd.read_csv(Path(args.run) / "summary.csv")
    out = proxy_associations(summary, _load_yaml(args.tasks))
    out.to_csv(Path(args.run) / "proxy_analysis.csv", index=False)
    _write_json(Path(args.run) / "proxy_analysis_manifest.json", {
        "method": "spearman", "correction": "holm", "status": "exploratory", "tests": int(out.p_value.notna().sum())})
    print("wrote proxy_analysis.csv rows:", len(out))
    return 0


def cmd_plots(args):
    from .reporting import make_plots
    make_plots(args.run, args.tasks)
    return 0


def cmd_drafts(args):
    from .reporting import write_drafts
    write_drafts(args.run, args.tasks, args.out)
    return 0


def cmd_arxiv_bundle(args):
    from .drafts import write_arxiv_bundle
    write_arxiv_bundle(args.run, args.tasks, args.out)
    return 0


def cmd_verify_run(args):
    from .reporting import verify_run
    problems = verify_run(args.run)
    for p in problems:
        print("PROBLEM:", p)
    print("verify-run problems:", len(problems))
    return 1 if problems else 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="aixi-bench",
        description="AIXI-Complexity benchmark: finite-state controlled tasks, local-only agents "
                    "(random, tabular Q, exact oracle, AC-CTW-UCT, local Qwen policy table).")
    p.add_argument("--version", action="version", version=f"aixi-bench {__version__}")
    sub = p.add_subparsers(dest="command")

    t = sub.add_parser("tasks", help="generate / validate / hash the frozen 20-task corpus")
    t.add_argument("action", choices=["generate", "validate", "hash"])
    t.add_argument("--generator-seed", type=int, default=20261003)
    t.add_argument("--tasks", default="task_specs/frozen_finite_tasks.yaml")
    t.add_argument("--out")
    t.set_defaults(fn=cmd_tasks)

    d = sub.add_parser("data", help="freeze / verify public exploration corpora and seed manifests")
    d.add_argument("action", choices=["freeze", "verify"])
    d.add_argument("--tasks", default="task_specs/frozen_finite_tasks.yaml")
    d.add_argument("--out", default="artifacts/data")
    d.add_argument("--data", default="artifacts/data")
    d.set_defaults(fn=cmd_data)

    e = sub.add_parser("env-check", help="run gymnasium check_env on every task")
    e.add_argument("--tasks", default="task_specs/frozen_finite_tasks.yaml")
    e.add_argument("--out")
    e.set_defaults(fn=cmd_env)

    g = sub.add_parser("model-gate", help="verify the pinned local Qwen model (license, cache, SHA-256)")
    g.add_argument("--out", default="artifacts/gates")
    g.set_defaults(fn=cmd_model_gate)

    lp = sub.add_parser("llm-policy", help="generate local LLM policy tables (greedy decoding)")
    lp.add_argument("--tasks", default="task_specs/frozen_finite_tasks.yaml")
    lp.add_argument("--data", default="artifacts/data")
    lp.add_argument("--model-gate", default="artifacts/gates/qwen_model_gate.json")
    lp.add_argument("--out", default="artifacts/models/local_llm_policy")
    lp.add_argument("--constrained", action="store_true",
                    help="secondary arm: decoding restricted to the digit tokens 0 and 1")
    lp.add_argument("--reverse", action="store_true", help="process task x replicate records in reverse order")
    lp.set_defaults(fn=cmd_llm_policy)

    r = sub.add_parser("run", help="update-free paired evaluation of all arms")
    r.add_argument("--tasks", default="task_specs/frozen_finite_tasks.yaml")
    r.add_argument("--data", default="artifacts/data")
    r.add_argument("--llm-dir", default="artifacts/models/local_llm_policy")
    r.add_argument("--out", default="results/local_run")
    r.add_argument("--replicates", type=int, default=10)
    r.add_argument("--processes", type=int, default=3)
    r.add_argument("--constrained-dir", default="artifacts/models/local_llm_constrained")
    r.add_argument("--task-ids")
    r.add_argument("--agents", default="random,oracle,q_learning,ac_ctw_uct")
    r.set_defaults(fn=cmd_run)

    for name, fn, helptext in [("metrics", cmd_metrics, "compute summary.csv and metric definitions"),
                               ("bootstrap", cmd_bootstrap, "10,000-resample paired and hierarchical bootstrap"),
                               ("proxy-analysis", cmd_proxy, "exploratory Holm-corrected Spearman proxy analysis"),
                               ("plots", cmd_plots, "local figures"),
                               ("verify-run", cmd_verify_run, "verify run completeness and contract")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--run", default="results/local_run")
        s.add_argument("--tasks", default="task_specs/frozen_finite_tasks.yaml")
        s.set_defaults(fn=fn)

    dr = sub.add_parser("drafts", help="local README / arXiv Section 4 / outreach drafts (never sent or published)")
    dr.add_argument("--run", default="results/local_run")
    dr.add_argument("--tasks", default="task_specs/frozen_finite_tasks.yaml")
    dr.add_argument("--out", default="draft_outputs")
    dr.set_defaults(fn=cmd_drafts)
    ab = sub.add_parser("arxiv-bundle", help="local arXiv upload bundle (never submitted)")
    ab.add_argument("--run", default="results/local_run")
    ab.add_argument("--tasks", default="task_specs/frozen_finite_tasks.yaml")
    ab.add_argument("--out", default="draft_outputs/arxiv_bundle")
    ab.set_defaults(fn=cmd_arxiv_bundle)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 0
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
