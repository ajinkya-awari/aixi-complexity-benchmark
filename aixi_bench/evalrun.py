"""Update-free paired evaluation of all enabled arms on identical tasks, replicates and seeds."""
import gzip
import json
import time
from multiprocessing import Pool
from pathlib import Path

from .accctw import ACCTWUCTAgent, PerceptModel
from .agents import OracleAgent, RandomAgent, TablePolicyAgent, q_table_policy, train_q_learning
from .analysis import solve_dp
from .data import corpus_path, eval_seeds, load_corpus_file
from .env import FiniteTaskEnv
from .llmpolicy import policy_table_from_record
from .tasks import FiniteTask

AGENT_ORDER = ["random", "oracle", "q_learning", "ac_ctw_uct", "local_llm_policy", "local_llm_constrained"]
LLM_ARMS = ("local_llm_policy", "local_llm_constrained")


def llm_record_path(llm_dir, task_id, rep):
    return Path(llm_dir) / task_id / f"rep_{rep:02d}.json"


def play_episode(env, agent, seed, horizon):
    """One held-out episode. Returns (return, steps list)."""
    obs, _ = env.reset(seed=seed)
    agent.begin_episode(int(obs), seed)
    total, steps = 0.0, []
    for t in range(horizon):
        t0 = time.perf_counter()
        a = agent.act(t)
        dt = time.perf_counter() - t0
        o2, r, _, truncated, _ = env.step(a)
        steps.append((t, int(obs), a, int(r), int(o2), dt))
        agent.observe(a, int(o2), r)
        total += r
        obs = o2
        if truncated:
            break
    return total, steps


def job_tag(agents_enabled):
    arms = set(agents_enabled)
    if arms == {"local_llm_policy"}:
        return "llm"
    if arms == {"local_llm_constrained"}:
        return "llmc"
    return "main"


def job_path(out_dir, task_id, rep, tag="main"):
    return Path(out_dir) / "jobs" / f"{task_id}__rep{rep:02d}__{tag}.json.gz"


def build_agents(task, rep, corpus, data_dir, llm_dir, agents_enabled, constrained_dir=None):
    """Instantiate only the requested arms. Returns (agents, status)."""
    agents, status = {}, {}
    if "random" in agents_enabled:
        agents["random"] = RandomAgent()
    if "oracle" in agents_enabled:
        agents["oracle"] = OracleAgent(solve_dp(task), task.m)
    if "q_learning" in agents_enabled:
        q = train_q_learning(corpus, task.m, f"{task.task_id}/{rep}")
        agents["q_learning"] = TablePolicyAgent("q_learning", q_table_policy(q), task.m)
    if "ac_ctw_uct" in agents_enabled:
        model = PerceptModel(task.n_obs, task.m)
        model.fit(corpus)
        agents["ac_ctw_uct"] = ACCTWUCTAgent(model, task.horizon)
    for name in agents:
        status[name] = "evaluated"
    for arm, directory in (("local_llm_policy", llm_dir), ("local_llm_constrained", constrained_dir)):
        if arm not in agents_enabled:
            continue
        rec_path = llm_record_path(directory, task.task_id, rep) if directory else None
        if rec_path is not None and rec_path.exists():
            record = json.loads(rec_path.read_text(encoding="utf-8"))
            table = policy_table_from_record(record)
            status[arm] = "evaluated" if table is not None else record["status"]
            if table is not None:
                agents[arm] = TablePolicyAgent(arm, table, task.m)
        else:
            status[arm] = "skipped"
    return agents, status


def run_job(job):
    spec, rep, data_dir, llm_dir, out_dir, agents_enabled, constrained_dir = job
    out_path = job_path(out_dir, spec["task_id"], rep, job_tag(agents_enabled))
    if out_path.exists():
        return str(out_path)
    task = FiniteTask.from_spec(spec)
    corpus = load_corpus_file(corpus_path(data_dir, task.task_id, rep))
    env = FiniteTaskEnv(task)
    agents, status = build_agents(task, rep, corpus, data_dir, llm_dir, agents_enabled, constrained_dir)
    episodes, steps_rows = [], []
    seeds = eval_seeds(task.task_id, rep)
    for name in AGENT_ORDER:
        if name not in agents or name not in agents_enabled:
            continue
        agent = agents[name]
        for idx, seed in enumerate(seeds):
            t0 = time.perf_counter()
            ret, steps = play_episode(env, agent, seed, task.horizon)
            runtime = time.perf_counter() - t0
            logloss = None
            if agent.episode_logloss is not None and steps:
                logloss = agent.episode_logloss / len(steps)
            episodes.append(dict(task_id=task.task_id, rep=rep, agent=name, episode=idx, seed=seed,
                                 ret=ret, success=int(ret >= task.success_min_return),
                                 steps=len(steps), runtime_s=runtime, logloss_bits=logloss))
            for s in steps:
                steps_rows.append([task.task_id, rep, name, idx, *s])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump({"episodes": episodes, "steps": steps_rows, "status": status,
                   "task_id": task.task_id, "rep": rep}, f)
    tmp.replace(out_path)
    return str(out_path)


def run_all(corpus_dict, data_dir, llm_dir, out_dir, replicates=10, agents_enabled=None, processes=3,
            task_ids=None, progress=print, constrained_dir=None):
    agents_enabled = agents_enabled or AGENT_ORDER
    jobs = [(spec, rep, str(data_dir), str(llm_dir), str(out_dir), agents_enabled, constrained_dir)
            for spec in corpus_dict["tasks"] if task_ids is None or spec["task_id"] in task_ids
            for rep in range(replicates)]
    done = 0
    with Pool(processes) as pool:
        for _ in pool.imap_unordered(run_job, jobs):
            done += 1
            progress(f"job {done}/{len(jobs)}")
    return len(jobs)


def collect(out_dir):
    """Merge job files into dataframes (episodes, steps, agent status)."""
    import pandas as pd
    eps, steps, stat = [], [], []
    for p in sorted((Path(out_dir) / "jobs").glob("*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        eps += d["episodes"]
        steps += d["steps"]
        stat += [dict(task_id=d["task_id"], rep=d["rep"], agent=a, status=s) for a, s in d["status"].items()]
    steps_df = pd.DataFrame(steps, columns=["task_id", "rep", "agent", "episode", "t", "obs", "action",
                                            "reward", "next_obs", "decision_runtime_s"])
    return pd.DataFrame(eps), steps_df, pd.DataFrame(stat)
