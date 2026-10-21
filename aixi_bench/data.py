"""Frozen public exploration corpora and seed manifests.

Only public quantities (observations, actions, rewards, horizon) are written to the corpus.
Held-out evaluation episodes are represented by frozen seeds; trajectories are generated at
evaluation time because they depend on the agent's actions.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env import FiniteTaskEnv
from .hashing import canonical_json, derive_seed, sha256_file, sha256_hex

MASTER_SEED_LABEL = "aixi-bench-data-v1"
N_REPLICATES = 10
N_EXPLORATION = 256
N_EVAL = 100


@dataclass
class Corpus:
    obs: np.ndarray  # (E, H + 1)
    act: np.ndarray  # (E, H)
    rew: np.ndarray  # (E, H)

    @property
    def horizon(self):
        return self.act.shape[1]


@dataclass
class PublicMeta:
    """Public interface metadata: nothing here reveals hidden tables, latents, or proxies."""
    n_actions: int
    n_obs: int
    horizon: int
    m: int
    reward_alphabet: tuple = (0, 1)


def public_meta(task) -> PublicMeta:
    return PublicMeta(2, task.n_obs, task.horizon, task.m)


def explore_seeds(task_id, rep, n=N_EXPLORATION):
    return [derive_seed(MASTER_SEED_LABEL, "explore", task_id, rep, i) for i in range(n)]


def eval_seeds(task_id, rep, n=N_EVAL):
    return [derive_seed(MASTER_SEED_LABEL, "eval", task_id, rep, i) for i in range(n)]


def collect_uniform_corpus(task, seeds) -> Corpus:
    env = FiniteTaskEnv(task)
    H = task.horizon
    obs = np.zeros((len(seeds), H + 1), dtype=np.int8)
    act = np.zeros((len(seeds), H), dtype=np.int8)
    rew = np.zeros((len(seeds), H), dtype=np.int8)
    for e, seed in enumerate(seeds):
        o, _ = env.reset(seed=seed)
        rng = np.random.default_rng(derive_seed("explore-policy", seed))
        obs[e, 0] = o
        for t in range(H):
            a = int(rng.integers(2))
            o, r, _, _, _ = env.step(a)
            act[e, t], rew[e, t], obs[e, t + 1] = a, int(r), o
    return Corpus(obs, act, rew)


def corpus_path(data_dir, task_id, rep):
    return Path(data_dir) / task_id / f"rep_{rep:02d}.npz"


def load_corpus_file(path) -> Corpus:
    z = np.load(path)
    return Corpus(z["obs"], z["act"], z["rew"])


def freeze_data(tasks, data_dir, replicates=N_REPLICATES, n_explore=N_EXPLORATION, n_eval=N_EVAL):
    data_dir = Path(data_dir)
    exploration, evaluation = {}, {}
    for task in tasks:
        for rep in range(replicates):
            seeds = explore_seeds(task.task_id, rep, n_explore)
            path = corpus_path(data_dir, task.task_id, rep)
            path.parent.mkdir(parents=True, exist_ok=True)
            c = collect_uniform_corpus(task, seeds)
            np.savez_compressed(path, obs=c.obs, act=c.act, rew=c.rew)
            exploration[f"{task.task_id}/{rep}"] = {"seeds": seeds, "file": str(path.relative_to(data_dir)),
                                                    "sha256": sha256_file(path)}
            evaluation[f"{task.task_id}/{rep}"] = {"seeds": eval_seeds(task.task_id, rep, n_eval)}
    manifests = {
        "exploration_manifest.json": {"policy": "uniform", "episodes_per_replicate": n_explore,
                                      "replicates": replicates, "entries": exploration},
        "evaluation_manifest.json": {"episodes_per_replicate": n_eval, "replicates": replicates,
                                     "entries": evaluation},
    }
    hashes = {}
    for name, body in manifests.items():
        (data_dir / name).write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        hashes[name] = sha256_hex(canonical_json(body).encode("utf-8"))
    (data_dir / "manifest_hashes.json").write_text(json.dumps(hashes, indent=1, sort_keys=True), encoding="utf-8")
    return hashes


def verify_data(data_dir, tasks, replicates=N_REPLICATES, n_explore=N_EXPLORATION, n_eval=N_EVAL):
    """Return a list of problems (empty if the frozen data matches the contract)."""
    data_dir = Path(data_dir)
    problems = []
    expl = json.loads((data_dir / "exploration_manifest.json").read_text(encoding="utf-8"))
    evl = json.loads((data_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((data_dir / "manifest_hashes.json").read_text(encoding="utf-8"))
    for name, body in (("exploration_manifest.json", expl), ("evaluation_manifest.json", evl)):
        if hashes[name] != sha256_hex(canonical_json(body).encode("utf-8")):
            problems.append(f"{name}: hash mismatch")
    for task in tasks:
        for rep in range(replicates):
            key = f"{task.task_id}/{rep}"
            if key not in expl["entries"] or key not in evl["entries"]:
                problems.append(f"{key}: missing")
                continue
            if len(expl["entries"][key]["seeds"]) != n_explore:
                problems.append(f"{key}: exploration seed count")
            if len(evl["entries"][key]["seeds"]) != n_eval:
                problems.append(f"{key}: evaluation seed count")
            if set(expl["entries"][key]["seeds"]) & set(evl["entries"][key]["seeds"]):
                problems.append(f"{key}: exploration/evaluation seed overlap")
            path = data_dir / expl["entries"][key]["file"]
            if sha256_file(path) != expl["entries"][key]["sha256"]:
                problems.append(f"{key}: corpus hash mismatch")
            elif load_corpus_file(path).obs.shape != (n_explore, task.horizon + 1):
                problems.append(f"{key}: corpus shape")
    return problems
