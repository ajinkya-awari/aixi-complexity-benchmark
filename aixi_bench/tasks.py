"""Frozen finite-state task representation, loading, and corpus validation."""
from dataclasses import dataclass, field

import numpy as np
import yaml

from .analysis import key_state_map
from .hashing import canonical_json, sha256_hex

FAMILY_IDS = ("reactive", "contextual", "delayed", "branching", "sparse_sequence")
VARIANT_IDS = ("V1", "V2", "V3", "V4")
HORIZONS = (8, 12, 16, 24, 32)


@dataclass
class FiniteTask:
    task_id: str
    family_id: str
    variant_id: str
    m: int
    T: np.ndarray  # (S, 2, S) transition probabilities
    R: np.ndarray  # (S, 2, S) integer rewards in {0, 1}
    obs_map: np.ndarray
    init: np.ndarray
    horizon: int
    success_min_return: int
    key_to_state: dict = field(default_factory=dict)

    @property
    def n_states(self):
        return self.T.shape[0]

    @property
    def n_obs(self):
        return int(self.obs_map.max()) + 1

    @classmethod
    def from_arrays(cls, task_id, family_id, variant_id, m, T, R, obs_map, init, horizon,
                    success_min_return):
        T = np.asarray(T, dtype=float)
        obs_map = np.asarray(obs_map, dtype=int)
        init = np.asarray(init, dtype=float)
        mapping, conflicts = key_state_map(T, obs_map, init, m)
        if conflicts:
            raise ValueError(f"{task_id}: public keys of order m={m} do not identify the latent state")
        return cls(task_id, family_id, variant_id, m, T, np.asarray(R, dtype=int), obs_map, init,
                   horizon, success_min_return, mapping)

    @classmethod
    def from_spec(cls, spec: dict):
        n = spec["num_latent_states"]
        T = np.zeros((n, 2, n))
        for s in range(n):
            for a in range(2):
                for s2, pm in spec["transitions"][s][a]:
                    T[s, a, s2] = pm / 1000.0
        init = np.zeros(n)
        for s, pm in spec["initial_distribution"]:
            init[s] = pm / 1000.0
        return cls.from_arrays(spec["task_id"], spec["family_id"], spec["variant_id"],
                               spec["public_history_order_m"], T, np.array(spec["rewards"]),
                               spec["observation_map"], init, spec["horizon"],
                               spec["success_min_return"])


def load_corpus(source) -> list:
    """Load FiniteTask objects from a corpus dict or a YAML path."""
    if not isinstance(source, dict):
        with open(source, "r", encoding="utf-8") as f:
            source = yaml.safe_load(f)
    return [FiniteTask.from_spec(t) for t in source["tasks"]]


def validate_corpus(corpus: dict) -> list:
    """Return problems found; an empty list means the corpus satisfies the contract."""
    problems = []
    tasks = corpus.get("tasks", [])
    if len(tasks) != 20:
        problems.append(f"expected 20 tasks, got {len(tasks)}")
    if "generator_seed" not in corpus:
        problems.append("generator_seed missing")
    for fam in FAMILY_IDS:
        variants = sorted(t["variant_id"] for t in tasks if t["family_id"] == fam)
        if variants != list(VARIANT_IDS):
            problems.append(f"family {fam} variants {variants} != {list(VARIANT_IDS)}")
        if set(corpus.get("family_variant_bundles", {}).get(fam, {})) != set(VARIANT_IDS):
            problems.append(f"family {fam} bundles incomplete")
    for t in tasks:
        tid = t["task_id"]
        if t["num_actions"] != 2:
            problems.append(f"{tid}: actions != 2")
        if not 2 <= t["num_latent_states"] <= 8:
            problems.append(f"{tid}: latent states out of range")
        if not 2 <= t["num_public_observations"] <= 4:
            problems.append(f"{tid}: observations out of range")
        if t["public_history_order_m"] > 3:
            problems.append(f"{tid}: m > 3")
        if t["horizon"] not in HORIZONS:
            problems.append(f"{tid}: horizon {t['horizon']} not allowed")
        body = {k: v for k, v in t.items() if k != "sha256"}
        if t.get("sha256") != sha256_hex(canonical_json(body).encode("utf-8")):
            problems.append(f"{tid}: sha256 mismatch")
    return problems
