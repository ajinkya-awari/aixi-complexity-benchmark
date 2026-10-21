"""Deterministic generator for the frozen 20-task corpus (5 families x 4 variants).

Every task is built so that the public observable-history key (last m action/observation
pairs plus the current observation) identifies the latent state. The generator verifies
this by exhaustive search and records the minimal sufficient order m. Probabilities are
stored as integer permille so the canonical serialization is float-free.
"""
import itertools
import random

import numpy as np

from .analysis import branching_factor, minimal_sufficient_m, solve_dp, uniform_value
from .hashing import canonical_json, derive_seed, sha256_hex
from .tasks import FAMILY_IDS, VARIANT_IDS, FiniteTask

GENERATOR_SEED = 20261003
MIN_ORACLE_RANDOM_GAP_PER_STEP = 0.03

BUNDLES = {
    "reactive": {
        "V1": dict(n_obs=2, horizon=8, stay_pm=0),
        "V2": dict(n_obs=3, horizon=12, stay_pm=0),
        "V3": dict(n_obs=4, horizon=16, stay_pm=0),
        "V4": dict(n_obs=4, horizon=24, stay_pm=500),
    },
    "contextual": {
        "V1": dict(n_obs=2, history_len=1, horizon=12, stay_pm=0),
        "V2": dict(n_obs=4, history_len=1, horizon=16, stay_pm=0),
        "V3": dict(n_obs=2, history_len=2, horizon=16, stay_pm=0),
        "V4": dict(n_obs=2, history_len=2, horizon=24, stay_pm=600),
    },
    "delayed": {
        "V1": dict(n_obs=2, delay=1, reward_rule="head", horizon=16),
        "V2": dict(n_obs=4, delay=1, reward_rule="head", horizon=16),
        "V3": dict(n_obs=2, delay=2, reward_rule="head", horizon=24),
        "V4": dict(n_obs=2, delay=2, reward_rule="all", horizon=32),
    },
    "branching": {
        "V1": dict(n_states=3, successors=2, horizon=12),
        "V2": dict(n_states=4, successors=2, horizon=16),
        "V3": dict(n_states=4, successors=3, horizon=24),
        "V4": dict(n_states=4, successors=4, horizon=32),
    },
    "sparse_sequence": {
        "V1": dict(length=2, n_obs=2, horizon=12),
        "V2": dict(length=3, n_obs=2, horizon=16),
        "V3": dict(length=4, n_obs=3, horizon=24),
        "V4": dict(length=6, n_obs=4, horizon=32),
    },
}

FAMILY_NAMES = {
    "reactive": "Observation directly identifies best action",
    "contextual": "Observation aliases latent states; bounded history needed",
    "delayed": "Reward depends on delayed action consequences",
    "branching": "Multiple action-conditioned transition branches",
    "sparse_sequence": "Success requires a fixed action/percept sequence",
}


def to_permille(weights):
    """Largest-remainder rounding of non-negative weights to integers summing to 1000."""
    total = float(sum(weights))
    raw = [1000.0 * w / total for w in weights]
    out = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - out[i], -i), reverse=True)
    for i in order[: 1000 - sum(out)]:
        out[i] += 1
    return out


def _mix(n, stay_index, stay_pm):
    weights = [(1000 - stay_pm) / n] * n
    weights[stay_index] += stay_pm
    return to_permille(weights)


def _empty(n):
    return [[[0] * n for _ in range(2)] for _ in range(n)], [[[0] * n for _ in range(2)] for _ in range(n)]


def build_reactive(p, rng):
    k = p["n_obs"]
    best = [rng.randrange(2) for _ in range(k)]
    if k >= 2 and len(set(best)) == 1:
        best[0] = 1 - best[0]
    Tpm, R = _empty(k)
    for s in range(k):
        for a in range(2):
            Tpm[s][a] = _mix(k, s, p["stay_pm"])
            for s2 in range(k):
                R[s][a][s2] = int(a == best[s])
    return Tpm, R, list(range(k)), to_permille([1] * k)


def build_contextual(p, rng):
    n_obs, k = p["n_obs"], p["history_len"]
    n_h = 2 ** k
    n = n_obs * n_h
    best = []
    for _ in range(n_obs):
        b = rng.randrange(2)
        best.append((b, 1 - b))
    Tpm, R = _empty(n)
    for o in range(n_obs):
        for h in range(n_h):
            s = o * n_h + h
            bit = bin(h).count("1") % 2
            for a in range(2):
                h2 = ((h << 1) | a) & (n_h - 1)
                for o2, pm in enumerate(_mix(n_obs, o, p["stay_pm"])):
                    s2 = o2 * n_h + h2
                    Tpm[s][a][s2] += pm
                for s2 in range(n):
                    R[s][a][s2] = int(a == best[o][bit])
    init = [0] * n
    for o, pm in enumerate(to_permille([1] * n_obs)):
        init[o * n_h] = pm
    return Tpm, R, [s // n_h for s in range(n)], init


def build_delayed(p, rng):
    n_obs, d = p["n_obs"], p["delay"]
    n_f = 2 ** d
    n = n_obs * n_f
    g = [rng.randrange(2) for _ in range(n_obs)]
    if len(set(g)) == 1:
        g[0] = 1 - g[0]
    Tpm, R = _empty(n)
    for o in range(n_obs):
        for f in range(n_f):
            s = o * n_f + f
            reward = (f >> (d - 1)) & 1 if p["reward_rule"] == "head" else int(f == n_f - 1)
            for a in range(2):
                f2 = ((f << 1) | int(a == g[o])) & (n_f - 1)
                for o2, pm in enumerate(to_permille([1] * n_obs)):
                    Tpm[s][a][o2 * n_f + f2] += pm
                for s2 in range(n):
                    R[s][a][s2] = reward
    init = [0] * n
    for o, pm in enumerate(to_permille([1] * n_obs)):
        init[o * n_f] = pm
    return Tpm, R, [s // n_f for s in range(n)], init


def build_branching(p, rng):
    n, b = p["n_states"], p["successors"]
    goal = rng.randrange(n)
    Tpm, R = _empty(n)
    for s in range(n):
        for a in range(2):
            succ = rng.sample(range(n), b)
            weights = [rng.randint(1, 10) for _ in succ]
            for s2, pm in zip(succ, to_permille(weights)):
                Tpm[s][a][s2] = pm
            for s2 in range(n):
                R[s][a][s2] = int(s2 == goal)
    return Tpm, R, list(range(n)), to_permille([1] * n)


def build_sparse_sequence(p, rng):
    length, n_obs = p["length"], p["n_obs"]
    seq = [rng.randrange(2) for _ in range(length)]
    Tpm, R = _empty(length)
    for s in range(length):
        for a in range(2):
            s2 = (s + 1) % length if a == seq[s] else 0
            Tpm[s][a][s2] = 1000
            for t in range(length):
                R[s][a][t] = int(a == seq[s] and s + 1 == length)
    init = [1000] + [0] * (length - 1)
    return Tpm, R, [min(s, n_obs - 1) for s in range(length)], init


BUILDERS = {
    "reactive": build_reactive,
    "contextual": build_contextual,
    "delayed": build_delayed,
    "branching": build_branching,
    "sparse_sequence": build_sparse_sequence,
}


def _round(x):
    return round(float(x), 6)


def build_task_spec(family, variant, seed):
    params = BUNDLES[family][variant]
    for attempt in range(200):
        rng = random.Random(derive_seed(seed, family, variant, attempt))
        Tpm, R, obs_map, init_pm = BUILDERS[family](params, rng)
        T = np.array(Tpm, dtype=float) / 1000.0
        init = np.array(init_pm, dtype=float) / 1000.0
        m = minimal_sufficient_m(T, np.array(obs_map), init)
        if m is None or not 2 <= len(obs_map) <= 8:
            continue
        task_id = f"{family}_{variant}"
        task = FiniteTask.from_arrays(task_id, family, variant, m, T, np.array(R), obs_map, init,
                                      params["horizon"], 1)
        v_star, v_rand = solve_dp(task).v0, uniform_value(task)
        if v_star - v_rand < MIN_ORACLE_RANDOM_GAP_PER_STEP * params["horizon"]:
            continue
        spec = {
            "task_id": task_id,
            "family_id": family,
            "family_name": FAMILY_NAMES[family],
            "variant_id": variant,
            "public_history_order_m": m,
            "num_actions": 2,
            "num_latent_states": len(obs_map),
            "num_public_observations": int(max(obs_map)) + 1,
            "horizon": params["horizon"],
            "reward_alphabet": [0, 1],
            "reward_encoding": ("fixed-width binary; reward is 1 bit; "
                                "observation uses ceil(log2(num_public_observations)) bits"),
            "observation_map": list(obs_map),
            "initial_distribution": [[s, pm] for s, pm in enumerate(init_pm) if pm > 0],
            "transitions": [[[[s2, pm] for s2, pm in enumerate(Tpm[s][a]) if pm > 0]
                             for a in range(2)] for s in range(len(obs_map))],
            "rewards": R,
            "terminal_states": [],
            "success_condition": "episode return >= success_min_return",
            "success_min_return": max(1, int(np.ceil(0.5 * v_star))),
            "generator_attempt": attempt,
        }
        task.success_min_return = spec["success_min_return"]
        bytes_len = len(canonical_json(spec).encode("utf-8"))
        spec["proxy_vector"] = {
            "reachable_history_state_count": len(task.key_to_state),
            "public_history_order_m": m,
            "horizon": params["horizon"],
            "branching_factor": _round(branching_factor(task)),
            "reward_density_uniform_policy": _round(v_rand / params["horizon"]),
            "canonical_task_spec_utf8_bytes": bytes_len,
        }
        spec["sha256"] = sha256_hex(canonical_json(spec).encode("utf-8"))
        return spec
    raise RuntimeError(f"no valid task for {family}/{variant}")


def generate_corpus(seed=GENERATOR_SEED):
    tasks = [build_task_spec(f, v, seed) for f, v in itertools.product(FAMILY_IDS, VARIANT_IDS)]
    return {
        "corpus_version": 1,
        "generator_seed": seed,
        "family_ids": list(FAMILY_IDS),
        "family_variant_bundles": {f: {v: dict(BUNDLES[f][v]) for v in VARIANT_IDS} for f in FAMILY_IDS},
        "tasks": tasks,
    }
