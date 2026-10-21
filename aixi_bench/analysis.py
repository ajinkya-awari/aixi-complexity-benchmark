"""Exact analysis of a frozen finite task: key sufficiency, DP oracle, proxies.

Hidden tables (T, R, latent states) are used here only for evaluator-side computation.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .interface import advance_window, flat_key, initial_window


def key_state_map(T, obs_map, init, m):
    """Return (key->latent dict, list of conflicting keys) by BFS over (latent, window)."""
    n_states = T.shape[0]
    start = initial_window(m)
    frontier = deque()
    seen = set()
    for s in range(n_states):
        if init[s] > 0:
            frontier.append((s, start))
            seen.add((s, start))
    mapping = {}
    conflicts = set()
    while frontier:
        s, win = frontier.popleft()
        key = flat_key(win, int(obs_map[s]))
        if mapping.setdefault(key, s) != s:
            conflicts.add(key)
        for a in range(T.shape[1]):
            new_win = advance_window(win, a, int(obs_map[s]), m)
            for s2 in range(n_states):
                if T[s, a, s2] > 0 and (s2, new_win) not in seen:
                    seen.add((s2, new_win))
                    frontier.append((s2, new_win))
    return mapping, sorted(conflicts)


def minimal_sufficient_m(T, obs_map, init, max_m=3):
    for m in range(max_m + 1):
        _, conflicts = key_state_map(T, obs_map, init, m)
        if not conflicts:
            return m
    return None


@dataclass
class Solution:
    v0: float
    policy: list  # policy[t][state] -> action
    key_to_state: dict

    def action(self, t: int, key: tuple) -> int:
        return int(self.policy[t][self.key_to_state[key]])


def solve_dp(task) -> Solution:
    """Exact finite-horizon DP with lowest-action tie break. Equivalent to DP over (t, public
    key) because every reachable key identifies exactly one latent state."""
    T, R = task.T, task.R
    n = T.shape[0]
    V = np.zeros(n)
    policy = [None] * task.horizon
    for t in range(task.horizon - 1, -1, -1):
        Q = np.einsum("sau,sau->sa", T, R + V[None, None, :])
        policy[t] = np.argmax(Q, axis=1)  # first (lowest) maximizer
        V = Q.max(axis=1)
    return Solution(v0=float(task.init @ V), policy=policy, key_to_state=task.key_to_state)


def uniform_value(task) -> float:
    T, R = task.T, task.R
    V = np.zeros(T.shape[0])
    for _ in range(task.horizon):
        V = 0.5 * np.einsum("sau,sau->s", T, R + V[None, None, :])
    return float(task.init @ V)


def branching_factor(task) -> float:
    """Mean number of distinct next percepts (observation, reward) per reachable (key, action)."""
    counts = []
    for s in task.key_to_state.values():
        for a in range(2):
            percepts = {(int(task.obs_map[s2]), int(task.R[s, a, s2]))
                        for s2 in range(task.T.shape[0]) if task.T[s, a, s2] > 0}
            counts.append(len(percepts))
    return float(np.mean(counts))
