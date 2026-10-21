"""Benchmark agents that act through the public history key.

Agent protocol (identical for every arm; no learning happens during evaluation):
    begin_episode(obs0, episode_seed) -> None
    act(t)                            -> action in {0, 1}
    observe(action, obs, reward)      -> None   (updates only within-episode context)
"""
import random

from .hashing import derive_seed
from .interface import KeyTracker

Q_ALPHA = 0.1
Q_GAMMA = 0.99
Q_PASSES = 20
DEFAULT_ACTION = 0  # declared action for public keys never seen in the exploration corpus


class Agent:
    name = "agent"
    episode_logloss = None

    def begin_episode(self, obs0, episode_seed):
        raise NotImplementedError

    def act(self, t):
        raise NotImplementedError

    def observe(self, action, obs, reward):
        pass


class RandomAgent(Agent):
    name = "random"

    def begin_episode(self, obs0, episode_seed):
        self.rng = random.Random(derive_seed("random-agent", episode_seed))

    def act(self, t):
        return self.rng.randrange(2)


class OracleAgent(Agent):
    """Evaluator-side reference policy; holds the exact DP solution (hidden tables)."""
    name = "oracle"

    def __init__(self, solution, m):
        self.solution = solution
        self.tracker = KeyTracker(m)

    def begin_episode(self, obs0, episode_seed):
        self.tracker.reset(obs0)

    def act(self, t):
        return self.solution.action(t, self.tracker.key())

    def observe(self, action, obs, reward):
        self.tracker.step(action, obs)


class TablePolicyAgent(Agent):
    """Stationary history-key -> action table; unseen keys use the declared default action."""

    def __init__(self, name, table, m):
        self.name = name
        self.table = table
        self.tracker = KeyTracker(m)

    def begin_episode(self, obs0, episode_seed):
        self.tracker.reset(obs0)

    def act(self, t):
        return self.table.get(self.tracker.key(), DEFAULT_ACTION)

    def observe(self, action, obs, reward):
        self.tracker.step(action, obs)


def corpus_keys(corpus, m):
    """Yield per-episode lists of (key_t, action_t, reward_t) plus the final key."""
    tracker = KeyTracker(m)
    for e in range(corpus.obs.shape[0]):
        keys = [tracker.reset(int(corpus.obs[e, 0]))]
        for t in range(corpus.horizon):
            keys.append(tracker.step(int(corpus.act[e, t]), int(corpus.obs[e, t + 1])))
        yield keys, corpus.act[e], corpus.rew[e]


def train_q_learning(corpus, m, seed_label):
    """Offline tabular Q-learning on the public corpus with the frozen contract:
    alpha=0.1, gamma=0.99, 20 deterministic shuffled passes, lowest-action tie break."""
    transitions = []
    for keys, acts, rews in corpus_keys(corpus, m):
        H = len(acts)
        for t in range(H):
            transitions.append((keys[t], int(acts[t]), float(rews[t]), keys[t + 1], t == H - 1))
    Q = {}
    for p in range(Q_PASSES):
        order = list(range(len(transitions)))
        random.Random(derive_seed("q-shuffle", seed_label, p)).shuffle(order)
        for i in order:
            k, a, r, k2, last = transitions[i]
            qk = Q.setdefault(k, [0.0, 0.0])
            target = r if last else r + Q_GAMMA * max(Q.get(k2, (0.0, 0.0)))
            qk[a] += Q_ALPHA * (target - qk[a])
    return Q


def q_table_policy(Q):
    return {k: (1 if q[1] > q[0] else 0) for k, q in Q.items()}
