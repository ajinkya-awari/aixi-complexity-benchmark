"""AC-CTW percept model and the MC-AIXI-inspired AC-CTW-UCT planner.

The model is an action-conditional CTW predictor over percepts (next observation, reward).
Each interaction cycle is encoded as bits: action (1 bit, side information inserted before
the percept), observation (fixed-width binary, prefix-free), reward (1 bit). Only percept bits
are predicted and trained; the action bit is context only. This is MC-AIXI-inspired, not a
faithful reproduction of MC-AIXI-CTW (Veness et al.).

After offline fitting on the public exploration corpus the model is frozen: planning and
evaluation condition on the within-episode bit history but never update tree counts.
"""
import math
import random
from bisect import bisect_right

from .agents import Agent
from .ctw import CTW
from .hashing import derive_seed

UCT_SIMULATIONS = 512
UCT_MAX_HORIZON = 16
UCT_EXPLORATION = math.sqrt(2.0)


def obs_bit_width(n_obs: int) -> int:
    return max(1, math.ceil(math.log2(n_obs)))


class PerceptModel:
    def __init__(self, n_obs: int, m: int):
        self.n_obs = n_obs
        self.obs_bits = obs_bit_width(n_obs)
        self.step_bits = 1 + self.obs_bits + 1
        self.depth = (m + 2) * self.step_bits  # covers earlier bits of the percept being predicted
        self.ctw = CTW(self.depth)
        self._memo = {}
        self._ids = {}      # interned history tuple -> state id (speeds up planning rollouts)
        self._hists = []
        self._trans = {}    # state_id * 2 + action -> (cum probs, rewards, next state ids)

    # ---- encoding (prefix-free: fixed width, most significant bit first) ----
    def percept_bits(self, obs: int, reward: int):
        return [(obs >> i) & 1 for i in range(self.obs_bits - 1, -1, -1)] + [int(reward)]

    def initial_hist(self, obs0: int) -> tuple:
        return self._push((0,) * self.depth, self.percept_bits(obs0, 0))

    def next_hist(self, hist, action, obs, reward) -> tuple:
        return self._push(hist, [action] + self.percept_bits(obs, reward))

    def _push(self, hist, bits):
        return (tuple(hist) + tuple(bits))[-self.depth:]

    @staticmethod
    def _ctx(hist, action):
        return tuple(reversed(hist + (action,)))  # most recent bit first

    # ---- training (the only place tree counts change) ----
    def clear_caches(self):
        """Drop memoized predictions; the next query recomputes through reversible CTW updates."""
        self._memo.clear()
        self._trans.clear()

    def intern(self, hist) -> int:
        sid = self._ids.get(hist)
        if sid is None:
            sid = self._ids[hist] = len(self._hists)
            self._hists.append(hist)
        return sid

    def entry(self, sid, action):
        key = sid * 2 + action
        e = self._trans.get(key)
        if e is None:
            hist = self._hists[sid]
            percepts, _, cum = self.percept_dist(hist, action)
            e = self._trans[key] = (
                cum, [r for _, r in percepts],
                [self.intern(self.next_hist(hist, action, o, r)) for o, r in percepts])
        return e

    def fit(self, corpus):
        self.clear_caches()
        for e in range(corpus.obs.shape[0]):
            hist = self.initial_hist(int(corpus.obs[e, 0]))
            for t in range(corpus.horizon):
                a, o, r = int(corpus.act[e, t]), int(corpus.obs[e, t + 1]), int(corpus.rew[e, t])
                ctx = list(reversed(hist + (a,)))
                for bit in self.percept_bits(o, r):
                    self.ctw.update(tuple(ctx), bit)
                    ctx.insert(0, bit)
                hist = self.next_hist(hist, a, o, r)
        self.clear_caches()

    # ---- frozen prediction ----
    def percept_dist(self, hist, action):
        """Normalized P(percept | hist, action) over valid percepts; memoized, model unchanged."""
        key = (hist, action)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        raw = {}
        nb = self.obs_bits + 1

        def rec(ctx, bits, prob):
            if len(bits) == nb:
                o = 0
                for b in bits[:-1]:
                    o = (o << 1) | b
                if o < self.n_obs:
                    raw[(o, bits[-1])] = prob
                return
            p1 = self.ctw.prob_bit(ctx, 1)
            for bit, p in ((0, 1.0 - p1), (1, p1)):
                undo = self.ctw.update(ctx, bit)
                rec((bit,) + ctx[: self.depth - 1], bits + [bit], prob * p)
                self.ctw.revert(undo)

        rec(self._ctx(hist, action), [], 1.0)
        total = sum(raw.values())
        percepts = sorted(raw)
        probs = [raw[p] / total for p in percepts]
        cum, acc = [], 0.0
        for p in probs:
            acc += p
            cum.append(acc)
        out = (percepts, probs, cum)
        self._memo[key] = out
        return out

    def sample_percept(self, hist, action, u):
        percepts, _, cum = self.percept_dist(hist, action)
        for percept, c in zip(percepts, cum):
            if u < c:
                return percept
        return percepts[-1]

    def logprob_bits(self, hist, action, obs, reward) -> float:
        """-log2 P(percept) under the frozen model (for log-loss reporting)."""
        percepts, probs, _ = self.percept_dist(hist, action)
        p = probs[percepts.index((obs, int(reward)))]
        return -math.log2(max(p, 1e-12))


class _Node:
    __slots__ = ("n", "na", "va", "kids")

    def __init__(self):
        self.n = 0
        self.na = [0, 0]
        self.va = [0.0, 0.0]
        self.kids = {}


class UCTPlanner:
    def __init__(self, model: PerceptModel, simulations=UCT_SIMULATIONS,
                 max_horizon=UCT_MAX_HORIZON, exploration=UCT_EXPLORATION):
        self.model = model
        self.simulations = simulations
        self.max_horizon = max_horizon
        self.exploration = exploration

    def plan(self, hist, remaining_steps, rng: random.Random) -> int:
        h = min(self.max_horizon, remaining_steps)
        sid = self.model.intern(hist)
        root = _Node()
        for _ in range(self.simulations):
            self._simulate(root, sid, h, h, rng)
        means = [root.va[a] / root.na[a] if root.na[a] else -1.0 for a in range(2)]
        best = max(means)
        tied = [a for a in range(2) if means[a] == best]
        return tied[0] if len(tied) == 1 else rng.choice(tied)

    def _rollout(self, sid, depth, rng):
        model, total = self.model, 0.0
        trans, rand, randrange = model._trans, rng.random, rng.randrange
        for _ in range(depth):
            a = randrange(2)
            e = trans.get(sid * 2 + a) or model.entry(sid, a)
            cum, rews, nexts = e
            i = bisect_right(cum, rand())
            if i == len(cum):
                i -= 1
            total += rews[i]
            sid = nexts[i]
        return total

    def _simulate(self, node, sid, depth, h, rng):
        if depth == 0:
            return 0.0
        if node.n == 0:
            node.n = 1
            return self._rollout(sid, depth, rng)
        unvisited = [a for a in range(2) if node.na[a] == 0]
        if unvisited:
            a = unvisited[0] if len(unvisited) == 1 else rng.choice(unvisited)
        else:
            log_n = math.log(node.n)
            scores = [node.va[i] / node.na[i] / h + self.exploration * math.sqrt(log_n / node.na[i])
                      for i in range(2)]
            best = max(scores)
            tied = [i for i in range(2) if scores[i] == best]
            a = tied[0] if len(tied) == 1 else rng.choice(tied)
        model = self.model
        cum, rews, nexts = model._trans.get(sid * 2 + a) or model.entry(sid, a)
        i = bisect_right(cum, rng.random())
        if i == len(cum):
            i -= 1
        child = node.kids.get((a, i))
        if child is None:
            child = node.kids[(a, i)] = _Node()
        ret = rews[i] + self._simulate(child, nexts[i], depth - 1, h, rng)
        node.n += 1
        node.na[a] += 1
        node.va[a] += ret
        return ret


class ACCTWUCTAgent(Agent):
    name = "ac_ctw_uct"

    def __init__(self, model: PerceptModel, horizon: int, planner: UCTPlanner = None):
        self.model = model
        self.horizon = horizon
        self.planner = planner or UCTPlanner(model)

    def begin_episode(self, obs0, episode_seed):
        self.seed = episode_seed
        self.hist = self.model.initial_hist(obs0)
        self.episode_logloss = 0.0
        self.n_percepts = 0
        self._last_hist = None

    def act(self, t):
        rng = random.Random(derive_seed("uct", self.seed, t))
        return self.planner.plan(self.hist, self.horizon - t, rng)

    def observe(self, action, obs, reward):
        self.episode_logloss += self.model.logprob_bits(self.hist, action, obs, reward)
        self.n_percepts += 1
        self.hist = self.model.next_hist(self.hist, action, obs, int(reward))
