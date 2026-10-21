import math
import random

from aixi_bench import accctw
from aixi_bench.accctw import PerceptModel, UCTPlanner
from aixi_bench.ctw import CTW
from aixi_bench.data import collect_uniform_corpus, explore_seeds
from aixi_bench.generator import GENERATOR_SEED, generate_corpus
from aixi_bench.tasks import load_corpus

TASKS = {t.task_id: t for t in load_corpus(generate_corpus(GENERATOR_SEED))}


def kt(n0, n1):
    """KT block probability of a sequence with n0 zeros and n1 ones."""
    p = 1.0
    a = b = 0
    for _ in range(n0):
        p *= (a + 0.5) / (a + b + 1)
        a += 1
    for _ in range(n1):
        p *= (b + 0.5) / (a + b + 1)
        b += 1
    return p


def test_depth0_ctw_equals_kt():
    ctw = CTW(0)
    for bit in [1, 1, 0, 1]:
        ctw.update((), bit)
    assert math.isclose(math.exp(ctw.root_log_pw()), kt(1, 3), rel_tol=1e-12)


def test_depth1_ctw_is_weighted_mixture_at_every_node():
    # sequence of (ctx, bit): context bit alternates, so the child split is informative.
    seq = [((0,), 1), ((1,), 0), ((0,), 1), ((1,), 0), ((0,), 1)]
    ctw = CTW(1)
    for ctx, bit in seq:
        ctw.update(ctx, bit)
    leaf0 = kt(0, 3)  # ctx 0 saw 1,1,1
    leaf1 = kt(2, 0)  # ctx 1 saw 0,0
    expected_root = 0.5 * root_pe_sequential(seq) + 0.5 * leaf0 * leaf1
    assert math.isclose(math.exp(ctw.root_log_pw()), expected_root, rel_tol=1e-9)
    assert math.isclose(math.exp(ctw.nodes[(0,)][3]), leaf0, rel_tol=1e-9)


def root_pe_sequential(seq):
    p, a, b = 1.0, 0, 0
    for _, bit in seq:
        p *= ((a if bit == 0 else b) + 0.5) / (a + b + 1)
        a, b = (a + 1, b) if bit == 0 else (a, b + 1)
    return p


def test_bit_probabilities_normalize_and_not_root_frequency():
    ctw = CTW(2)
    rng = random.Random(0)
    for _ in range(200):
        c = (rng.randrange(2), rng.randrange(2))
        ctw.update(c, c[0] ^ c[1])  # bit is XOR of context: root frequency would be 0.5
    for c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        p1 = ctw.prob_bit(c, 1)
        assert math.isclose(p1 + ctw.prob_bit(c, 0), 1.0, rel_tol=1e-9)
        assert abs(p1 - (c[0] ^ c[1])) < 0.1


def test_update_revert_restores_exact_state():
    ctw = CTW(3)
    rng = random.Random(1)
    for _ in range(50):
        ctw.update(tuple(rng.randrange(2) for _ in range(3)), rng.randrange(2))
    before = ctw.digest()
    undo1 = ctw.update((1, 1, 0), 1)
    undo2 = ctw.update((0, 1, 0), 0)
    assert ctw.digest() != before
    ctw.revert(undo2)
    ctw.revert(undo1)
    assert ctw.digest() == before


def fitted(task_id, rep=0, episodes=256):
    task = TASKS[task_id]
    c = collect_uniform_corpus(task, explore_seeds(task_id, rep, episodes))
    model = PerceptModel(task.n_obs, task.m)
    model.fit(c)
    return task, model


def test_prefix_free_fixed_width_encoding():
    model = PerceptModel(4, 1)
    codes = {tuple(model.percept_bits(o, r)) for o in range(4) for r in (0, 1)}
    assert len(codes) == 8 and all(len(c) == model.obs_bits + 1 for c in codes)


def test_action_conditioning_changes_percept_distribution():
    # reactive_V1: reward = 1 iff action == best[obs]; reward prediction must depend on action
    task, model = fitted("reactive_V1")
    hist = model.initial_hist(0)
    d0 = dict(zip(*model.percept_dist(hist, 0)[:2]))
    d1 = dict(zip(*model.percept_dist(hist, 1)[:2]))
    def p_r1(d):
        return sum(p for (_, r), p in d.items() if r == 1)

    assert abs(p_r1(d0) - p_r1(d1)) > 0.3


def test_context_sensitivity_changes_prediction():
    task, model = fitted("contextual_V1")
    h_a = model.next_hist(model.initial_hist(0), 0, 0, 0)
    h_b = model.next_hist(model.initial_hist(0), 1, 0, 0)
    assert model.percept_dist(h_a, 0)[:2] != model.percept_dist(h_b, 0)[:2]


def test_reward_dependence_in_context():
    model = PerceptModel(2, 1)
    h = model.initial_hist(0)
    assert model.next_hist(h, 0, 1, 0) != model.next_hist(h, 0, 1, 1)


def test_planning_does_not_mutate_model():
    task, model = fitted("reactive_V1", episodes=64)
    digest = model.ctw.digest()
    planner = UCTPlanner(model, simulations=64)
    planner.plan(model.initial_hist(0), 8, random.Random(0))
    model.clear_caches()  # force recomputation through update/revert
    planner.plan(model.initial_hist(1), 8, random.Random(1))
    assert model.ctw.digest() == digest


def test_planner_deterministic_restore():
    task, model = fitted("reactive_V1", episodes=64)
    planner = UCTPlanner(model, simulations=64)
    a = planner.plan(model.initial_hist(0), 8, random.Random(5))
    b = planner.plan(model.initial_hist(0), 8, random.Random(5))
    assert a == b


def test_toy_task_action_preference():
    task, model = fitted("reactive_V2")
    planner = UCTPlanner(model, simulations=128)
    for key, s in task.key_to_state.items():
        best = int(task.R[s, 1, 0] == 1)
        assert planner.plan(model.initial_hist(key[-1]), 8, random.Random(3)) == best


def test_uct_contract_defaults():
    assert accctw.UCT_SIMULATIONS == 512
    assert accctw.UCT_MAX_HORIZON == 16
    assert math.isclose(accctw.UCT_EXPLORATION, math.sqrt(2))
    p = UCTPlanner(PerceptModel(2, 0))
    assert (p.simulations, p.max_horizon) == (512, 16)


def test_planning_horizon_is_min_16_remaining():
    class Spy(UCTPlanner):
        def _simulate(self, node, hist, depth, h, rng):
            self.seen = h
            return 0.0
    model = PerceptModel(2, 0)
    spy = Spy(model, simulations=1)
    spy.plan(model.initial_hist(0), 5, random.Random(0))
    assert spy.seen == 5
    spy.plan(model.initial_hist(0), 30, random.Random(0))
    assert spy.seen == 16


def test_agent_logloss_uses_frozen_model():
    task, model = fitted("reactive_V1", episodes=64)
    digest = model.ctw.digest()
    agent = accctw.ACCTWUCTAgent(model, task.horizon, UCTPlanner(model, simulations=8))
    agent.begin_episode(0, 123)
    a = agent.act(0)
    agent.observe(a, 0, 1)
    assert agent.episode_logloss > 0 and model.ctw.digest() == digest
