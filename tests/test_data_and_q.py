import dataclasses

from aixi_bench import agents
from aixi_bench.data import (
    Corpus,
    PublicMeta,
    collect_uniform_corpus,
    eval_seeds,
    explore_seeds,
    freeze_data,
    public_meta,
    verify_data,
)
from aixi_bench.generator import GENERATOR_SEED, generate_corpus
from aixi_bench.tasks import load_corpus

TASKS = {t.task_id: t for t in load_corpus(generate_corpus(GENERATOR_SEED))}


def test_seed_counts_and_disjoint():
    e, v = explore_seeds("reactive_V1", 0), eval_seeds("reactive_V1", 0)
    assert len(e) == 256 and len(v) == 100 and not set(e) & set(v)
    assert explore_seeds("reactive_V1", 0) == e and explore_seeds("reactive_V1", 1) != e


def test_freeze_and_verify_roundtrip(tmp_path):
    tasks = [TASKS["reactive_V1"], TASKS["contextual_V1"]]
    freeze_data(tasks, tmp_path, replicates=2, n_explore=8, n_eval=4)
    assert verify_data(tmp_path, tasks, replicates=2, n_explore=8, n_eval=4) == []
    p = next(tmp_path.glob("reactive_V1/rep_00.npz"))
    p.write_bytes(p.read_bytes()[:-1] + b"x")
    assert verify_data(tmp_path, tasks, replicates=2, n_explore=8, n_eval=4)


def test_public_corpus_has_no_hidden_fields():
    task = TASKS["delayed_V3"]
    collect_uniform_corpus(task, explore_seeds(task.task_id, 0, 4))
    assert {f.name for f in dataclasses.fields(Corpus)} == {"obs", "act", "rew"}
    assert {f.name for f in dataclasses.fields(PublicMeta)} == {
        "n_actions", "n_obs", "horizon", "m", "reward_alphabet"}
    assert public_meta(task).m == task.m


def test_q_learning_contract_constants():
    assert (agents.Q_ALPHA, agents.Q_GAMMA, agents.Q_PASSES, agents.DEFAULT_ACTION) == (0.1, 0.99, 20, 0)


def test_q_learning_deterministic_and_learns_reactive():
    task = TASKS["reactive_V2"]
    c = collect_uniform_corpus(task, explore_seeds(task.task_id, 0))
    q1 = agents.train_q_learning(c, task.m, "x")
    q2 = agents.train_q_learning(c, task.m, "x")
    assert q1 == q2
    policy = agents.q_table_policy(q1)
    # reward depends only on action matching best[s]; Q should recover the optimal action
    for key, s in task.key_to_state.items():
        best = int(task.R[s, 1, 0] == 1)
        assert policy[key] == best


def test_q_lowest_action_tie_break():
    assert agents.q_table_policy({(0,): [0.5, 0.5]}) == {(0,): 0}


def test_q_logloss_not_applicable():
    assert agents.RandomAgent().episode_logloss is None
