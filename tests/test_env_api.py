import pytest
from gymnasium.utils.env_checker import check_env

from aixi_bench.env import FiniteTaskEnv
from aixi_bench.generator import GENERATOR_SEED, generate_corpus
from aixi_bench.tasks import load_corpus

TASKS = load_corpus(generate_corpus(GENERATOR_SEED))


@pytest.mark.parametrize("task", TASKS, ids=[t.task_id for t in TASKS])
def test_check_env(task):
    check_env(FiniteTaskEnv(task), skip_render_check=True)


def test_api_shapes_and_truncation():
    task = TASKS[0]
    env = FiniteTaskEnv(task)
    out = env.reset(seed=3)
    assert len(out) == 2 and out[1] == {}
    for i in range(task.horizon):
        o, r, term, trunc, info = env.step(0)
        assert term is False and info == {}
        assert trunc is (i == task.horizon - 1)


def test_seed_reproducible_and_action_space_seeded():
    env = FiniteTaskEnv(TASKS[5])
    env.reset(seed=11)
    a = [env.action_space.sample() for _ in range(5)]
    env.reset(seed=11)
    assert a == [env.action_space.sample() for _ in range(5)]
