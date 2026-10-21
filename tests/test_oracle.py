import numpy as np

from aixi_bench.analysis import solve_dp, uniform_value
from aixi_bench.generator import GENERATOR_SEED, generate_corpus
from aixi_bench.tasks import FiniteTask, load_corpus

TASKS = load_corpus(generate_corpus(GENERATOR_SEED))


def toy():
    # 2 states, obs identity; action 1 always gives reward 1 and action 0 gives 0.
    n = 2
    T = np.full((n, 2, n), 0.5)
    R = np.zeros((n, 2, n), dtype=int)
    R[:, 1, :] = 1
    return FiniteTask.from_arrays("toy", "toy", "V1", m=0, T=T, R=R, obs_map=[0, 1],
                                  init=np.array([0.5, 0.5]), horizon=8, success_min_return=1)


def test_oracle_hand_solvable():
    task = toy()
    sol = solve_dp(task)
    assert sol.v0 == 8.0
    assert all(sol.policy[t][s] == 1 for t in range(8) for s in range(2))
    assert uniform_value(task) == 4.0


def test_oracle_beats_random_on_every_task():
    for task in TASKS:
        assert solve_dp(task).v0 > uniform_value(task), task.task_id


def test_oracle_keys_cover_reachable():
    for task in TASKS:
        sol = solve_dp(task)
        for key in task.key_to_state:
            for t in range(task.horizon):
                assert sol.action(t, key) in (0, 1)
