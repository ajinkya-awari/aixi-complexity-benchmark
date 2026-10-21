from aixi_bench.accctw import ACCTWUCTAgent, PerceptModel, UCTPlanner
from aixi_bench.agents import OracleAgent, RandomAgent, TablePolicyAgent, q_table_policy, train_q_learning
from aixi_bench.analysis import solve_dp
from aixi_bench.data import collect_uniform_corpus, eval_seeds, explore_seeds
from aixi_bench.env import FiniteTaskEnv
from aixi_bench.evalrun import job_tag, play_episode
from aixi_bench.generator import GENERATOR_SEED, generate_corpus
from aixi_bench.tasks import load_corpus

TASK = {t.task_id: t for t in load_corpus(generate_corpus(GENERATOR_SEED))}["contextual_V1"]


def make_agents():
    c = collect_uniform_corpus(TASK, explore_seeds(TASK.task_id, 0, 64))
    model = PerceptModel(TASK.n_obs, TASK.m)
    model.fit(c)
    return {
        "random": RandomAgent(),
        "oracle": OracleAgent(solve_dp(TASK), TASK.m),
        "q": TablePolicyAgent("q", q_table_policy(train_q_learning(c, TASK.m, "t")), TASK.m),
        "uct": ACCTWUCTAgent(model, TASK.horizon, UCTPlanner(model, simulations=16)),
    }, model


def test_all_agents_share_one_interface_and_seeds():
    agents, _ = make_agents()
    env = FiniteTaskEnv(TASK)
    seeds = eval_seeds(TASK.task_id, 0)[:3]
    firsts = {}
    for name, agent in agents.items():
        for seed in seeds:
            ret, steps = play_episode(env, agent, seed, TASK.horizon)
            assert len(steps) == TASK.horizon
            firsts.setdefault(seed, set()).add(steps[0][1])  # same initial observation for every agent
    assert all(len(v) == 1 for v in firsts.values())


def test_oracle_matches_analytic_value_on_average():
    agents, _ = make_agents()
    env = FiniteTaskEnv(TASK)
    rets = [play_episode(env, agents["oracle"], s, TASK.horizon)[0] for s in eval_seeds(TASK.task_id, 0)]
    assert abs(sum(rets) / len(rets) - solve_dp(TASK).v0) < 1.0


def test_no_learning_during_evaluation():
    agents, model = make_agents()
    env = FiniteTaskEnv(TASK)
    digest = model.ctw.digest()
    table = dict(agents["q"].table)
    for agent in agents.values():
        for seed in eval_seeds(TASK.task_id, 0)[:2]:
            play_episode(env, agent, seed, TASK.horizon)
    assert model.ctw.digest() == digest and agents["q"].table == table


def test_job_tag_separates_llm_arm():
    assert job_tag(["local_llm_policy"]) == "llm"
    assert job_tag(["random", "oracle"]) == "main"
