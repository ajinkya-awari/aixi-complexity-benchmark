import inspect

from aixi_bench import llmpolicy, modelgate
from aixi_bench.data import collect_uniform_corpus, explore_seeds
from aixi_bench.generator import GENERATOR_SEED, generate_corpus
from aixi_bench.tasks import load_corpus

TASK = {t.task_id: t for t in load_corpus(generate_corpus(GENERATOR_SEED))}["contextual_V1"]
CORPUS = collect_uniform_corpus(TASK, explore_seeds(TASK.task_id, 0))


def test_pins():
    assert modelgate.MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert modelgate.MODEL_REVISION == "7ae557604adf67be50417f59c2c2f167def9a775"


def test_validation_rules():
    assert llmpolicy.validate_output("0101", 4) == ("valid", [0, 1, 0, 1], "")
    assert llmpolicy.validate_output(" 0101\n", 4)[0] == "valid"
    assert llmpolicy.validate_output("010", 4)[0] == "invalid"       # incomplete coverage
    assert llmpolicy.validate_output("01012", 4)[0] == "invalid"     # out-of-range action
    assert llmpolicy.validate_output("Sure: 0101", 4)[0] == "invalid"  # schema
    assert llmpolicy.validate_output("", 4)[0] == "invalid"


def test_prompt_deterministic_and_public_only():
    keys1, p1 = llmpolicy.build_prompt(CORPUS, TASK.m)
    keys2, p2 = llmpolicy.build_prompt(CORPUS, TASK.m)
    assert p1 == p2 and keys1 == keys2
    for forbidden in ("latent", "oracle", "contextual", "proxy", "transition"):
        assert forbidden not in p1.lower()


def test_required_keys_are_observed_public_keys():
    keys = llmpolicy.required_keys(CORPUS, TASK.m)
    assert set(keys) <= set(TASK.key_to_state)


def test_greedy_decoding_contract():
    src = inspect.getsource(llmpolicy.QwenPolicyGenerator.generate)
    assert "do_sample=False" in src and "max_new_tokens=n_keys + MAX_NEW_TOKENS_SLACK" in src
    assert "temperature=None" in src and "top_p=None" in src


def test_invalid_has_no_substituted_table():
    assert llmpolicy.policy_table_from_record({"status": "invalid"}) is None
    assert llmpolicy.policy_table_from_record({"status": "skipped"}) is None


def test_gate_skips_on_wrong_revision(tmp_path):
    gate = modelgate.run_gate(tmp_path, revision="deadbeef")
    assert gate["status"] == "skipped" and "pinned" in gate["reason"]


def test_license_parser():
    assert modelgate.parse_license("---\nlicense: apache-2.0\nlanguage:\n- en\n---\n# x") == "apache-2.0"
    assert modelgate.parse_license("# no front matter") is None


def test_constrained_decoding_contract():
    src = inspect.getsource(llmpolicy.QwenPolicyGenerator.generate_constrained)
    assert "do_sample=False" in src and "max_new_tokens=n_keys" in src and "min_new_tokens=n_keys" in src
    assert "prefix_allowed_tokens_fn" in src and "temperature=None" in src and "top_p=None" in src


def test_constrained_record_marks_mode():
    class Fake:
        def generate_constrained(self, prompt, n_keys):
            return "0" * n_keys

    rec = llmpolicy.generate_policy_record(Fake(), CORPUS, TASK.m, {}, constrained=True)
    assert rec["status"] == "valid" and rec["decoding"]["mode"] == "constrained"
