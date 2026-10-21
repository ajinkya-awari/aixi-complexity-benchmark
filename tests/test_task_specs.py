import pytest

from aixi_bench.generator import FAMILY_IDS, GENERATOR_SEED, VARIANT_IDS, generate_corpus
from aixi_bench.hashing import canonical_json, sha256_hex
from aixi_bench.tasks import load_corpus, validate_corpus

PROXY_KEYS = {
    "reachable_history_state_count", "public_history_order_m", "horizon",
    "branching_factor", "reward_density_uniform_policy", "canonical_task_spec_utf8_bytes",
}


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(GENERATOR_SEED)


def test_exactly_20_tasks_5x4(corpus):
    tasks = corpus["tasks"]
    assert len(tasks) == 20
    for fam in FAMILY_IDS:
        assert sorted(t["variant_id"] for t in tasks if t["family_id"] == fam) == list(VARIANT_IDS)
    assert FAMILY_IDS == ("reactive", "contextual", "delayed", "branching", "sparse_sequence")


def test_corpus_declares_seed_and_bundles(corpus):
    assert corpus["generator_seed"] == GENERATOR_SEED
    for fam in FAMILY_IDS:
        assert set(corpus["family_variant_bundles"][fam]) == set(VARIANT_IDS)


def test_task_contract(corpus):
    for t in corpus["tasks"]:
        assert t["num_actions"] == 2
        assert 2 <= t["num_latent_states"] <= 8
        assert 2 <= t["num_public_observations"] <= 4
        assert t["public_history_order_m"] <= 3
        assert t["horizon"] in (8, 12, 16, 24, 32)
        assert t["reward_alphabet"] == [0, 1]
        assert t["terminal_states"] == []
        assert set(t["proxy_vector"]) == PROXY_KEYS
        assert "K" not in t and "kolmogorov" not in canonical_json(t).lower()


def test_hash_is_canonical_sha256(corpus):
    for t in corpus["tasks"]:
        body = {k: v for k, v in t.items() if k != "sha256"}
        assert t["sha256"] == sha256_hex(canonical_json(body).encode("utf-8"))


def test_generation_is_deterministic(corpus):
    again = generate_corpus(GENERATOR_SEED)
    assert canonical_json(again) == canonical_json(corpus)


def test_validate_accepts_and_rejects(corpus):
    assert validate_corpus(corpus) == []
    bad = dict(corpus, tasks=corpus["tasks"][:-1])
    assert validate_corpus(bad)
    tampered = [dict(t) for t in corpus["tasks"]]
    tampered[0]["horizon"] = 9
    assert validate_corpus(dict(corpus, tasks=tampered))


def test_public_key_sufficiency_verified(corpus):
    for task in load_corpus(corpus):
        assert task.key_to_state, task.task_id
