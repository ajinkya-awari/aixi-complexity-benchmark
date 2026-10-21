"""Local LLM policy-table arm (Qwen2.5-0.5B-Instruct, deterministic greedy decoding).

The model sees only the shared public exploration corpus (per-key action statistics) and
public interface metadata, and emits one complete policy table: a string with one action digit
per required history key. Invalid output is recorded as invalid; there is no fallback.
"""
import json
import re

from .agents import corpus_keys
from .hashing import sha256_hex
from .modelgate import MODEL_ID, MODEL_REVISION

SCHEMA_VERSION = "policy-digits-v1"
MAX_NEW_TOKENS_SLACK = 8  # fixed rule: max_new_tokens = n_keys + 8 (one token per digit + slack)
PROMPT_TEMPLATE = (
    "You choose actions for a sequential decision task with 2 actions (0 and 1). "
    "The task has <N> numbered situations. For each situation you are given statistics from "
    "random exploration: how often each action was tried and the mean reward it earned.\n\n"
    "<TABLE>\n\n"
    "Choose the best action (0 or 1) for every situation. Reply with ONLY a string of exactly "
    "<N> digits, each 0 or 1, one digit per situation in order, with no spaces or other text."
)


def required_keys(corpus, m):
    """Sorted public history keys observed at decision time in the exploration corpus."""
    seen = set()
    for keys, _, _ in corpus_keys(corpus, m):
        seen.update(keys[:-1])
    return sorted(seen)


def key_statistics(corpus, m):
    stats = {}
    for keys, acts, rews in corpus_keys(corpus, m):
        for t in range(len(acts)):
            s = stats.setdefault(keys[t], [[0, 0.0], [0, 0.0]])
            s[int(acts[t])][0] += 1
            s[int(acts[t])][1] += float(rews[t])
    return stats


def build_prompt(corpus, m):
    keys = required_keys(corpus, m)
    stats = key_statistics(corpus, m)
    lines = []
    for i, key in enumerate(keys, 1):
        parts = []
        for a in range(2):
            n, total = stats[key][a]
            if n:
                parts.append("action " + str(a) + ": tried " + str(n) + ", mean reward " + "%.2f" % (total / n))
            else:
                parts.append("action " + str(a) + ": never tried")
        lines.append(str(i) + ". " + "; ".join(parts))
    text = PROMPT_TEMPLATE.replace("<N>", str(len(keys))).replace("<TABLE>", "\n".join(lines))
    return keys, text


def validate_output(text: str, n_keys: int):
    """Return (status, parsed_actions|None, reason). Strict schema: exactly n_keys digits in {0,1}."""
    s = text.strip()
    if not re.fullmatch(r"[01]+", s):
        return "invalid", None, "output is not a pure 0/1 digit string"
    if len(s) != n_keys:
        return "invalid", None, "expected " + str(n_keys) + " digits, got " + str(len(s))
    return "valid", [int(c) for c in s], ""


class QwenPolicyGenerator:
    def __init__(self, gate: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = gate["cache_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        self.model.train(False)  # inference mode
        torch.set_num_threads(1)
        self.torch = torch

    def generate(self, prompt: str, n_keys: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                 return_dict=True)
        with self.torch.no_grad():
            out = self.model.generate(**ids, do_sample=False, max_new_tokens=n_keys + MAX_NEW_TOKENS_SLACK,
                                      temperature=None, top_p=None, top_k=None)
        return self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_policy_record(generator, corpus, m, tokenizer_hashes: dict):
    keys, prompt = build_prompt(corpus, m)
    raw = generator.generate(prompt, len(keys))
    status, actions, reason = validate_output(raw, len(keys))
    return {
        "status": status, "reason": reason, "raw_output": raw, "n_keys": len(keys),
        "keys": [list(k) for k in keys], "actions": actions,
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "schema_version": SCHEMA_VERSION,
        "decoding": {"do_sample": False, "max_new_tokens_rule": "n_keys + " + str(MAX_NEW_TOKENS_SLACK),
                     "temperature": None, "top_p": None},
        "prompt_template_hash": sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        "prompt_instance_hash": sha256_hex(prompt.encode("utf-8")),
        "tokenizer_hash": sha256_hex(json.dumps(tokenizer_hashes, sort_keys=True).encode("utf-8")),
    }


def policy_table_from_record(record):
    if record.get("status") != "valid":
        return None
    return {tuple(k): a for k, a in zip(record["keys"], record["actions"])}
