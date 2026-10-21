"""Local Qwen model gate: pinned revision, license, cache path, SHA-256 and tokenizer manifests.

Only Qwen/Qwen2.5-0.5B-Instruct at one pinned revision is permitted. If any check fails the
arm is recorded as skipped; no other model and no random behavior is ever substituted.
"""
import json
import re
from pathlib import Path

from .hashing import sha256_file

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ALLOWED_PATTERNS = ["config.json", "generation_config.json", "model.safetensors", "merges.txt",
                    "tokenizer.json", "tokenizer_config.json", "vocab.json", "README.md", "LICENSE"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def parse_license(readme_text: str):
    match = re.search(r"^license:\s*([^\n]+)$", readme_text.split("---")[1] if readme_text.startswith("---") else "",
                      re.M)
    return match.group(1).strip() if match else None


def run_gate(out_dir, model_id=MODEL_ID, revision=MODEL_REVISION, snapshot_path=None):
    """Verify the model and write gate artifacts. Returns the gate dict (status pass|skipped)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = {"model_id": model_id, "revision": revision, "status": "skipped", "reason": None}
    try:
        if model_id != MODEL_ID or revision != MODEL_REVISION:
            raise RuntimeError("model id or revision differs from the pinned values")
        if snapshot_path is None:
            from huggingface_hub import snapshot_download
            snapshot_path = snapshot_download(model_id, revision=revision, allow_patterns=ALLOWED_PATTERNS)
        snap = Path(snapshot_path)
        if snap.name != revision:
            raise RuntimeError(f"cache snapshot {snap.name} is not the pinned revision")
        license_id = parse_license((snap / "README.md").read_text(encoding="utf-8"))
        if license_id != "apache-2.0":
            raise RuntimeError(f"unexpected model-card license: {license_id}")
        files = {p.name: sha256_file(p) for p in sorted(snap.iterdir()) if p.is_file() or p.is_symlink()}
        if "model.safetensors" not in files:
            raise RuntimeError("model.safetensors missing from cache")
        tokenizer = {n: files[n] for n in TOKENIZER_FILES if n in files}
        (out_dir / "qwen_cache_sha256_manifest.json").write_text(
            json.dumps({"cache_path": str(snap), "files": files}, indent=1), encoding="utf-8")
        (out_dir / "qwen_tokenizer_manifest.json").write_text(
            json.dumps({"tokenizer_files": tokenizer, "revision": revision}, indent=1), encoding="utf-8")
        gate.update(status="pass", license=license_id, cache_path=str(snap), n_files=len(files))
    except Exception as exc:  # any failure => the arm is skipped, never substituted
        gate["reason"] = f"{type(exc).__name__}: {exc}"
    (out_dir / "qwen_model_gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    return gate
