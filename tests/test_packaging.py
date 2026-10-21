import re
import tomllib
from pathlib import Path

from aixi_bench.__main__ import main

ROOT = Path(__file__).resolve().parent.parent


def load():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_build_system_and_entrypoint():
    cfg = load()
    assert cfg["build-system"]["build-backend"] == "setuptools.build_meta"
    assert cfg["build-system"]["requires"] == ["setuptools==83.0.0", "wheel==0.47.0"]
    assert cfg["project"]["scripts"] == {"aixi-bench": "aixi_bench.__main__:main"}


def test_pinned_runtime_dependencies():
    deps = set(load()["project"]["dependencies"])
    for pin in ["gymnasium==1.3.0", "numpy==2.5.1", "torch==2.13.0", "transformers==5.14.1", "pandas==3.0.5",
                "matplotlib==3.11.1", "scipy==1.18.0", "pydantic==2.13.4", "PyYAML==6.0.3"]:
        assert pin in deps
    dev = set(load()["project"]["optional-dependencies"]["dev"])
    assert {"pytest==9.1.1", "ruff==0.16.2"} <= dev


def test_no_remote_or_classic_control_dependencies():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for banned in ("openai", "anthropic", "box2d", "lunarlander", "classic-control", "mujoco"):
        assert banned not in text
    for path in (ROOT / "aixi_bench").glob("*.py"):
        src = path.read_text(encoding="utf-8")
        pattern = r"classic_control|box2d|lunarlander|cartpole|import openai|import anthropic"
        assert not re.search(pattern, src, re.I), path


def test_cli_help_exits_zero(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "aixi-bench" in capsys.readouterr().out
