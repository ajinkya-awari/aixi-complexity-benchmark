from pathlib import Path

from aixi_bench import drafts

ROOT = Path(__file__).resolve().parent.parent


def test_bundle_builds_and_is_consistent(tmp_path):
    out = tmp_path / "bundle"
    drafts.write_arxiv_bundle(ROOT / "results/local_run", ROOT / "task_specs/frozen_finite_tasks.yaml", out,
                              repo_root=ROOT, llm_dir=ROOT / "artifacts/models/local_llm_policy",
                              constrained_dir=ROOT / "artifacts/models/local_llm_constrained")
    tex = (out / "main.tex").read_text(encoding="utf-8")
    assert "@@" not in tex and "post-hoc" in tex and "86 of 200" in tex
    for name in ("suite_results", "paired_contrasts", "proxy_summary", "task_corpus"):
        assert (out / "tables" / (name + ".tex")).exists()
    assert (out / "figures" / "gap_by_task.png").exists() and (out / "MANIFEST.sha256").exists()
    assert (tmp_path / "aixi_complexity_arxiv_bundle.zip").exists()


def test_constrained_tables_match_primary_on_valid_records():
    assert drafts.overlap_identical(ROOT / "artifacts/models/local_llm_policy",
                                    ROOT / "artifacts/models/local_llm_constrained")
