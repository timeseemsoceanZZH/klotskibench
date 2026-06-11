"""
Tests for one-shot prompting support (prompt_mode one_shot_rules_v3).

Covers:
- default prompt_mode is zero_shot_rules_v3
- zero-shot output is unchanged by default
- one-shot prompt contains exactly one exemplar block
- one-shot prompt has no oracle leakage
- runner can load one-shot exemplars
- --dry-run works with one_shot_rules_v3
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _import_runner():
    import run_mini_ollama_all_tasks as m  # noqa: WPS433

    return m


def _import_exporter():
    import export_one_shot_exemplars as e  # noqa: WPS433

    return e


# ---------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------


def test_default_prompt_mode_is_zero_shot_rules_v3():
    m = _import_runner()
    args = m.parse_runner_args([])
    assert args.prompt_mode == "zero_shot_rules_v3"
    assert args.exemplar_root is None


def test_prompt_mode_one_shot_accepted():
    m = _import_runner()
    args = m.parse_runner_args(["--prompt-mode", "one_shot_rules_v3"])
    assert args.prompt_mode == "one_shot_rules_v3"


# ---------------------------------------------------------------------------
# Zero-shot prompt is unchanged (requirement: default behaviour unmodified)
# ---------------------------------------------------------------------------


def test_zero_shot_prompt_identical_to_build_prompt():
    m = _import_runner()
    case = {
        "task": "s1",
        "input": {"state": {"grid_size": [5, 4], "blocks": {}}},
    }
    assert m.build_prompt(case) == m.build_prompt(case)


def test_build_prompt_signature_unchanged():
    """build_prompt still works with just a case dict."""
    m = _import_runner()
    case = {
        "task": "r1",
        "input": {
            "initial_state": {
                "blocks": {"Caocao": {"shape": [2, 2], "pos": [0, 1]}},
                "grid_size": [5, 4],
            }
        },
        "gold": {"optimal_depth": 1},
        "meta": {"shortest_solution_actions": []},
    }
    prompt = m.build_prompt(case)
    assert "5 rows by 4 columns" in prompt
    assert "[3, 1]" in prompt


# ---------------------------------------------------------------------------
# One-shot prompt structure
# ---------------------------------------------------------------------------


def _minimal_exemplar(task: str) -> dict:
    return {
        "task": task,
        "input": {"state": {"grid_size": [5, 4], "blocks": {}}},
        "exemplar_output": {"label": "valid"},
        "description": "Example description.",
    }


def test_one_shot_prompt_contains_example_section():
    m = _import_runner()
    case = {
        "task": "s1",
        "input": {"state": {"grid_size": [5, 4], "blocks": {}}},
    }
    exemplar = _minimal_exemplar("s1")
    prompt = m.build_one_shot_prompt(case, exemplar)
    assert "--- Example ---" in prompt
    assert "--- End of Example ---" in prompt


def test_one_shot_prompt_contains_exactly_one_exemplar():
    m = _import_runner()
    case = {
        "task": "t1",
        "input": {"current_state": {"grid_size": [5, 4], "blocks": {}}},
    }
    exemplar = {
        "task": "t1",
        "input": {"current_state": {"grid_size": [5, 4], "blocks": {}}},
        "exemplar_output": {"legal_actions": []},
        "description": "No legal moves from empty state.",
    }
    prompt = m.build_one_shot_prompt(case, exemplar)
    assert prompt.count("--- Example ---") == 1
    assert prompt.count("--- End of Example ---") == 1


def test_one_shot_prompt_includes_exemplar_description():
    m = _import_runner()
    desc = "Unique marker string XYZ123"
    case = {"task": "s1", "input": {"state": {"grid_size": [5, 4], "blocks": {}}}}
    exemplar = {
        "task": "s1",
        "input": {"state": {"grid_size": [5, 4], "blocks": {}}},
        "exemplar_output": {"label": "valid"},
        "description": desc,
    }
    prompt = m.build_one_shot_prompt(case, exemplar)
    assert desc in prompt


def test_one_shot_prompt_separates_test_from_example():
    m = _import_runner()
    case = {"task": "s1", "input": {"state": {"grid_size": [5, 4], "blocks": {}}}}
    exemplar = _minimal_exemplar("s1")
    prompt = m.build_one_shot_prompt(case, exemplar)
    # The test case instruction must appear after the example block
    example_end = prompt.index("--- End of Example ---")
    assert "Now solve the following test case" in prompt[example_end:]


def test_one_shot_prompt_includes_klotski_rules():
    m = _import_runner()
    case = {"task": "s1", "input": {"state": {"grid_size": [5, 4], "blocks": {}}}}
    exemplar = _minimal_exemplar("s1")
    prompt = m.build_one_shot_prompt(case, exemplar)
    assert "5 rows by 4 columns" in prompt
    assert "[3, 1]" in prompt


def test_one_shot_prompt_no_oracle_leakage():
    m = _import_runner()
    poisoned_case = {
        "task": "r1",
        "input": {
            "initial_state": {
                "blocks": {"Caocao": {"shape": [2, 2], "pos": [0, 1]}},
                "grid_size": [5, 4],
            }
        },
        "gold": {"optimal_depth": 999, "shortest_solution_actions": [{"block_id": "LEAK"}]},
        "meta": {"optimal_depth": 999, "shortest_solution_actions": [{"block_id": "LEAK"}]},
    }
    exemplar = {
        "task": "r1",
        "input": {"initial_state": {"blocks": {}, "grid_size": [5, 4]}},
        "exemplar_output": {"predicted_trajectory": []},
        "description": "Empty board.",
    }
    prompt = m.build_one_shot_prompt(poisoned_case, exemplar)
    lower = prompt.lower()
    for term in m.PROMPT_LEAKAGE_TERMS:
        assert term not in lower, f"one-shot prompt leaks {term!r}"
    assert "LEAK" not in prompt
    assert "999" not in prompt
    assert '"meta"' not in prompt


# ---------------------------------------------------------------------------
# load_exemplars helper
# ---------------------------------------------------------------------------


def test_load_exemplars_reads_all_tasks(tmp_path: Path):
    m = _import_runner()
    exporter = _import_exporter()

    # Write real exemplars via the exporter
    exporter.export_exemplars(tmp_path)

    exemplars = m.load_exemplars(tmp_path)
    assert set(exemplars.keys()) == set(m.ALL_TASKS)
    for task, ex in exemplars.items():
        assert ex["task"] == task
        assert "input" in ex
        assert "exemplar_output" in ex


def test_load_exemplars_raises_on_missing_file(tmp_path: Path):
    m = _import_runner()
    with pytest.raises(FileNotFoundError):
        m.load_exemplars(tmp_path / "nonexistent")


def test_exported_exemplars_have_required_fields(tmp_path: Path):
    exporter = _import_exporter()
    exporter.export_exemplars(tmp_path)
    for task in ("s1", "s2", "t1", "t2", "t3", "r1", "r2"):
        path = tmp_path / task / "case.json"
        assert path.is_file(), f"Missing exemplar for {task}"
        ex = json.loads(path.read_text(encoding="utf-8"))
        assert ex["task"] == task
        assert "input" in ex
        assert "exemplar_output" in ex
        assert "description" in ex
        assert ex["description"]  # non-empty


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_run_metadata_has_prompt_mode_fields(tmp_path: Path):
    """Zero-shot dry run should include prompt_mode / num_shots in metadata."""
    out = tmp_path / "meta_test"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_mini_ollama_all_tasks.py"),
            "--dry-run",
            "--force-rebuild",
            "--out-dir",
            str(out),
            "--max-cases-per-task",
            "1",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    meta = json.loads((out / "run_metadata_qwen3-4b.json").read_text(encoding="utf-8"))
    assert meta["prompt_mode"] == "zero_shot_rules_v3"
    assert meta["num_shots"] == 0
    assert meta["exemplar_root"] is None


# ---------------------------------------------------------------------------
# End-to-end dry run with one_shot_rules_v3
# ---------------------------------------------------------------------------


def test_dry_run_one_shot_pipeline(tmp_path: Path):
    """Full pipeline dry run with one_shot_rules_v3: creates outputs and metadata."""
    exemplar_dir = tmp_path / "exemplars"
    exporter = _import_exporter()
    exporter.export_exemplars(exemplar_dir)

    out = tmp_path / "dry_run_one_shot"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_mini_ollama_all_tasks.py"),
            "--dry-run",
            "--force-rebuild",
            "--out-dir",
            str(out),
            "--max-cases-per-task",
            "1",
            "--prompt-mode",
            "one_shot_rules_v3",
            "--exemplar-root",
            str(exemplar_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    slug = "qwen3-4b"
    assert (out / f"predictions_{slug}.json").is_file()
    assert (out / f"eval_summary_{slug}.json").is_file()

    meta = json.loads((out / f"run_metadata_{slug}.json").read_text(encoding="utf-8"))
    assert meta["prompt_mode"] == "one_shot_rules_v3"
    assert meta["num_shots"] == 1
    assert meta["exemplar_root"] == str(exemplar_dir)
    assert meta["dry_run"] is True


def test_dry_run_one_shot_requires_exemplar_root():
    """one_shot_rules_v3 without --exemplar-root must exit non-zero."""
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_mini_ollama_all_tasks.py"),
            "--dry-run",
            "--force-rebuild",
            "--max-cases-per-task",
            "1",
            "--prompt-mode",
            "one_shot_rules_v3",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
