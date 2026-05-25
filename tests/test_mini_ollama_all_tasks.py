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


def test_parse_json_pure_object():
    m = _import_runner()
    obj, err = m.parse_json_object('{"label": "invalid"}')
    assert err is None
    assert obj == {"label": "invalid"}


def test_parse_json_fenced_block():
    m = _import_runner()
    raw = 'Here is the answer:\n```json\n{"legal_actions": [{"block_id": "A", "direction": "up"}]}\n```\n'
    obj, err = m.parse_json_object(raw)
    assert err is None
    assert obj["legal_actions"][0]["block_id"] == "A"


def test_parse_json_with_extra_text():
    m = _import_runner()
    raw = 'Sure! {"label": "valid", "note": "x"} is my answer.'
    obj, err = m.parse_json_object(raw)
    assert err is None
    assert obj["label"] == "valid"


def test_normalize_r2_invalid_prediction():
    m = _import_runner()
    pred = m._normalize_prediction(
        "r2",
        {
            "label": "invalid",
            "first_error_step": 3,
            "step_validity_sequence": [1, 1, 0],
        },
    )
    assert pred["label"] == "invalid"
    assert pred["first_error_step"] == 3


def test_prompts_include_klotski_rules_and_r1_goal():
    m = _import_runner()
    case = {
        "task": "r1",
        "input": {
            "initial_state": {
                "blocks": {"CaoCao": {"shape": [2, 2], "pos": [0, 1]}},
                "grid_size": [5, 4],
            }
        },
        "gold": {"optimal_depth": 42},
        "meta": {"shortest_solution_actions": [{"block_id": "SECRET", "direction": "up"}]},
    }
    prompt = m.build_prompt(case)
    assert "5 rows by 4 columns" in prompt
    assert "[3, 1]" in prompt
    assert "reaching a state in G" in prompt
    assert "legal detours" in prompt.lower()
    assert "predicted_trajectory" in prompt
    assert "SECRET" not in prompt
    for term in m.PROMPT_LEAKAGE_TERMS:
        assert term not in prompt.lower()
    assert '"meta"' not in prompt


def test_s2_prompt_requires_localization_fields():
    m = _import_runner()
    prompt = m.build_prompt({"task": "s2", "input": {"state": {"blocks": {}, "grid_size": [5, 4]}}})
    assert "Do not output only error_type" in prompt
    for field in ("block_ids", "cells", "block_id", "missing_block_id"):
        assert field in prompt


def test_r2_prompt_first_error_step_convention():
    m = _import_runner()
    prompt = m.build_prompt(
        {
            "task": "r2",
            "input": {
                "initial_state": {"blocks": {}, "grid_size": [5, 4]},
                "candidate_trajectory": [],
            },
        }
    )
    assert "first_error_step is 1-indexed" in prompt
    assert "first 0 in step_validity_sequence" in prompt
    assert "len(candidate_trajectory) + 1" in prompt


def test_prompts_no_oracle_leakage():
    m = _import_runner()
    outputs = m.acquire_task_outputs(force_rebuild=False)
    cases, _ = m.select_cases(outputs, m.ALL_TASKS, None)
    assert cases, "expected validation cases"
    for case in cases:
        poisoned = {
            **case,
            "gold": {
                "optimal_depth": 999,
                "shortest_solution_actions": [{"block_id": "LEAK", "direction": "up"}],
                "label": "invalid",
            },
            "meta": {
                "optimal_depth": 999,
                "shortest_solution_actions": [{"block_id": "LEAK", "direction": "up"}],
                "goal_state": {"blocks": {}, "grid_size": [5, 4]},
            },
        }
        prompt = m.build_prompt(poisoned)
        lower = prompt.lower()
        for term in m.PROMPT_LEAKAGE_TERMS:
            assert term not in lower, f"{case['task']}: prompt contains {term!r}"
        assert '"meta"' not in prompt
        assert " meta" not in lower and "meta " not in lower
        assert "LEAK" not in prompt
        assert "999" not in prompt


def test_cli_default_decoding_options():
    m = _import_runner()
    args = m.parse_runner_args([])
    assert args.temperature == 0.0
    assert args.seed == 42
    assert args.top_p is None
    assert args.num_predict is None


def test_build_ollama_options_defaults_and_optional():
    m = _import_runner()
    opts = m.build_ollama_options()
    assert opts == {"temperature": 0.0, "seed": 42}
    assert "top_p" not in opts
    assert "num_predict" not in opts

    opts_full = m.build_ollama_options(temperature=0.1, seed=7, top_p=0.9, num_predict=256)
    assert opts_full["temperature"] == 0.1
    assert opts_full["seed"] == 7
    assert opts_full["top_p"] == 0.9
    assert opts_full["num_predict"] == 256


def test_ollama_chat_payload_includes_options():
    m = _import_runner()
    import mini_bench_run_ollama as ollama  # noqa: WPS433

    opts = m.build_ollama_options()
    body = ollama.build_ollama_chat_request_body("qwen3:4b", "hello", options=opts)
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["seed"] == 42

    body_minimal = ollama.build_ollama_chat_request_body("m", "hi", options=None)
    assert "options" not in body_minimal

    body_extra = ollama.build_ollama_chat_request_body(
        "m",
        "hi",
        options=m.build_ollama_options(top_p=0.95, num_predict=128),
    )
    assert body_extra["options"]["top_p"] == 0.95
    assert body_extra["options"]["num_predict"] == 128


def _load_legacy_klotski_state():
    import importlib.util

    path = ROOT / "legacy" / "klotski_state.py"
    spec = importlib.util.spec_from_file_location("klotski_state_real", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_t2_malformed_predictions_sanitized():
    m = _import_runner()
    legacy_state = _load_legacy_klotski_state()

    missing_grid, flag_grid = m.sanitize_prediction_for_task(
        "t2",
        {"next_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 0]}}}},
    )
    assert flag_grid is True
    assert missing_grid == {"next_state": m.T2_SAFE_NEXT_STATE}

    missing_ns, flag_ns = m.sanitize_prediction_for_task("t2", {})
    assert flag_ns is True
    assert missing_ns == {"next_state": m.T2_SAFE_NEXT_STATE}

    with pytest.raises(KeyError):
        legacy_state.normalize_state({"blocks": {}})
    legacy_state.normalize_state(missing_grid["next_state"])

    outputs = m.acquire_task_outputs(force_rebuild=False)
    good_ns = dict(outputs["t2"]["cases"][0]["gold"]["next_state"])
    good_pred, sanitized = m.sanitize_prediction_for_task(
        "t2",
        {"next_state": good_ns},
    )
    assert sanitized is False
    assert good_pred == {"next_state": good_ns}


def test_malformed_t2_without_grid_size_sanitized_before_eval_path():
    m = _import_runner()
    legacy_state = _load_legacy_klotski_state()
    malformed = m._normalize_prediction("t2", {"next_state": {"blocks": {}}})
    pred, sanitized = m.sanitize_prediction_for_task("t2", malformed)
    assert sanitized is True
    assert pred["next_state"]["grid_size"] == [5, 4]
    assert "blocks" in pred["next_state"]
    legacy_state.normalize_state(pred["next_state"])


def test_dry_run_pipeline(tmp_path: Path):
    out = tmp_path / "dry_run_test"
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
    slug = "qwen3-4b"
    assert (out / f"predictions_{slug}.json").is_file()
    assert (out / f"eval_summary_{slug}.json").is_file()
    preds = json.loads((out / f"predictions_{slug}.json").read_text(encoding="utf-8"))
    cases_path = out / "run_metadata_qwen3-4b.json"
    meta = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(preds["predictions"]) == meta["n_cases"]
    assert meta["dry_run"] is True
    assert meta["temperature"] == 0.0
    assert meta["seed"] == 42
    assert meta["top_p"] is None
    assert meta["num_predict"] is None
