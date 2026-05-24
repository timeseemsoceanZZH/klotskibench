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
