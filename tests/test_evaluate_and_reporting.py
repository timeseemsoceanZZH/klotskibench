from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_evaluate_benchmark_dispatches_s1_and_r2():
    from src.evaluate import evaluate_benchmark
    from src.reporting import build_main_task_table
    from src import evaluate as evaluate_mod

    cases = [
        {"task": "s1", "input": {"state": {}}, "gold": {"label": "valid"}},
        {"task": "r2", "gold": {"label": "valid"}},
    ]
    preds = [{"label": "valid"}, {"label": "valid"}]
    out = evaluate_benchmark(cases, preds)
    assert out["results_by_task"]["s1"]["metrics"] is not None
    assert "accuracy" in out["results_by_task"]["s1"]["metrics"]
    assert out["results_by_task"]["r2"]["metrics"] is not None
    assert set(evaluate_mod.TASKS_WITH_METRICS) == {
        "s1",
        "s2",
        "t1",
        "t2",
        "t3",
        "r1",
        "r2",
    }
    table = build_main_task_table(out)
    assert {r["task"] for r in table} == {"r2", "s1"}


def test_build_oracle_state_table():
    from src.reporting import build_oracle_state_table

    cases = [
        {
            "task": "r1",
            "meta": {"canonical_state_key": "k1", "optimal_depth": 2},
            "gold": {"optimal_depth": 2},
        }
    ]
    rows = build_oracle_state_table(cases)
    assert rows[0]["canonical_state_key"] == "k1"
    assert rows[0]["optimal_depth"] == 2
