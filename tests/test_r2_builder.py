from __future__ import annotations

import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_build_r2_cases_has_valid_and_invalid_labels():
    builder = importlib.import_module("src.tasks.r2_builder")

    exact_depth_cases = [
        {
            "initial_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 0]}}, "grid_size": [5, 4]},
            "optimal_depth": 1,
            "shortest_solution_actions": [{"block_id": "A", "direction": "right"}],
            "meta": {"canonical_state_key": ("A", 0, 0)},
        }
    ]

    cases = builder.build_r2_cases(
        exact_depth_cases=exact_depth_cases,
        include_valid=True,
        invalid_types=("wrong_action",),
        seed=1,
    )
    assert cases
    labels = [c["gold"]["label"] for c in cases]
    assert "valid" in labels
    assert "invalid" in labels
    invalid_case = next(c for c in cases if c["gold"]["label"] == "invalid")
    assert "first_error_step" in invalid_case["gold"]
    assert "step_validity_sequence" in invalid_case["gold"]
