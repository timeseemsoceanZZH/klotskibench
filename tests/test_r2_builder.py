from __future__ import annotations

import importlib
import pathlib
import sys
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _is_solved_stub(state: dict) -> bool:
    return state["blocks"]["A"]["pos"] == [0, 1]


def _legal_actions_stub(state: dict) -> list[dict[str, str]]:
    c = state["blocks"]["A"]["pos"][1]
    actions = []
    if c < 3:
        actions.append({"block_id": "A", "direction": "right"})
    return actions


def _apply_action_stub(state: dict, action: dict[str, str]) -> dict:
    out = {
        "blocks": {"A": {"shape": [1, 1], "pos": list(state["blocks"]["A"]["pos"])}},
        "grid_size": [5, 4],
    }
    if action["direction"] == "right":
        out["blocks"]["A"]["pos"][1] += 1
    return out


def test_build_r2_cases_has_valid_and_invalid_labels():
    builder = importlib.import_module("src.tasks.r2_builder")

    exact_depth_cases = [
        {
            "initial_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 0]}}, "grid_size": [5, 4]},
            "goal_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 1]}}, "grid_size": [5, 4]},
            "optimal_depth": 1,
            "shortest_solution_actions": [{"block_id": "A", "direction": "right"}],
            "meta": {"canonical_state_key": ("A", 0, 0)},
        }
    ]

    with (
        patch("src.tasks.r2_builder.is_solved_state", side_effect=_is_solved_stub),
        patch("src.tasks.r2_builder.moves_adapter.get_legal_actions", _legal_actions_stub),
        patch("src.tasks.r2_builder.moves_adapter.apply_action", _apply_action_stub),
    ):
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
