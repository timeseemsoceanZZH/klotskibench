from __future__ import annotations

import pathlib
import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _initial_state() -> dict:
    return {"blocks": {"A": {"shape": [1, 1], "pos": [0, 0]}}, "grid_size": [5, 4]}


def _packaged_goal_state() -> dict:
    return {"blocks": {"A": {"shape": [1, 1], "pos": [0, 1]}}, "grid_size": [5, 4]}


def _alternate_solved_state() -> dict:
    return {"blocks": {"A": {"shape": [1, 1], "pos": [0, 2]}}, "grid_size": [5, 4]}


def _is_solved_stub(state: dict) -> bool:
    return int(state["blocks"]["A"]["pos"][1]) >= 1


def _legal_actions_stub(state: dict) -> list[dict[str, str]]:
    c = int(state["blocks"]["A"]["pos"][1])
    actions: list[dict[str, str]] = []
    if c < 3:
        actions.append({"block_id": "A", "direction": "right"})
    if c > 0:
        actions.append({"block_id": "A", "direction": "left"})
    return actions


def _apply_action_stub(state: dict, action: dict[str, str]) -> dict:
    out = {
        "blocks": {"A": {"shape": [1, 1], "pos": list(state["blocks"]["A"]["pos"])}},
        "grid_size": [5, 4],
    }
    if action["direction"] == "right":
        out["blocks"]["A"]["pos"][1] += 1
    elif action["direction"] == "left":
        out["blocks"]["A"]["pos"][1] -= 1
    return out


@pytest.fixture(autouse=True)
def r1_env_patch():
    targets = (
        "src.core.goal.is_solved_state",
        "src.metrics.r1_metrics.is_solved_state",
        "src.core.moves_adapter.get_legal_actions",
        "src.core.moves_adapter.apply_action",
        "src.metrics.r1_metrics.moves_adapter.get_legal_actions",
        "src.metrics.r1_metrics.moves_adapter.apply_action",
    )
    with ExitStack() as stack:
        stack.enter_context(patch("src.core.goal.is_solved_state", side_effect=_is_solved_stub))
        stack.enter_context(
            patch("src.metrics.r1_metrics.is_solved_state", side_effect=_is_solved_stub)
        )
        stack.enter_context(
            patch("src.core.moves_adapter.get_legal_actions", side_effect=_legal_actions_stub)
        )
        stack.enter_context(
            patch("src.metrics.r1_metrics.moves_adapter.get_legal_actions", side_effect=_legal_actions_stub)
        )
        stack.enter_context(
            patch("src.core.moves_adapter.apply_action", side_effect=_apply_action_stub)
        )
        stack.enter_context(
            patch("src.metrics.r1_metrics.moves_adapter.apply_action", side_effect=_apply_action_stub)
        )
        yield


def _base_case() -> dict:
    return {
        "input": {"initial_state": _initial_state()},
        "gold": {"optimal_depth": 1},
        "meta": {
            "goal_state": _packaged_goal_state(),
            "shortest_solution_actions": [{"block_id": "A", "direction": "right"}],
        },
    }


def test_r1_success_without_packaged_goal_layout_match(r1_env_patch):
    from src.metrics.r1_metrics import compute_r1_metrics

    case = _base_case()
    pred = {"predicted_trajectory": [{"block_id": "A", "direction": "right"}, {"block_id": "A", "direction": "right"}]}
    metrics = compute_r1_metrics([case], [pred])
    assert metrics["sr_at_n"][1] == 1.0
    assert case["meta"]["goal_state"]["blocks"]["A"]["pos"] == [0, 1]
    assert _alternate_solved_state()["blocks"]["A"]["pos"] == [0, 2]


def test_r1_legal_detour_success_lower_efficiency(r1_env_patch):
    from src.metrics.r1_metrics import compute_r1_metrics

    case = _base_case()
    detour = [
        {"block_id": "A", "direction": "right"},
        {"block_id": "A", "direction": "left"},
        {"block_id": "A", "direction": "right"},
    ]
    metrics = compute_r1_metrics([case], [{"predicted_trajectory": detour}])
    assert metrics["sr_at_n"][1] == 1.0
    assert metrics["efficiency"] == pytest.approx(1.0 / 3.0)
    assert metrics["tcr_t"][1] == 1.0
    assert metrics["tcr_t"][2] == 0.0


def test_r1_reach_goal_then_leave_not_success(r1_env_patch):
    from src.metrics.r1_metrics import compute_r1_metrics

    case = _base_case()
    trajectory = [
        {"block_id": "A", "direction": "right"},
        {"block_id": "A", "direction": "left"},
    ]
    metrics = compute_r1_metrics([case], [{"predicted_trajectory": trajectory}])
    assert metrics["sr_at_n"][1] == 0.0


def test_r1_step_budget_enforced_when_set(r1_env_patch):
    from src.metrics.r1_metrics import compute_r1_metrics

    case = _base_case()
    case["meta"]["step_budget_n"] = 1
    detour = [
        {"block_id": "A", "direction": "right"},
        {"block_id": "A", "direction": "left"},
        {"block_id": "A", "direction": "right"},
    ]
    metrics = compute_r1_metrics([case], [{"predicted_trajectory": detour}])
    assert metrics["sr_at_n"][1] == 0.0
