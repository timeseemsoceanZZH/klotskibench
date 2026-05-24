from __future__ import annotations

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
    return out


def test_compute_r1_metrics_strict_success():
    from src.metrics.r1_metrics import compute_r1_metrics

    case = {
        "input": {"initial_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 0]}}, "grid_size": [5, 4]}},
        "gold": {"optimal_depth": 1},
        "meta": {
            "goal_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 1]}}, "grid_size": [5, 4]},
            "shortest_solution_actions": [{"block_id": "A", "direction": "right"}],
            "shortest_solution_states": [],
        },
    }
    pred = {"predicted_trajectory": [{"block_id": "A", "direction": "right"}]}
    with (
        patch("src.metrics.r1_metrics.is_solved_state", side_effect=_is_solved_stub),
        patch("src.metrics.r1_metrics.moves_adapter.get_legal_actions", _legal_actions_stub),
        patch("src.metrics.r1_metrics.moves_adapter.apply_action", _apply_action_stub),
    ):
        metrics = compute_r1_metrics([case], [pred])
    assert metrics["sr_at_n"][1] == 1.0
    assert metrics["alive_t"][1] == 1.0
    assert metrics["tcr_t"][1] == 1.0
    assert metrics["efficiency"] == 1.0
