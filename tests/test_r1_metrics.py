from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


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
    metrics = compute_r1_metrics([case], [pred])
    assert metrics["sr_at_n"][1] == 1.0
    assert metrics["alive_t"][1] == 1.0
    assert metrics["tcr_t"][1] == 1.0
    assert metrics["efficiency"] == 1.0
