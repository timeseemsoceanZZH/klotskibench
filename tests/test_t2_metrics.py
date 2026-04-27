from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_compute_t2_metrics_basic():
    from src.metrics.t2_metrics import compute_t2_metrics

    case = {
        "input": {
            "current_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 0]}}, "grid_size": [5, 4]},
            "action": {"block_id": "A", "direction": "right"},
        },
        "gold": {"next_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 1]}}, "grid_size": [5, 4]}},
    }
    pred = {"next_state": {"blocks": {"A": {"shape": [1, 1], "pos": [0, 1]}}, "grid_size": [5, 4]}}
    metrics = compute_t2_metrics([case], [pred])
    assert metrics["next_state_exact_match"] == 1.0
    assert metrics["tvr"] == 1.0
    assert metrics["ovr"] == 0.0
    assert metrics["bvr"] == 0.0
