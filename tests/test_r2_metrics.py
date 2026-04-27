from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_compute_r2_metrics_basic():
    from src.metrics.r2_metrics import compute_r2_metrics

    cases = [
        {"gold": {"label": "valid"}},
        {
            "gold": {
                "label": "invalid",
                "first_error_step": 2,
                "step_validity_sequence": [1, 0, 0],
            }
        },
    ]
    preds = [
        {"label": "valid"},
        {
            "label": "invalid",
            "first_error_step": 2,
            "step_validity_sequence": [1, 0, 0],
        },
    ]
    metrics = compute_r2_metrics(cases, preds)
    assert metrics["trajectory_verification_accuracy"] == 1.0
    assert metrics["first_error_localization_accuracy"] == 1.0
    assert metrics["step_level_verification_accuracy"] == 1.0
