"""Metrics for R2 trajectory verification tasks."""

from __future__ import annotations

from typing import Any, Sequence


def compute_r2_metrics(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Compute R2 metrics from gold cases and model predictions."""
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return {
            "trajectory_verification_accuracy": 0.0,
            "first_error_localization_accuracy": 0.0,
            "step_level_verification_accuracy": 0.0,
        }

    label_correct = 0
    invalid_total = 0
    invalid_first_error_correct = 0
    step_correct = 0
    step_total = 0

    for case, pred in zip(cases, predictions):
        gold = case["gold"]
        pred_label = pred.get("label")
        gold_label = gold["label"]
        if pred_label == gold_label:
            label_correct += 1

        if gold_label == "invalid":
            invalid_total += 1
            if pred.get("first_error_step") == gold.get("first_error_step"):
                invalid_first_error_correct += 1

            gold_seq = list(gold.get("step_validity_sequence", []))
            pred_seq = list(pred.get("step_validity_sequence", []))
            comp_len = max(len(gold_seq), len(pred_seq))
            for idx in range(comp_len):
                gv = gold_seq[idx] if idx < len(gold_seq) else None
                pv = pred_seq[idx] if idx < len(pred_seq) else None
                if gv == pv:
                    step_correct += 1
                step_total += 1

    first_error_acc = (
        float(invalid_first_error_correct) / float(invalid_total) if invalid_total else 0.0
    )
    step_acc = float(step_correct) / float(step_total) if step_total else 0.0
    return {
        "trajectory_verification_accuracy": float(label_correct) / float(len(cases)),
        "first_error_localization_accuracy": first_error_acc,
        "step_level_verification_accuracy": step_acc,
    }


__all__ = [
    "compute_r2_metrics",
]
