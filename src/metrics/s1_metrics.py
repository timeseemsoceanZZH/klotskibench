"""Metrics for S1 state validity classification."""

from __future__ import annotations

from typing import Any, Sequence


def compute_s1_metrics(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    - ``accuracy`` overall
    - ``precision_invalid`` / ``recall_invalid`` with **invalid** as the positive class
    """
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return {
            "accuracy": 0.0,
            "precision_invalid": 0.0,
            "recall_invalid": 0.0,
        }

    n = float(len(cases))
    correct = 0
    tp = fp = fn = 0
    for case, pred in zip(cases, predictions):
        gold = case["gold"]["label"]
        plabel = _extract_label(pred)
        is_invalid_gold = gold == "invalid"
        is_invalid_pred = plabel == "invalid" if plabel is not None else None

        if plabel is not None and plabel == gold:
            correct += 1
        if is_invalid_pred and is_invalid_gold:
            tp += 1
        elif is_invalid_pred and not is_invalid_gold:
            fp += 1
        elif (not is_invalid_pred) and is_invalid_gold:
            fn += 1

    prec_denom = tp + fp
    rec_denom = tp + fn
    return {
        "accuracy": float(correct) / n,
        "precision_invalid": (float(tp) / float(prec_denom)) if prec_denom else 0.0,
        "recall_invalid": (float(tp) / float(rec_denom)) if rec_denom else 0.0,
    }


def _extract_label(pred: dict[str, Any]) -> str | None:
    if "label" in pred:
        return str(pred["label"])
    return None


__all__ = [
    "compute_s1_metrics",
]
