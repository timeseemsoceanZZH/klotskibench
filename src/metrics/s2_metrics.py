"""Metrics for S2 error localization (block_id + error_type)."""

from __future__ import annotations

from typing import Any, Sequence


def compute_s2_metrics(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    - ``exact_error_match`` (headline): list equality on structured errors
    - ``location_accuracy``: primary error ``block_id`` match
    - ``error_type_accuracy``: primary error ``error_type`` match
    """
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return {
            "exact_error_match": 0.0,
            "location_accuracy": 0.0,
            "error_type_accuracy": 0.0,
        }

    n = len(cases)
    exact = 0
    loc_ok = 0
    type_ok = 0
    for case, pred in zip(cases, predictions):
        gold = case["gold"]["errors"]
        pred_errors = pred.get("errors")
        g0 = gold[0] if gold and isinstance(gold, list) else None
        p0 = pred_errors[0] if isinstance(pred_errors, list) and pred_errors else None
        if g0 and isinstance(p0, dict):
            if str(g0.get("block_id")) == str(p0.get("block_id")):
                loc_ok += 1
            if str(g0.get("error_type")) == str(p0.get("error_type")):
                type_ok += 1
        if not isinstance(pred_errors, list):
            continue
        if len(gold) != len(pred_errors):
            continue
        match = True
        for ge, pe in zip(gold, pred_errors):
            if not isinstance(pe, dict):
                match = False
                break
            if str(ge.get("block_id")) != str(pe.get("block_id")):
                match = False
                break
            if str(ge.get("error_type")) != str(pe.get("error_type")):
                match = False
                break
        if match:
            exact += 1

    nf = float(n)
    return {
        "exact_error_match": float(exact) / nf,
        "location_accuracy": float(loc_ok) / nf,
        "error_type_accuracy": float(type_ok) / nf,
    }


__all__ = [
    "compute_s2_metrics",
]
