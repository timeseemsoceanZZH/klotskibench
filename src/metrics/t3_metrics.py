"""Metrics for T3 transition validity (label + reason)."""

from __future__ import annotations

from typing import Any, Sequence


def compute_t3_metrics(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    - ``label_accuracy`` / ``invalid_reason_accuracy`` (on gold-invalid + reason)
    - ``joint_verification_reason_accuracy``: per sample, label must match, and
      for gold ``invalid`` with a ``reason``, the predicted reason must match too;
      for gold ``valid``, only the label is required.
    """
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return {
            "label_accuracy": 0.0,
            "invalid_reason_accuracy": 0.0,
            "joint_verification_reason_accuracy": 0.0,
        }

    n = float(len(cases))
    label_ok = 0
    reason_ok = 0
    reason_n = 0
    joint_ok = 0
    for case, pred in zip(cases, predictions):
        g = case["gold"]
        gl = g.get("label")
        pl = pred.get("label")
        if pl == gl:
            label_ok += 1
        if gl == "invalid" and "reason" in g:
            reason_n += 1
            if _reasons_match(pred.get("reason"), g.get("reason")):
                reason_ok += 1
        if _joint_matches(g, pred):
            joint_ok += 1

    out: dict[str, float] = {
        "label_accuracy": float(label_ok) / n,
        "joint_verification_reason_accuracy": float(joint_ok) / n,
    }
    if reason_n:
        out["invalid_reason_accuracy"] = float(reason_ok) / float(reason_n)
    else:
        out["invalid_reason_accuracy"] = 0.0
    return out


def _reasons_match(pred_reason: Any, gold_reason: Any) -> bool:
    if pred_reason == gold_reason:
        return True
    return (
        gold_reason == "wrong_moved_block_position" and pred_reason == "wrong_transition"
    )


def _joint_matches(g: dict[str, Any], pred: dict[str, Any]) -> bool:
    gl = g.get("label")
    if gl != pred.get("label"):
        return False
    if gl == "valid":
        return True
    if "reason" in g:
        return _reasons_match(pred.get("reason"), g.get("reason"))
    return True


__all__ = [
    "compute_t3_metrics",
]
