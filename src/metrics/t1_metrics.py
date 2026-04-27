"""Metrics for T1 legal action-set prediction (methodology v1)."""

from __future__ import annotations

from typing import Any, Sequence


def compute_t1_metrics(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    - ``exact_set_match`` (headline): set equality of legal moves
    - ``move_precision`` / ``move_recall`` per instance (sets of
      ``(block_id, direction)``), then macro-averaged over the dataset
    """
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return {
            "exact_set_match": 0.0,
            "move_precision": 0.0,
            "move_recall": 0.0,
        }

    exact = 0
    prec_sum = 0.0
    rec_sum = 0.0
    for case, pred in zip(cases, predictions):
        gold_acts = case["gold"]["legal_actions"]
        pred_acts = pred.get("legal_actions", pred.get("predicted_legal_actions", []))
        gset = _action_set(gold_acts)
        pset = _action_set(pred_acts) if isinstance(pred_acts, list) else set()
        if gset == pset:
            exact += 1
        prec, rec = _set_precision_recall(gset, pset)
        prec_sum += prec
        rec_sum += rec

    n = float(len(cases))
    return {
        "exact_set_match": float(exact) / n,
        "move_precision": prec_sum / n,
        "move_recall": rec_sum / n,
    }


def _set_precision_recall(
    gold: set[tuple[str, str]],
    pred: set[tuple[str, str]],
) -> tuple[float, float]:
    if not pred and not gold:
        return 1.0, 1.0
    if not pred:
        return 0.0, 0.0
    if not gold:
        return 0.0, 0.0
    inter = len(gold & pred)
    return float(inter) / float(len(pred)), float(inter) / float(len(gold))


def _action_set(actions: list[dict[str, str]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for a in actions:
        if not isinstance(a, dict):
            continue
        out.add((str(a["block_id"]), str(a["direction"])))
    return out


__all__ = [
    "compute_t1_metrics",
]
