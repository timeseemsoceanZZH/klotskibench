"""Metrics for T2 next-state prediction tasks."""

from __future__ import annotations

from typing import Any, Sequence

from src.core import moves_adapter, state_adapter


def evaluate_t2_sample(
    case: dict[str, Any],
    predicted_next_state: dict[str, Any],
) -> dict[str, float]:
    """Evaluate one T2 sample and return sample-level indicator metrics."""
    current_state = state_adapter.normalize_state(case["input"]["current_state"])
    action = dict(case["input"]["action"])
    gold_next_state = state_adapter.normalize_state(case["gold"]["next_state"])
    predicted = state_adapter.normalize_state(predicted_next_state)

    exact_match = float(
        state_adapter.canonicalize_state(predicted)
        == state_adapter.canonicalize_state(gold_next_state)
    )
    ovr = float(_has_overlap_violation(predicted))
    bvr = float(_has_boundary_violation(predicted))
    sodr = float(_has_stationary_object_drift(current_state, action["block_id"], predicted))
    tvr = float(_is_true_legal_successor(current_state, action, predicted))

    return {
        "next_state_exact_match": exact_match,
        "ovr": ovr,
        "bvr": bvr,
        "sodr": sodr,
        "tvr": tvr,
    }


def compute_t2_metrics(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Compute aggregate T2 metrics over a dataset."""
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return {
            "next_state_exact_match": 0.0,
            "ovr": 0.0,
            "bvr": 0.0,
            "sodr": 0.0,
            "tvr": 0.0,
        }

    totals = {
        "next_state_exact_match": 0.0,
        "ovr": 0.0,
        "bvr": 0.0,
        "sodr": 0.0,
        "tvr": 0.0,
    }
    for case, pred in zip(cases, predictions):
        predicted_next_state = pred.get("next_state", pred)
        sample = evaluate_t2_sample(case, predicted_next_state)
        for key in totals:
            totals[key] += sample[key]

    count = float(len(cases))
    return {k: v / count for k, v in totals.items()}


def _is_true_legal_successor(
    current_state: state_adapter.KlotskiState,
    action: dict[str, str],
    predicted_next_state: state_adapter.KlotskiState,
) -> bool:
    try:
        expected = moves_adapter.apply_action_checked(current_state, action)
    except Exception:
        return False
    return (
        state_adapter.canonicalize_state(expected)
        == state_adapter.canonicalize_state(predicted_next_state)
    )


def _has_boundary_violation(state: state_adapter.KlotskiState) -> bool:
    rows = int(state["grid_size"][0])
    cols = int(state["grid_size"][1])
    for block in state["blocks"].values():
        r = int(block["pos"][0])
        c = int(block["pos"][1])
        h = int(block["shape"][0])
        w = int(block["shape"][1])
        if r < 0 or c < 0 or r + h > rows or c + w > cols:
            return True
    return False


def _has_overlap_violation(state: state_adapter.KlotskiState) -> bool:
    occupied: set[tuple[int, int]] = set()
    for block in state["blocks"].values():
        r = int(block["pos"][0])
        c = int(block["pos"][1])
        h = int(block["shape"][0])
        w = int(block["shape"][1])
        for dr in range(h):
            for dc in range(w):
                cell = (r + dr, c + dc)
                if cell in occupied:
                    return True
                occupied.add(cell)
    return False


def _has_stationary_object_drift(
    current_state: state_adapter.KlotskiState,
    action_block_id: str,
    predicted_next_state: state_adapter.KlotskiState,
) -> bool:
    for block_id, current_block in current_state["blocks"].items():
        if block_id == action_block_id:
            continue
        pred_block = predicted_next_state["blocks"].get(block_id)
        if pred_block is None:
            return True
        if list(current_block["pos"]) != list(pred_block["pos"]):
            return True
        if list(current_block["shape"]) != list(pred_block["shape"]):
            return True
    return False


__all__ = [
    "evaluate_t2_sample",
    "compute_t2_metrics",
]
