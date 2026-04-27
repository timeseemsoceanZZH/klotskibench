"""Transition oracle for one-step Klotski state transitions."""

from __future__ import annotations

import copy
from typing import Any

from src.core import moves_adapter, state_adapter

TransitionLabel = dict[str, str]


def get_next_state(
    current_state: dict[str, Any],
    action: dict[str, str],
) -> state_adapter.KlotskiState:
    """Return gold next state for a legal one-step transition."""
    normalized = state_adapter.normalize_state(current_state)
    if not state_adapter.validate_state(normalized):
        raise ValueError("current_state must be valid")
    return moves_adapter.apply_action_checked(normalized, action)


def evaluate_candidate_transition(
    current_state: dict[str, Any],
    action: dict[str, str],
    candidate_next_state: dict[str, Any],
) -> TransitionLabel:
    """Classify candidate transition as valid or invalid with v1 reason."""
    normalized_current = state_adapter.normalize_state(current_state)
    normalized_candidate = state_adapter.normalize_state(candidate_next_state)
    expected_next_state = get_next_state(normalized_current, action)

    if (
        state_adapter.canonicalize_state(normalized_candidate)
        == state_adapter.canonicalize_state(expected_next_state)
    ):
        return {"label": "valid"}

    if _has_boundary_violation(normalized_candidate):
        return {"label": "invalid", "reason": "boundary"}
    if _has_overlap_violation(normalized_candidate):
        return {"label": "invalid", "reason": "overlap"}

    if _has_stationary_object_drift(normalized_current, action["block_id"], normalized_candidate):
        return {"label": "invalid", "reason": "stationary_object_drift"}

    return {"label": "invalid", "reason": "wrong_transition"}


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
    candidate_next_state: state_adapter.KlotskiState,
) -> bool:
    for block_id, block in current_state["blocks"].items():
        if block_id == action_block_id:
            continue
        candidate_block = candidate_next_state["blocks"].get(block_id)
        if candidate_block is None:
            return True
        if list(block["pos"]) != list(candidate_block["pos"]):
            return True
        if list(block["shape"]) != list(candidate_block["shape"]):
            return True
    return False


def make_invalid_candidate_next_state(
    current_state: dict[str, Any],
    action: dict[str, str],
    invalid_reason: str,
) -> state_adapter.KlotskiState:
    """Build a synthetic invalid candidate next state for T3."""
    normalized_current = state_adapter.normalize_state(current_state)
    expected_next = get_next_state(normalized_current, action)

    if invalid_reason == "boundary":
        return _make_boundary_candidate(expected_next, action["block_id"])
    if invalid_reason == "overlap":
        return _make_overlap_candidate(expected_next, action["block_id"])
    if invalid_reason == "stationary_object_drift":
        candidate = _make_stationary_drift_candidate(normalized_current, expected_next, action["block_id"])
        if candidate is not None:
            return candidate
    if invalid_reason == "wrong_transition":
        candidate = _make_wrong_transition_candidate(normalized_current, action, expected_next)
        if candidate is not None:
            return candidate

    raise ValueError(f"Could not synthesize invalid candidate for reason: {invalid_reason}")


def _make_boundary_candidate(
    expected_next: state_adapter.KlotskiState,
    action_block_id: str,
) -> state_adapter.KlotskiState:
    candidate = copy.deepcopy(expected_next)
    rows = int(candidate["grid_size"][0])
    block = candidate["blocks"][action_block_id]
    h = int(block["shape"][0])
    block["pos"][0] = rows - h + 1
    return candidate


def _make_overlap_candidate(
    expected_next: state_adapter.KlotskiState,
    action_block_id: str,
) -> state_adapter.KlotskiState:
    candidate = copy.deepcopy(expected_next)
    for block_id, block in candidate["blocks"].items():
        if block_id == action_block_id:
            continue
        candidate["blocks"][action_block_id]["pos"] = list(block["pos"])
        return candidate
    raise ValueError("Need at least two blocks to synthesize overlap")


def _make_stationary_drift_candidate(
    current_state: state_adapter.KlotskiState,
    expected_next: state_adapter.KlotskiState,
    action_block_id: str,
) -> state_adapter.KlotskiState | None:
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for block_id in current_state["blocks"].keys():
        if block_id == action_block_id:
            continue
        for dr, dc in deltas:
            candidate = copy.deepcopy(expected_next)
            candidate["blocks"][block_id]["pos"][0] += dr
            candidate["blocks"][block_id]["pos"][1] += dc
            if not state_adapter.validate_state(candidate):
                continue
            if (
                state_adapter.canonicalize_state(candidate)
                == state_adapter.canonicalize_state(expected_next)
            ):
                continue
            return candidate
    return None


def _make_wrong_transition_candidate(
    current_state: state_adapter.KlotskiState,
    action: dict[str, str],
    expected_next: state_adapter.KlotskiState,
) -> state_adapter.KlotskiState | None:
    legal_actions = moves_adapter.get_legal_actions(current_state)
    for other_action in legal_actions:
        if other_action["block_id"] != action["block_id"]:
            continue
        if other_action["direction"] == action["direction"]:
            continue
        candidate = moves_adapter.apply_action(current_state, other_action)
        if (
            state_adapter.canonicalize_state(candidate)
            == state_adapter.canonicalize_state(expected_next)
        ):
            continue
        return candidate
    return None


__all__ = [
    "get_next_state",
    "evaluate_candidate_transition",
    "make_invalid_candidate_next_state",
]
