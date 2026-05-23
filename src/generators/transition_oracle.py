"""Transition oracle for one-step Klotski state transitions."""

from __future__ import annotations

import copy
from typing import Any

from src.core import moves_adapter, state_adapter

TransitionLabel = dict[str, str]

T3_PAPER_INVALID_REASONS: tuple[str, ...] = (
    "overlap",
    "boundary",
    "stationary_object_drift",
    "shape_change",
    "wrong_moved_block_position",
)

# Backward-compatible alias accepted at synthesis/build time only.
T3_INVALID_REASON_ALIASES: dict[str, str] = {
    "wrong_transition": "wrong_moved_block_position",
}


def normalize_t3_invalid_reason(reason: str) -> str:
    """Map legacy reason names to paper-facing T3 reasons."""
    return T3_INVALID_REASON_ALIASES.get(reason, reason)


def is_supported_t3_invalid_reason(reason: str) -> bool:
    """Return whether ``reason`` is a paper or legacy alias invalid reason."""
    normalized = normalize_t3_invalid_reason(reason)
    return normalized in T3_PAPER_INVALID_REASONS or reason in T3_INVALID_REASON_ALIASES


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
    """Classify candidate transition as valid or invalid with a paper T3 reason."""
    normalized_current = state_adapter.normalize_state(current_state)
    normalized_candidate = state_adapter.normalize_state(candidate_next_state)
    expected_next_state = get_next_state(normalized_current, action)

    if (
        state_adapter.canonicalize_state(normalized_candidate)
        == state_adapter.canonicalize_state(expected_next_state)
    ):
        return {"label": "valid"}

    reason = _classify_invalid_reason(
        normalized_current,
        action["block_id"],
        normalized_candidate,
        expected_next_state,
    )
    return {"label": "invalid", "reason": reason}


def _classify_invalid_reason(
    current_state: state_adapter.KlotskiState,
    action_block_id: str,
    candidate_next_state: state_adapter.KlotskiState,
    expected_next_state: state_adapter.KlotskiState,
) -> str:
    """Paper priority: overlap, boundary, shape_change, stationary drift, wrong move."""
    if _has_overlap_violation(candidate_next_state):
        return "overlap"
    if _has_boundary_violation(candidate_next_state):
        return "boundary"
    if _has_stationary_shape_change(current_state, action_block_id, candidate_next_state):
        return "shape_change"
    if _has_stationary_position_drift(current_state, action_block_id, candidate_next_state):
        return "stationary_object_drift"
    return "wrong_moved_block_position"


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


def _has_stationary_shape_change(
    current_state: state_adapter.KlotskiState,
    action_block_id: str,
    candidate_next_state: state_adapter.KlotskiState,
) -> bool:
    for block_id, block in current_state["blocks"].items():
        if block_id == action_block_id:
            continue
        candidate_block = candidate_next_state["blocks"].get(block_id)
        if candidate_block is None:
            continue
        if list(block["shape"]) != list(candidate_block["shape"]):
            return True
    return False


def _has_stationary_position_drift(
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
    return False


def make_invalid_candidate_next_state(
    current_state: dict[str, Any],
    action: dict[str, str],
    invalid_reason: str,
) -> state_adapter.KlotskiState:
    """Build a synthetic invalid candidate next state for T3."""
    normalized_reason = normalize_t3_invalid_reason(invalid_reason)
    if normalized_reason not in T3_PAPER_INVALID_REASONS:
        raise ValueError(f"Unsupported invalid reason: {invalid_reason}")

    normalized_current = state_adapter.normalize_state(current_state)
    expected_next = get_next_state(normalized_current, action)

    if normalized_reason == "boundary":
        return _make_boundary_candidate(expected_next, action["block_id"])
    if normalized_reason == "overlap":
        return _make_overlap_candidate(expected_next, action["block_id"])
    if normalized_reason == "shape_change":
        candidate = _make_shape_change_candidate(
            normalized_current, expected_next, action["block_id"]
        )
        if candidate is not None:
            return candidate
    if normalized_reason == "stationary_object_drift":
        candidate = _make_stationary_drift_candidate(
            normalized_current, expected_next, action["block_id"]
        )
        if candidate is not None:
            return candidate
    if normalized_reason == "wrong_moved_block_position":
        candidate = _make_wrong_moved_block_candidate(
            normalized_current, action, expected_next
        )
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
    for block_id in candidate["blocks"]:
        if block_id == action_block_id:
            continue
        candidate["blocks"][action_block_id]["pos"] = list(
            candidate["blocks"][block_id]["pos"]
        )
        return candidate
    raise ValueError("Need at least two blocks to synthesize overlap")


def _successor_layout(
    current_state: state_adapter.KlotskiState,
    expected_next: state_adapter.KlotskiState,
    action_block_id: str,
) -> state_adapter.KlotskiState:
    """Merge current and expected next so stationary blocks are present for edits."""
    layout = copy.deepcopy(current_state)
    for block_id, block in expected_next["blocks"].items():
        layout["blocks"][block_id] = copy.deepcopy(block)
    layout["grid_size"] = list(expected_next["grid_size"])
    if action_block_id not in layout["blocks"]:
        moved = expected_next["blocks"].get(action_block_id)
        if moved is not None:
            layout["blocks"][action_block_id] = copy.deepcopy(moved)
    return layout


def _make_shape_change_candidate(
    current_state: state_adapter.KlotskiState,
    expected_next: state_adapter.KlotskiState,
    action_block_id: str,
) -> state_adapter.KlotskiState | None:
    """Change a stationary block's shape on the expected layout without moving it."""
    rows, cols = int(current_state["grid_size"][0]), int(current_state["grid_size"][1])
    block_ids = [bid for bid in current_state["blocks"] if bid != action_block_id]

    for block_id in block_ids:
        block = current_state["blocks"][block_id]
        h, w = int(block["shape"][0]), int(block["shape"][1])
        shape_attempts: list[tuple[int, int]] = []
        for nh in range(1, rows + 1):
            for nw in range(1, cols + 1):
                if (nh, nw) != (h, w):
                    shape_attempts.append((nh, nw))

        for nh, nw in shape_attempts:
            candidate = _successor_layout(current_state, expected_next, action_block_id)
            candidate["blocks"][block_id]["shape"] = [nh, nw]
            if (
                state_adapter.canonicalize_state(candidate)
                == state_adapter.canonicalize_state(expected_next)
            ):
                continue
            reason = _classify_invalid_reason(
                current_state,
                action_block_id,
                candidate,
                expected_next,
            )
            if reason == "shape_change":
                return candidate
    return None


def _make_stationary_drift_candidate(
    current_state: state_adapter.KlotskiState,
    expected_next: state_adapter.KlotskiState,
    action_block_id: str,
) -> state_adapter.KlotskiState | None:
    """Nudge a stationary block's position only (shape unchanged)."""
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for block_id in current_state["blocks"]:
        if block_id == action_block_id:
            continue
        for dr, dc in deltas:
            candidate = _successor_layout(current_state, expected_next, action_block_id)
            candidate["blocks"][block_id]["pos"][0] += dr
            candidate["blocks"][block_id]["pos"][1] += dc
            if (
                state_adapter.canonicalize_state(candidate)
                == state_adapter.canonicalize_state(expected_next)
            ):
                continue
            reason = _classify_invalid_reason(
                current_state,
                action_block_id,
                candidate,
                expected_next,
            )
            if reason == "stationary_object_drift":
                return candidate
    return None


def _make_wrong_moved_block_candidate(
    current_state: state_adapter.KlotskiState,
    action: dict[str, str],
    expected_next: state_adapter.KlotskiState,
) -> state_adapter.KlotskiState | None:
    """Place the moved block at a wrong position while keeping stationary blocks fixed."""
    action_block_id = action["block_id"]
    legal_actions = moves_adapter.get_legal_actions(current_state)
    for other_action in legal_actions:
        if other_action["block_id"] != action_block_id:
            continue
        if other_action["direction"] == action["direction"]:
            continue
        candidate = moves_adapter.apply_action_checked(current_state, other_action)
        if (
            state_adapter.canonicalize_state(candidate)
            == state_adapter.canonicalize_state(expected_next)
        ):
            continue
        reason = _classify_invalid_reason(
            current_state,
            action_block_id,
            candidate,
            expected_next,
        )
        if reason == "wrong_moved_block_position":
            return candidate

    expected_block = expected_next["blocks"][action_block_id]
    base_pos = list(expected_block["pos"])
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate = _successor_layout(current_state, expected_next, action_block_id)
        candidate["blocks"][action_block_id]["pos"] = [
            base_pos[0] + dr,
            base_pos[1] + dc,
        ]
        if (
            state_adapter.canonicalize_state(candidate)
            == state_adapter.canonicalize_state(expected_next)
        ):
            continue
        reason = _classify_invalid_reason(
            current_state,
            action_block_id,
            candidate,
            expected_next,
        )
        if reason == "wrong_moved_block_position":
            return candidate
    return None


# Legacy name retained for internal/tests that import the old helper symbol.
_make_wrong_transition_candidate = _make_wrong_moved_block_candidate


__all__ = [
    "T3_PAPER_INVALID_REASONS",
    "T3_INVALID_REASON_ALIASES",
    "normalize_t3_invalid_reason",
    "is_supported_t3_invalid_reason",
    "get_next_state",
    "evaluate_candidate_transition",
    "make_invalid_candidate_next_state",
]
