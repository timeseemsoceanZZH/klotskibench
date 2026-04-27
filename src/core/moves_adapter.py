"""Thin adapter around legacy move-generation and transition utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

from klotski_moves import (
    ActionDict,
    apply_action as _apply_action,
    build_occupied_by_id as _build_occupied_by_id,
    enumerate_predecessor_states as _enumerate_predecessor_states,
    get_legal_actions as _get_legal_actions,
)
from klotski_state import KlotskiState


def get_legal_actions(state: KlotskiState) -> List[ActionDict]:
    """Return legal one-step actions from the given state."""
    return _get_legal_actions(state)


def apply_action(state: KlotskiState, action: ActionDict) -> KlotskiState:
    """Apply one action and return the next state."""
    return _apply_action(state, action)


def build_occupied_by_id(state: KlotskiState) -> Dict[Tuple[int, int], str]:
    """Map occupied cell coordinates to owning block id."""
    return _build_occupied_by_id(state)


def apply_action_checked(state: KlotskiState, action: ActionDict) -> KlotskiState:
    """Apply action only if it is legal from the given state."""
    legal_actions = _get_legal_actions(state)
    if action not in legal_actions:
        raise ValueError(f"Illegal action for state: {action}")
    return _apply_action(state, action)


def enumerate_predecessor_states(state: KlotskiState) -> list[KlotskiState]:
    """All states S such that a single legal move from S matches ``state`` (reverse step)."""
    return _enumerate_predecessor_states(state)


__all__ = [
    "ActionDict",
    "KlotskiState",
    "get_legal_actions",
    "apply_action",
    "build_occupied_by_id",
    "apply_action_checked",
    "enumerate_predecessor_states",
]
