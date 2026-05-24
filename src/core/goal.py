"""Solved-state predicate for trajectory tasks (R1/R2)."""

from __future__ import annotations

from typing import Any, Dict

from src.core import state_adapter

KlotskiState = state_adapter.KlotskiState


def is_solved_state(state: Dict[str, Any] | KlotskiState) -> bool:
    """
    Return whether ``state`` is in the solved-state set G.

    This is the benchmark goal predicate (Cao Cao at the exit on a valid board),
    not equality with any single packaged ``meta.goal_state`` layout.
    """
    normalized = state_adapter.normalize_state(state)
    return state_adapter.is_goal_state(normalized)


__all__ = [
    "is_solved_state",
]
