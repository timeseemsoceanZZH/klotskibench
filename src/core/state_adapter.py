"""Thin adapter around legacy state utilities."""

from __future__ import annotations

from typing import Any, Dict

from klotski_state import (
    CanonicalKey,
    KlotskiState,
    canonicalize_state as _canonicalize_state,
    is_goal_state as _is_goal_state,
    normalize_state as _normalize_state,
    to_flat_state as _to_flat_state,
    validate_state as _validate_state,
)


def normalize_state(state: Dict[str, Any]) -> KlotskiState:
    """Normalize flat or nested input state into legacy normalized format."""
    return _normalize_state(state)


def canonicalize_state(state: KlotskiState) -> CanonicalKey:
    """Return hashable canonical key for deduplication."""
    return _canonicalize_state(state)


def validate_state(state: KlotskiState) -> bool:
    """Validate bounds and overlap constraints for a normalized state."""
    return _validate_state(state)


def is_goal_state(state: KlotskiState) -> bool:
    """Return whether the state satisfies the legacy goal predicate."""
    return _is_goal_state(state)


def to_flat_state(state: KlotskiState) -> Dict[str, Any]:
    """Convert normalized state back to legacy flat representation."""
    return _to_flat_state(state)


__all__ = [
    "CanonicalKey",
    "KlotskiState",
    "normalize_state",
    "canonicalize_state",
    "validate_state",
    "is_goal_state",
    "to_flat_state",
]
