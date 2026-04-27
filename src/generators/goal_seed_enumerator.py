"""Enumerate solved-state seeds for reverse BFS generation."""

from __future__ import annotations

from typing import Any, Iterable

from src.core import moves_adapter, state_adapter
from src.generators.solved_layout_enumeration import enumerate_solved_huarong_states

# Benchmark reference solved layout (one valid encoding). The exhaustive enumerator
# covers all valid Cao Cao--at--exit packings; this remains the traditional reference.
DEFAULT_GOAL_SEED_CANDIDATES: list[dict[str, Any]] = [
    {
        "Caocao": {"shape": [2, 2], "pos": [3, 1]},
        "H1": {"shape": [2, 1], "pos": [0, 0]},
        "H2": {"shape": [2, 1], "pos": [0, 3]},
        "H3": {"shape": [2, 1], "pos": [2, 0]},
        "H4": {"shape": [2, 1], "pos": [2, 3]},
        "V1": {"shape": [1, 2], "pos": [2, 1]},
        "S1": {"shape": [1, 1], "pos": [0, 1]},
        "S2": {"shape": [1, 1], "pos": [0, 2]},
        "S3": {"shape": [1, 1], "pos": [4, 0]},
        "S4": {"shape": [1, 1], "pos": [4, 3]},
        "Grid_Size": [5, 4],
    }
]


def _iter_normalized_valid_goals(
    candidate_states: Iterable[dict[str, Any]],
) -> list[state_adapter.KlotskiState]:
    """Normalize, validate, and keep only solved candidate states."""
    normalized_goals: list[state_adapter.KlotskiState] = []
    seen: set[state_adapter.CanonicalKey] = set()

    for raw_state in candidate_states:
        try:
            normalized = state_adapter.normalize_state(raw_state)
        except Exception:
            continue

        if not state_adapter.validate_state(normalized):
            continue
        if not state_adapter.is_goal_state(normalized):
            continue

        key = state_adapter.canonicalize_state(normalized)
        if key in seen:
            continue
        seen.add(key)
        normalized_goals.append(normalized)

    return normalized_goals


def filter_goal_states_with_predecessors(
    goal_states: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only solved states that have at least one reverse predecessor."""
    normalized_goals = _iter_normalized_valid_goals(goal_states)
    filtered: list[dict[str, Any]] = []

    for goal_state in normalized_goals:
        if not moves_adapter.enumerate_predecessor_states(goal_state):
            continue
        filtered.append(state_adapter.to_flat_state(goal_state))

    return filtered


def enumerate_goal_seed_states(
    max_goals: int | None = 8_000,
    require_predecessor_space: bool = True,
) -> list[dict[str, Any]]:
    """Enumerate valid solved Hua Rong 5x4 goal seeds (Caocao at ``[3,1]`` with shape ``[2,2]``).

    :param max_goals: After canonical sorting, at most this many goal states (``None`` = all
        discovered solved layouts, often ~10^5; very heavy for reverse BFS). The default
        trade-off uses many distinct solved packings with labeled pieces while keeping BFS
        build times practical.
    :param require_predecessor_space: When true, only keep goals with at least one
        one-step reverse predecessor in the move graph (shallow exact-depth coverage).
    """
    solved = enumerate_solved_huarong_states()
    if not solved:
        return [dict(s) for s in DEFAULT_GOAL_SEED_CANDIDATES]

    flats: list[dict[str, Any]] = [
        state_adapter.to_flat_state(s) for s in solved
    ]
    flats.sort(
        key=lambda f: state_adapter.canonicalize_state(
            state_adapter.normalize_state(f)
        )
    )
    if require_predecessor_space:
        out: list[dict[str, Any]] = []
        for f in flats:
            if not moves_adapter.enumerate_predecessor_states(
                state_adapter.normalize_state(f)
            ):
                continue
            out.append(f)
            if max_goals is not None and len(out) >= max_goals:
                return out
        return out
    if max_goals is not None:
        return flats[:max_goals]
    return flats


__all__ = [
    "DEFAULT_GOAL_SEED_CANDIDATES",
    "enumerate_goal_seed_states",
    "filter_goal_states_with_predecessors",
]
