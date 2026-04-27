"""Thin adapter around legacy BFS shortest-path utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from klotski_bfs import (
    ActionDict,
    StateRecord,
    bfs_shortest_paths as _bfs_shortest_paths,
    reverse_bfs_shortest_paths as _reverse_bfs_shortest_paths,
    reverse_bfs_shortest_paths_multi as _reverse_bfs_shortest_paths_multi,
    states_at_depth as _states_at_depth,
    states_in_depth_range as _states_in_depth_range,
)
from klotski_state import CanonicalKey, KlotskiState


def forward_bfs_shortest_paths(
    start_state: KlotskiState,
    max_depth: Optional[int] = None,
) -> Dict[CanonicalKey, StateRecord]:
    """Run forward BFS and return shortest-path records keyed by canonical state."""
    return _bfs_shortest_paths(start_state=start_state, max_depth=max_depth)


def reverse_bfs_shortest_paths(
    goal_seed_state: KlotskiState,
    is_goal_fn: Callable[[KlotskiState], bool],
    max_depth: Optional[int] = None,
) -> Dict[CanonicalKey, StateRecord]:
    """Run reverse BFS from a single goal seed state."""
    return _reverse_bfs_shortest_paths(
        goal_seed_state=goal_seed_state,
        is_goal_fn=is_goal_fn,
        max_depth=max_depth,
    )


def reverse_bfs_shortest_paths_multi(
    goal_seed_states: List[KlotskiState],
    is_goal_fn: Callable[[KlotskiState], bool],
    max_depth: Optional[int] = None,
) -> Dict[CanonicalKey, StateRecord]:
    """Run reverse BFS from multiple goal seed states."""
    return _reverse_bfs_shortest_paths_multi(
        goal_seed_states=goal_seed_states,
        is_goal_fn=is_goal_fn,
        max_depth=max_depth,
    )


def states_at_depth(
    records: Dict[CanonicalKey, StateRecord],
    depth: int,
) -> List[StateRecord]:
    """Return records at exactly one BFS depth."""
    return _states_at_depth(records=records, depth=depth)


def states_in_depth_range(
    records: Dict[CanonicalKey, StateRecord],
    depth_min: int,
    depth_max: int,
) -> List[StateRecord]:
    """Return records with depth in [depth_min, depth_max]."""
    return _states_in_depth_range(records=records, depth_min=depth_min, depth_max=depth_max)


__all__ = [
    "ActionDict",
    "CanonicalKey",
    "KlotskiState",
    "StateRecord",
    "forward_bfs_shortest_paths",
    "reverse_bfs_shortest_paths",
    "reverse_bfs_shortest_paths_multi",
    "states_at_depth",
    "states_in_depth_range",
]
