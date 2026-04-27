"""Per-depth exact-depth candidate stats from precomputed reverse-BFS records."""
from __future__ import annotations

from typing import Any, Dict

from src.core import bfs_adapter, state_adapter

RecordMap = Dict[str, Any]


def candidate_count_at_depth(records: RecordMap, depth: int) -> int:
    """Count how many *non-goal* valid states are exactly ``depth`` away in reverse-BFS records.

    This matches :func:`build_exact_depth_cases_with_stats` candidate_count semantics.
    """
    at_depth = bfs_adapter.states_at_depth(records=records, depth=depth)
    n = 0
    for record in at_depth:
        state = record["state"] if isinstance(record, dict) else record.state
        if depth >= 1 and state_adapter.is_goal_state(state):
            continue
        if not state_adapter.validate_state(state):
            continue
        n += 1
    return n


def candidate_counts_in_range(
    records: RecordMap, depth_min: int, depth_max: int
) -> list[tuple[int, int]]:
    """Return (depth, candidate_count) for each depth in ``[depth_min, depth_max]``."""
    return [
        (d, candidate_count_at_depth(records, d))
        for d in range(depth_min, depth_max + 1)
    ]


__all__ = [
    "candidate_count_at_depth",
    "candidate_counts_in_range",
]
