"""Build reverse-BFS records from solved goal seeds."""

from __future__ import annotations

import random
from typing import Any, Iterable

from src.core import bfs_adapter, state_adapter


def _normalize_valid_goal_seeds(
    goal_seed_states: Iterable[dict[str, Any]],
) -> list[state_adapter.KlotskiState]:
    """Normalize and validate goal seed states."""
    normalized: list[state_adapter.KlotskiState] = []
    seen: set[state_adapter.CanonicalKey] = set()

    for raw_state in goal_seed_states:
        candidate = state_adapter.normalize_state(raw_state)
        if not state_adapter.validate_state(candidate):
            continue
        if not state_adapter.is_goal_state(candidate):
            continue

        key = state_adapter.canonicalize_state(candidate)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)

    return normalized


def build_reverse_bfs_records(
    goal_seed_states: Iterable[dict[str, Any]],
    max_depth: int | None = None,
) -> dict[state_adapter.CanonicalKey, bfs_adapter.StateRecord]:
    """Run reverse BFS from solved goal seeds."""
    normalized_seeds = _normalize_valid_goal_seeds(goal_seed_states)
    if not normalized_seeds:
        return {}

    return bfs_adapter.reverse_bfs_shortest_paths_multi(
        goal_seed_states=normalized_seeds,
        is_goal_fn=state_adapter.is_goal_state,
        max_depth=max_depth,
    )


def build_reverse_bfs_dataset(
    goal_seed_states: Iterable[dict[str, Any]],
    max_depth: int | None = None,
) -> list[dict[str, Any]]:
    """Create flat export rows from reverse-BFS records."""
    records = build_reverse_bfs_records(goal_seed_states=goal_seed_states, max_depth=max_depth)
    rows: list[dict[str, Any]] = []
    for key, record in records.items():
        rows.append(
            {
                "canonical_key": key,
                "depth": _record_depth(record),
                "state": state_adapter.to_flat_state(_record_state(record)),
            }
        )

    rows.sort(key=lambda row: row["depth"])
    return rows


def build_exact_depth_cases(
    records: dict[state_adapter.CanonicalKey, bfs_adapter.StateRecord],
    depth: int,
    num_samples: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build exact-depth cases from precomputed reverse-BFS records."""
    if depth < 0:
        return []

    rng = random.Random(seed)
    at_depth = bfs_adapter.states_at_depth(records=records, depth=depth)

    candidates = []
    for record in at_depth:
        state = _record_state(record)
        if depth >= 1 and state_adapter.is_goal_state(state):
            continue
        if not state_adapter.validate_state(state):
            continue
        candidates.append(record)

    if num_samples is not None and len(candidates) > num_samples:
        candidates = rng.sample(candidates, num_samples)

    return [
        {
            "case_id": f"depth{depth}_{idx}",
            "initial_state": _record_state(record),
            "goal_state": _record_path_states(record)[-1],
            "optimal_depth": depth,
            "shortest_solution_actions": list(_record_path_actions(record)),
            "shortest_solution_states": list(_record_path_states(record)),
            "meta": {
                "difficulty_bucket": f"depth_{depth}",
                "canonical_state_key": state_adapter.canonicalize_state(_record_state(record)),
            },
        }
        for idx, record in enumerate(candidates)
    ]


def build_exact_depth_cases_with_stats(
    records: dict[state_adapter.CanonicalKey, bfs_adapter.StateRecord],
    depth: int,
    num_samples: int | None = None,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build exact-depth cases and basic candidate/export stats."""
    if depth < 0:
        return [], {"candidate_count": 0, "exported_count": 0}

    at_depth = bfs_adapter.states_at_depth(records=records, depth=depth)
    candidate_records = []
    for record in at_depth:
        state = _record_state(record)
        if depth >= 1 and state_adapter.is_goal_state(state):
            continue
        if not state_adapter.validate_state(state):
            continue
        candidate_records.append(record)

    candidate_count = len(candidate_records)
    cases = build_exact_depth_cases(
        records=records,
        depth=depth,
        num_samples=num_samples,
        seed=seed,
    )
    stats = {
        "candidate_count": candidate_count,
        "exported_count": len(cases),
    }
    return cases, stats


def _record_depth(record: bfs_adapter.StateRecord) -> int:
    """Extract integer depth from legacy-compatible state record."""
    if isinstance(record, dict):
        return int(record["depth"])
    return int(getattr(record, "depth"))


def _record_state(record: bfs_adapter.StateRecord) -> state_adapter.KlotskiState:
    """Extract state payload from legacy-compatible state record."""
    if isinstance(record, dict):
        return record["state"]
    return getattr(record, "state")


def _record_path_actions(record: bfs_adapter.StateRecord) -> list[dict[str, str]]:
    """Extract shortest path actions from legacy-compatible state record."""
    if isinstance(record, dict):
        return list(record["path_actions"])
    return list(getattr(record, "path_actions"))


def _record_path_states(record: bfs_adapter.StateRecord) -> list[state_adapter.KlotskiState]:
    """Extract shortest path states from legacy-compatible state record."""
    if isinstance(record, dict):
        return list(record["path_states"])
    return list(getattr(record, "path_states"))


__all__ = [
    "build_reverse_bfs_records",
    "build_reverse_bfs_dataset",
    "build_exact_depth_cases",
    "build_exact_depth_cases_with_stats",
]
