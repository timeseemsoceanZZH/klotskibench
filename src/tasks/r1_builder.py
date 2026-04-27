"""Build R1 trajectory-generation task cases."""

from __future__ import annotations

from typing import Any, Sequence

from src.core import state_adapter


def build_r1_cases(
    exact_depth_cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build R1 cases from solvable exact-depth case objects."""
    cases: list[dict[str, Any]] = []
    for case in exact_depth_cases:
        initial_state = state_adapter.normalize_state(case["initial_state"])
        goal_state = state_adapter.normalize_state(case["goal_state"])
        shortest_solution_actions = list(case["shortest_solution_actions"])
        shortest_solution_states = [
            state_adapter.normalize_state(s) for s in case["shortest_solution_states"]
        ]

        cases.append(
            {
                "task": "r1",
                "input": {
                    "initial_state": initial_state,
                },
                "gold": {
                    "optimal_depth": int(case["optimal_depth"]),
                },
                "meta": {
                    "canonical_state_key": case["meta"]["canonical_state_key"],
                    "goal_state": goal_state,
                    "shortest_solution_actions": shortest_solution_actions,
                    "shortest_solution_states": shortest_solution_states,
                },
            }
        )
    return cases


def build_r1_cases_with_stats(
    exact_depth_cases: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build R1 cases and export stats."""
    cases = build_r1_cases(exact_depth_cases=exact_depth_cases)
    stats = {
        "candidate_count": len(exact_depth_cases),
        "exported_count": len(cases),
    }
    return cases, stats


__all__ = [
    "build_r1_cases",
    "build_r1_cases_with_stats",
]
