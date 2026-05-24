"""Build R2 trajectory-verification task cases."""

from __future__ import annotations

import random
from typing import Any, Sequence

from src.core import moves_adapter, state_adapter
from src.core.goal import is_solved_state
from src.generators.trajectory_corruptor import (
    R2_PAPER_INVALID_TRAJECTORY_TYPES,
    V1_INVALID_TRAJECTORY_TYPES,
    corrupt_trajectory,
)


def build_r2_cases(
    exact_depth_cases: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_types: Sequence[str] = R2_PAPER_INVALID_TRAJECTORY_TYPES,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build R2 cases from shortest-solution trajectory metadata."""
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []

    for invalid_type in invalid_types:
        if invalid_type not in V1_INVALID_TRAJECTORY_TYPES:
            raise ValueError(f"Unsupported invalid trajectory type: {invalid_type}")

    for case_idx, case in enumerate(exact_depth_cases):
        initial_state = state_adapter.normalize_state(case["initial_state"])
        goal_state = state_adapter.normalize_state(case["goal_state"])
        gold_actions = [dict(action) for action in case["shortest_solution_actions"]]
        canonical_state_key = case["meta"]["canonical_state_key"]
        optimal_depth = int(case["optimal_depth"])

        if include_valid:
            validity = evaluate_candidate_trajectory(initial_state, gold_actions)
            if validity["label"] == "valid":
                cases.append(
                    {
                        "task": "r2",
                        "input": {
                            "initial_state": initial_state,
                            "candidate_trajectory": gold_actions,
                        },
                        "gold": {"label": "valid"},
                        "meta": {
                            "canonical_state_key": canonical_state_key,
                            "optimal_depth": optimal_depth,
                            "trajectory_type": "oracle_shortest",
                            "source_case_index": case_idx,
                            "goal_state": goal_state,
                        },
                    }
                )

        for invalid_type in invalid_types:
            candidate_trajectory = corrupt_trajectory(
                actions=gold_actions,
                invalid_type=invalid_type,
                rng=rng,
            )
            validity = evaluate_candidate_trajectory(
                initial_state, candidate_trajectory
            )
            if validity["label"] != "invalid":
                continue
            cases.append(
                {
                    "task": "r2",
                    "input": {
                        "initial_state": initial_state,
                        "candidate_trajectory": candidate_trajectory,
                    },
                    "gold": validity,
                    "meta": {
                        "canonical_state_key": canonical_state_key,
                        "optimal_depth": optimal_depth,
                        "trajectory_type": invalid_type,
                        "source_case_index": case_idx,
                        "goal_state": goal_state,
                    },
                }
            )

    return cases


def build_r2_coverage_cases(
    exact_depth_cases: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_types: Sequence[str] = R2_PAPER_INVALID_TRAJECTORY_TYPES,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build one valid and one invalid case per trajectory type when possible."""
    if not exact_depth_cases:
        return []

    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    case_order = list(exact_depth_cases)
    rng.shuffle(case_order)

    if include_valid:
        for case in case_order:
            initial_state = state_adapter.normalize_state(case["initial_state"])
            goal_state = state_adapter.normalize_state(case["goal_state"])
            gold_actions = [dict(action) for action in case["shortest_solution_actions"]]
            validity = evaluate_candidate_trajectory(initial_state, gold_actions)
            if validity["label"] != "valid":
                continue
            cases.append(
                {
                    "task": "r2",
                    "input": {
                        "initial_state": initial_state,
                        "candidate_trajectory": gold_actions,
                    },
                    "gold": {"label": "valid"},
                    "meta": {
                        "canonical_state_key": case["meta"]["canonical_state_key"],
                        "optimal_depth": int(case["optimal_depth"]),
                        "trajectory_type": "oracle_shortest",
                        "goal_state": goal_state,
                    },
                }
            )
            break

    for invalid_type in invalid_types:
        if invalid_type not in V1_INVALID_TRAJECTORY_TYPES:
            raise ValueError(f"Unsupported invalid trajectory type: {invalid_type}")

        for case in case_order:
            initial_state = state_adapter.normalize_state(case["initial_state"])
            goal_state = state_adapter.normalize_state(case["goal_state"])
            gold_actions = [dict(action) for action in case["shortest_solution_actions"]]
            candidate_trajectory = corrupt_trajectory(
                actions=gold_actions,
                invalid_type=invalid_type,
                rng=rng,
            )
            validity = evaluate_candidate_trajectory(
                initial_state, candidate_trajectory
            )
            if validity["label"] != "invalid":
                continue
            cases.append(
                {
                    "task": "r2",
                    "input": {
                        "initial_state": initial_state,
                        "candidate_trajectory": candidate_trajectory,
                    },
                    "gold": validity,
                    "meta": {
                        "canonical_state_key": case["meta"]["canonical_state_key"],
                        "optimal_depth": int(case["optimal_depth"]),
                        "trajectory_type": invalid_type,
                        "goal_state": goal_state,
                    },
                }
            )
            break

    return cases


def build_r2_cases_with_stats(
    exact_depth_cases: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_types: Sequence[str] = R2_PAPER_INVALID_TRAJECTORY_TYPES,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build R2 cases and aggregate validity stats."""
    cases = build_r2_cases(
        exact_depth_cases=exact_depth_cases,
        include_valid=include_valid,
        invalid_types=invalid_types,
        seed=seed,
    )
    valid_count = sum(1 for case in cases if case["gold"]["label"] == "valid")
    invalid_count = len(cases) - valid_count
    stats = {
        "candidate_count": len(cases),
        "exported_count": len(cases),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
    }
    return cases, stats


def evaluate_candidate_trajectory(
    initial_state: state_adapter.KlotskiState,
    candidate_trajectory: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Verify a candidate trajectory against simulator rules and the solved predicate.

    Valid iff every step is legal, every intermediate state is valid, transitions
    follow the simulator, and the final state after the **complete** trajectory
    satisfies ``is_solved_state`` (final_state ∈ G). Shortest-path optimality is
    not required; legal detours that end in G are valid.

    ``first_error_step`` is 1-based. For illegal transitions it is the failing step
    index. For ``goal_not_reached``, all executed steps are marked valid in
    ``step_validity_sequence`` and ``first_error_step = len(candidate_trajectory) + 1``,
    meaning the first failure is detected only after executing the full trajectory
    when the goal predicate is checked.
    """
    current_state = state_adapter.normalize_state(initial_state)
    step_validity_sequence: list[int] = []
    first_error_step: int | None = None

    for step_idx, action in enumerate(candidate_trajectory):
        if first_error_step is not None:
            step_validity_sequence.append(0)
            continue

        legal_actions = moves_adapter.get_legal_actions(current_state)
        if not _action_matches_any(action, legal_actions):
            first_error_step = step_idx + 1
            step_validity_sequence.append(0)
            continue

        current_state = moves_adapter.apply_action(current_state, action)
        if not state_adapter.validate_state(current_state):
            first_error_step = step_idx + 1
            step_validity_sequence.append(0)
            continue

        step_validity_sequence.append(1)

    if first_error_step is not None:
        return {
            "label": "invalid",
            "first_error_step": first_error_step,
            "step_validity_sequence": step_validity_sequence,
        }

    if is_solved_state(current_state):
        return {"label": "valid"}

    trajectory_len = len(candidate_trajectory)
    return {
        "label": "invalid",
        "first_error_step": trajectory_len + 1,
        "step_validity_sequence": [1] * trajectory_len,
    }


def _action_matches_any(
    action: dict[str, str], legal_actions: list[dict[str, str]]
) -> bool:
    return any(
        str(action.get("block_id")) == str(legal.get("block_id"))
        and str(action.get("direction")) == str(legal.get("direction"))
        for legal in legal_actions
    )


__all__ = [
    "R2_PAPER_INVALID_TRAJECTORY_TYPES",
    "V1_INVALID_TRAJECTORY_TYPES",
    "build_r2_cases",
    "build_r2_coverage_cases",
    "build_r2_cases_with_stats",
    "evaluate_candidate_trajectory",
]
