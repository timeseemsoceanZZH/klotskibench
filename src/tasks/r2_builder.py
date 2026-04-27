"""Build R2 trajectory-verification task cases."""

from __future__ import annotations

import random
from typing import Any, Sequence

from src.core import moves_adapter, state_adapter
from src.generators.trajectory_corruptor import (
    V1_INVALID_TRAJECTORY_TYPES,
    corrupt_trajectory,
)


def build_r2_cases(
    exact_depth_cases: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_types: Sequence[str] = V1_INVALID_TRAJECTORY_TYPES,
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
        gold_actions = [dict(action) for action in case["shortest_solution_actions"]]
        canonical_state_key = case["meta"]["canonical_state_key"]
        optimal_depth = int(case["optimal_depth"])

        if include_valid:
            validity = _evaluate_action_trajectory(initial_state, gold_actions)
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
                        },
                    }
                )

        for invalid_type in invalid_types:
            candidate_trajectory = corrupt_trajectory(
                actions=gold_actions,
                invalid_type=invalid_type,
                rng=rng,
            )
            validity = _evaluate_action_trajectory(initial_state, candidate_trajectory)
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
                    },
                }
            )

    return cases


def build_r2_cases_with_stats(
    exact_depth_cases: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_types: Sequence[str] = V1_INVALID_TRAJECTORY_TYPES,
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


def _evaluate_action_trajectory(
    initial_state: state_adapter.KlotskiState,
    candidate_trajectory: list[dict[str, str]],
) -> dict[str, Any]:
    """Evaluate trajectory with prefix-based truncation validity semantics."""
    current_state = state_adapter.normalize_state(initial_state)
    step_validity_sequence: list[int] = []
    first_error_step: int | None = None

    for step_idx, action in enumerate(candidate_trajectory):
        if first_error_step is not None:
            step_validity_sequence.append(0)
            continue

        legal_actions = moves_adapter.get_legal_actions(current_state)
        if action not in legal_actions:
            first_error_step = step_idx
            step_validity_sequence.append(0)
            continue

        current_state = moves_adapter.apply_action(current_state, action)
        if not state_adapter.validate_state(current_state):
            first_error_step = step_idx
            step_validity_sequence.append(0)
            continue

        step_validity_sequence.append(1)

    if first_error_step is None:
        return {"label": "valid"}
    return {
        "label": "invalid",
        "first_error_step": first_error_step + 1,
        "step_validity_sequence": step_validity_sequence,
    }


__all__ = [
    "V1_INVALID_TRAJECTORY_TYPES",
    "build_r2_cases",
    "build_r2_cases_with_stats",
]
