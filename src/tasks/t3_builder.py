"""Build T3 transition-validation cases."""

from __future__ import annotations

from typing import Any, Sequence

from src.core import moves_adapter, state_adapter
from src.generators import transition_oracle

V1_INVALID_REASONS: tuple[str, ...] = (
    "overlap",
    "boundary",
    "stationary_object_drift",
    "wrong_transition",
)


def build_t3_cases(
    legal_state_items: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_reasons: Sequence[str] = V1_INVALID_REASONS,
) -> list[dict[str, Any]]:
    """Build T3 cases for transition validity classification."""
    cases: list[dict[str, Any]] = []

    invalid_reason_set = tuple(invalid_reasons)
    for reason in invalid_reason_set:
        if reason not in V1_INVALID_REASONS:
            raise ValueError(f"Unsupported invalid reason: {reason}")

    for item in legal_state_items:
        current_state = state_adapter.normalize_state(item["state"])
        legal_actions = moves_adapter.get_legal_actions(current_state)
        canonical_state_key = item.get(
            "canonical_state_key",
            state_adapter.canonicalize_state(current_state),
        )
        optimal_depth = item.get("optimal_depth")

        for action in legal_actions:
            expected_next_state = transition_oracle.get_next_state(current_state, action)

            if include_valid:
                cases.append(
                    {
                        "task": "t3",
                        "input": {
                            "current_state": current_state,
                            "action": action,
                            "candidate_next_state": expected_next_state,
                        },
                        "gold": {"label": "valid"},
                        "meta": {
                            "optimal_depth": optimal_depth,
                            "canonical_state_key": canonical_state_key,
                        },
                    }
                )

            for invalid_reason in invalid_reason_set:
                try:
                    invalid_candidate = transition_oracle.make_invalid_candidate_next_state(
                        current_state=current_state,
                        action=action,
                        invalid_reason=invalid_reason,
                    )
                except ValueError:
                    continue

                gold = transition_oracle.evaluate_candidate_transition(
                    current_state=current_state,
                    action=action,
                    candidate_next_state=invalid_candidate,
                )
                if gold.get("label") != "invalid":
                    continue
                cases.append(
                    {
                        "task": "t3",
                        "input": {
                            "current_state": current_state,
                            "action": action,
                            "candidate_next_state": invalid_candidate,
                        },
                        "gold": gold,
                        "meta": {
                            "optimal_depth": optimal_depth,
                            "canonical_state_key": canonical_state_key,
                            "requested_invalid_reason": invalid_reason,
                        },
                    }
                )

    return cases


def build_t3_cases_with_stats(
    legal_state_items: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_reasons: Sequence[str] = V1_INVALID_REASONS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build T3 cases and aggregate label stats."""
    cases = build_t3_cases(
        legal_state_items=legal_state_items,
        include_valid=include_valid,
        invalid_reasons=invalid_reasons,
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


__all__ = [
    "V1_INVALID_REASONS",
    "build_t3_cases",
    "build_t3_cases_with_stats",
]
