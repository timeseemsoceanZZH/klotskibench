"""Build T3 transition-validation cases."""

from __future__ import annotations

import random
from typing import Any, Sequence

from src.core import moves_adapter, state_adapter
from src.generators import transition_oracle

T3_PAPER_INVALID_REASONS: tuple[str, ...] = transition_oracle.T3_PAPER_INVALID_REASONS

# Backward-compatible export (legacy name; paper reasons + alias).
V1_INVALID_REASONS: tuple[str, ...] = T3_PAPER_INVALID_REASONS + ("wrong_transition",)


def build_t3_cases(
    legal_state_items: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_reasons: Sequence[str] = T3_PAPER_INVALID_REASONS,
) -> list[dict[str, Any]]:
    """Build T3 cases for transition validity classification."""
    cases: list[dict[str, Any]] = []

    invalid_reason_set = tuple(invalid_reasons)
    for reason in invalid_reason_set:
        if not transition_oracle.is_supported_t3_invalid_reason(reason):
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
                normalized_reason = transition_oracle.normalize_t3_invalid_reason(
                    invalid_reason
                )
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
                if gold.get("reason") != normalized_reason:
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
                            "requested_invalid_reason": normalized_reason,
                        },
                    }
                )

    return cases


def build_t3_coverage_cases(
    legal_state_items: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_reasons: Sequence[str] = T3_PAPER_INVALID_REASONS,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build T3 cases with at least one invalid example per paper reason when possible."""
    if not legal_state_items:
        return []

    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    if include_valid:
        first = legal_state_items[0]
        state = state_adapter.normalize_state(first["state"])
        legal_actions = moves_adapter.get_legal_actions(state)
        if legal_actions:
            action = legal_actions[0]
            expected = transition_oracle.get_next_state(state, action)
            cases.append(
                {
                    "task": "t3",
                    "input": {
                        "current_state": state,
                        "action": action,
                        "candidate_next_state": expected,
                    },
                    "gold": {"label": "valid"},
                    "meta": {
                        "optimal_depth": first.get("optimal_depth"),
                        "canonical_state_key": first.get(
                            "canonical_state_key",
                            state_adapter.canonicalize_state(state),
                        ),
                    },
                }
            )

    items_shuffled = list(legal_state_items)
    rng.shuffle(items_shuffled)

    for reason in invalid_reasons:
        normalized = transition_oracle.normalize_t3_invalid_reason(reason)
        if normalized not in T3_PAPER_INVALID_REASONS:
            raise ValueError(f"Unsupported invalid reason: {reason}")

        found = False
        for item in items_shuffled:
            current_state = state_adapter.normalize_state(item["state"])
            actions = list(moves_adapter.get_legal_actions(current_state))
            rng.shuffle(actions)
            for action in actions:
                try:
                    invalid_candidate = transition_oracle.make_invalid_candidate_next_state(
                        current_state=current_state,
                        action=action,
                        invalid_reason=reason,
                    )
                except ValueError:
                    continue
                gold = transition_oracle.evaluate_candidate_transition(
                    current_state=current_state,
                    action=action,
                    candidate_next_state=invalid_candidate,
                )
                if gold.get("label") != "invalid" or gold.get("reason") != normalized:
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
                            "optimal_depth": item.get("optimal_depth"),
                            "canonical_state_key": item.get(
                                "canonical_state_key",
                                state_adapter.canonicalize_state(current_state),
                            ),
                            "requested_invalid_reason": normalized,
                        },
                    }
                )
                found = True
                break
            if found:
                break

    return cases


def build_t3_cases_with_stats(
    legal_state_items: Sequence[dict[str, Any]],
    include_valid: bool = True,
    invalid_reasons: Sequence[str] = T3_PAPER_INVALID_REASONS,
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
    "T3_PAPER_INVALID_REASONS",
    "V1_INVALID_REASONS",
    "build_t3_cases",
    "build_t3_coverage_cases",
    "build_t3_cases_with_stats",
]
