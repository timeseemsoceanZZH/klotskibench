"""Build T2 transition-prediction cases."""

from __future__ import annotations

from typing import Any, Sequence

from src.core import moves_adapter, state_adapter
from src.generators import transition_oracle


def build_t2_cases(
    legal_state_items: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build T2 cases from legal current state and legal action."""
    cases: list[dict[str, Any]] = []

    for item in legal_state_items:
        current_state = state_adapter.normalize_state(item["state"])
        legal_actions = moves_adapter.get_legal_actions(current_state)
        canonical_state_key = item.get(
            "canonical_state_key",
            state_adapter.canonicalize_state(current_state),
        )
        optimal_depth = item.get("optimal_depth")

        for action in legal_actions:
            next_state = transition_oracle.get_next_state(current_state, action)
            cases.append(
                {
                    "task": "t2",
                    "input": {
                        "current_state": current_state,
                        "action": action,
                    },
                    "gold": {"next_state": next_state},
                    "meta": {
                        "optimal_depth": optimal_depth,
                        "canonical_state_key": canonical_state_key,
                    },
                }
            )

    return cases


def build_t2_cases_with_stats(
    legal_state_items: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build T2 cases and export stats."""
    cases = build_t2_cases(legal_state_items=legal_state_items)
    stats = {
        "candidate_count": len(cases),
        "exported_count": len(cases),
    }
    return cases, stats


__all__ = [
    "build_t2_cases",
    "build_t2_cases_with_stats",
]
