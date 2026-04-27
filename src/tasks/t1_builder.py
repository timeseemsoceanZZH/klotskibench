"""Build T1 legal-action-set task cases from legal state pool items."""

from __future__ import annotations

from typing import Any, Sequence

from src.core import moves_adapter, state_adapter


def build_t1_cases(
    legal_state_items: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build T1 cases where gold is the complete legal action set."""
    cases: list[dict[str, Any]] = []
    for item in legal_state_items:
        current_state = state_adapter.normalize_state(item["state"])
        legal_actions = moves_adapter.get_legal_actions(current_state)
        cases.append(
            {
                "task": "t1",
                "input": {"current_state": current_state},
                "gold": {"legal_actions": legal_actions},
                "meta": {
                    "optimal_depth": item.get("optimal_depth"),
                    "canonical_state_key": item.get(
                        "canonical_state_key",
                        state_adapter.canonicalize_state(current_state),
                    ),
                },
            }
        )
    return cases


def build_t1_cases_with_stats(
    legal_state_items: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build T1 cases and simple export stats."""
    cases = build_t1_cases(legal_state_items=legal_state_items)
    stats = {
        "candidate_count": len(legal_state_items),
        "exported_count": len(cases),
    }
    return cases, stats


__all__ = [
    "build_t1_cases",
    "build_t1_cases_with_stats",
]
