"""Build S1 validity-classification task cases."""

from __future__ import annotations

from typing import Any, Sequence

from src.core import state_adapter
from src.generators.invalid_state_generator import generate_invalid_state_samples


def build_s1_cases(
    legal_state_items: Sequence[dict[str, Any]],
    error_types: Sequence[str] = ("overlap", "boundary"),
    max_invalid_per_state: int = 2,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build S1 cases with both valid and invalid labels."""
    valid_source_states = [
        state_adapter.normalize_state(item["state"]) for item in legal_state_items
    ]

    invalid_samples = generate_invalid_state_samples(
        source_states=valid_source_states,
        error_types=error_types,
        max_per_state=max_invalid_per_state,
        seed=seed,
    )

    cases: list[dict[str, Any]] = []

    for item in legal_state_items:
        state = state_adapter.normalize_state(item["state"])
        cases.append(
            {
                "task": "s1",
                "input": {"state": state},
                "gold": {"label": "valid"},
                "meta": {
                    "optimal_depth": item.get("optimal_depth"),
                    "canonical_state_key": item.get(
                        "canonical_state_key",
                        state_adapter.canonicalize_state(state),
                    ),
                },
            }
        )

    for sample in invalid_samples:
        invalid_state = state_adapter.normalize_state(sample["invalid_state"])
        cases.append(
            {
                "task": "s1",
                "input": {"state": invalid_state},
                "gold": {"label": "invalid"},
                "meta": {
                    "error_type": sample["meta"]["error_type"],
                    "source_canonical_state_key": sample["meta"]["source_canonical_state_key"],
                    "errors": list(sample["errors"]),
                },
            }
        )

    return cases


__all__ = [
    "build_s1_cases",
]
