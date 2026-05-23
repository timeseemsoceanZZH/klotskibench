"""Build S2 error-localization task cases."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from src.core import state_adapter
from src.generators.invalid_state_generator import (
    PAPER_STATE_ERROR_TYPES,
    generate_invalid_state_coverage,
    generate_invalid_state_samples,
)


def build_s2_cases(
    source_states: Iterable[dict[str, Any]],
    error_types: Sequence[str] = PAPER_STATE_ERROR_TYPES,
    max_invalid_per_state: int = 5,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build S2 cases for block-level error localization and typing."""
    invalid_samples = generate_invalid_state_samples(
        source_states=source_states,
        error_types=error_types,
        max_per_state=max_invalid_per_state,
        seed=seed,
    )
    return _cases_from_invalid_samples(invalid_samples)


def build_s2_coverage_cases(
    source_states: Iterable[dict[str, Any]],
    error_types: Sequence[str] = PAPER_STATE_ERROR_TYPES,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build S2 cases with exactly one case per paper error type."""
    invalid_samples = generate_invalid_state_coverage(
        source_states=source_states,
        error_types=error_types,
        seed=seed,
    )
    return _cases_from_invalid_samples(invalid_samples)


def _cases_from_invalid_samples(
    invalid_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for idx, sample in enumerate(invalid_samples):
        invalid_state = state_adapter.normalize_state(sample["invalid_state"])
        cases.append(
            {
                "task": "s2",
                "input": {"state": invalid_state},
                "gold": {"errors": list(sample["errors"])},
                "meta": {
                    "error_type": sample["meta"]["error_type"],
                    "source_canonical_state_key": sample["meta"]["source_canonical_state_key"],
                    "source_index": sample["meta"].get("source_index"),
                    "sample_index_within_state": sample["meta"].get("sample_index_within_state"),
                },
            }
        )

    return cases


__all__ = [
    "build_s2_cases",
    "build_s2_coverage_cases",
]
