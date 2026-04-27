"""Generate controlled invalid states from valid normalized source states."""

from __future__ import annotations

import copy
import random
from typing import Any, Iterable, Sequence

from src.core import state_adapter

SUPPORTED_ERROR_TYPES: tuple[str, ...] = ("overlap", "boundary")


def generate_invalid_state(
    source_state: dict[str, Any],
    error_type: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate one invalid normalized state with one primary error."""
    if error_type not in SUPPORTED_ERROR_TYPES:
        raise ValueError(f"Unsupported error_type: {error_type}")

    source_normalized = state_adapter.normalize_state(source_state)
    if not state_adapter.validate_state(source_normalized):
        raise ValueError("source_state must be a valid normalized state")

    rng = random.Random(seed)
    if error_type == "overlap":
        invalid_state, block_id = _corrupt_overlap(source_normalized, rng)
    else:
        invalid_state, block_id = _corrupt_boundary(source_normalized, rng)

    if state_adapter.validate_state(invalid_state):
        raise RuntimeError("Generated state is still valid; corruption failed")

    return {
        "source_state": source_normalized,
        "invalid_state": invalid_state,
        "errors": [{"block_id": block_id, "error_type": error_type}],
        "meta": {
            "source_canonical_state_key": state_adapter.canonicalize_state(source_normalized),
            "error_type": error_type,
        },
    }


def generate_invalid_state_samples(
    source_states: Iterable[dict[str, Any]],
    error_types: Sequence[str],
    max_per_state: int = 2,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate multiple invalid-state samples with structured annotations."""
    if max_per_state <= 0:
        return []
    if not error_types:
        return []

    for error_type in error_types:
        if error_type not in SUPPORTED_ERROR_TYPES:
            raise ValueError(f"Unsupported error_type: {error_type}")

    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    for state_idx, source_state in enumerate(source_states):
        normalized = state_adapter.normalize_state(source_state)
        if not state_adapter.validate_state(normalized):
            raise ValueError("All source_states must be valid normalized states")

        chosen_error_types = list(error_types)
        rng.shuffle(chosen_error_types)
        chosen_error_types = chosen_error_types[:max_per_state]

        for err_idx, error_type in enumerate(chosen_error_types):
            derived_seed = rng.randint(0, 2**31 - 1)
            sample = generate_invalid_state(
                source_state=normalized,
                error_type=error_type,
                seed=derived_seed,
            )
            sample["meta"]["source_index"] = state_idx
            sample["meta"]["sample_index_within_state"] = err_idx
            samples.append(sample)

    return samples


def _corrupt_overlap(
    source_state: state_adapter.KlotskiState,
    rng: random.Random,
) -> tuple[state_adapter.KlotskiState, str]:
    """Move one block onto another block's top-left cell to force overlap."""
    block_ids = list(source_state["blocks"].keys())
    if len(block_ids) < 2:
        raise ValueError("Need at least two blocks to generate overlap error")

    shuffled = list(block_ids)
    rng.shuffle(shuffled)
    actor_block_id = shuffled[0]
    target_candidates = [bid for bid in shuffled[1:] if bid != actor_block_id]
    target_block_id = target_candidates[0]

    invalid_state = copy.deepcopy(source_state)
    invalid_state["blocks"][actor_block_id]["pos"] = list(
        invalid_state["blocks"][target_block_id]["pos"]
    )
    return invalid_state, actor_block_id


def _corrupt_boundary(
    source_state: state_adapter.KlotskiState,
    rng: random.Random,
) -> tuple[state_adapter.KlotskiState, str]:
    """Move one block out of bounds by one row or one column."""
    rows, cols = int(source_state["grid_size"][0]), int(source_state["grid_size"][1])
    block_ids = list(source_state["blocks"].keys())
    if not block_ids:
        raise ValueError("State has no blocks")

    actor_block_id = rng.choice(block_ids)
    actor_block = source_state["blocks"][actor_block_id]
    h = int(actor_block["shape"][0])
    w = int(actor_block["shape"][1])

    invalid_state = copy.deepcopy(source_state)
    # Push outside grid by one cell while keeping corruption simple and explicit.
    if rng.choice([True, False]):
        invalid_state["blocks"][actor_block_id]["pos"][0] = rows - h + 1
    else:
        invalid_state["blocks"][actor_block_id]["pos"][1] = cols - w + 1
    return invalid_state, actor_block_id


__all__ = [
    "SUPPORTED_ERROR_TYPES",
    "generate_invalid_state",
    "generate_invalid_state_samples",
]
