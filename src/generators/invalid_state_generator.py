"""Generate controlled invalid states from valid normalized source states."""

from __future__ import annotations

import copy
import random
from typing import Any, Iterable, Sequence

from src.core import state_adapter

PAPER_STATE_ERROR_TYPES: tuple[str, ...] = (
    "overlap",
    "boundary",
    "missing_block",
    "identity_mismatch",
    "shape_change",
)

# Backward-compatible alias used by existing imports.
SUPPORTED_ERROR_TYPES: tuple[str, ...] = PAPER_STATE_ERROR_TYPES

_GEOMETRIC_INVALID_TYPES: frozenset[str] = frozenset(
    {"overlap", "boundary", "shape_change"}
)


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
        invalid_state, errors = _corrupt_overlap(source_normalized, rng)
    elif error_type == "boundary":
        invalid_state, errors = _corrupt_boundary(source_normalized, rng)
    elif error_type == "missing_block":
        invalid_state, errors = _corrupt_missing_block(source_normalized, rng)
    elif error_type == "shape_change":
        invalid_state, errors = _corrupt_shape_change(source_normalized, rng)
    else:
        invalid_state, errors = _corrupt_identity_mismatch(source_normalized, rng)

    if error_type in _GEOMETRIC_INVALID_TYPES and state_adapter.validate_state(invalid_state):
        raise RuntimeError("Generated state is still valid; corruption failed")

    return {
        "source_state": source_normalized,
        "invalid_state": invalid_state,
        "errors": errors,
        "meta": {
            "source_canonical_state_key": state_adapter.canonicalize_state(source_normalized),
            "error_type": error_type,
        },
    }


def generate_invalid_state_samples(
    source_states: Iterable[dict[str, Any]],
    error_types: Sequence[str],
    max_per_state: int = 5,
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
        if max_per_state >= len(chosen_error_types):
            pass
        else:
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


def generate_invalid_state_coverage(
    source_states: Iterable[dict[str, Any]],
    error_types: Sequence[str] = PAPER_STATE_ERROR_TYPES,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate one sample per error type using the first source state that succeeds."""
    if not error_types:
        return []

    for error_type in error_types:
        if error_type not in SUPPORTED_ERROR_TYPES:
            raise ValueError(f"Unsupported error_type: {error_type}")

    normalized_sources: list[state_adapter.KlotskiState] = []
    for raw in source_states:
        normalized = state_adapter.normalize_state(raw)
        if state_adapter.validate_state(normalized):
            normalized_sources.append(normalized)
    if not normalized_sources:
        raise ValueError("No valid source_states for coverage generation")

    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    for error_type in error_types:
        order = list(normalized_sources)
        rng.shuffle(order)
        last_error: Exception | None = None
        for src in order:
            try:
                sample = generate_invalid_state(
                    source_state=src,
                    error_type=error_type,
                    seed=rng.randint(0, 2**31 - 1),
                )
                sample["meta"]["coverage_error_type"] = error_type
                samples.append(sample)
                break
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                continue
        else:
            msg = f"Could not generate coverage sample for error_type={error_type!r}"
            if last_error is not None:
                msg = f"{msg}: {last_error}"
            raise RuntimeError(msg)

    return samples


def _corrupt_overlap(
    source_state: state_adapter.KlotskiState,
    rng: random.Random,
) -> tuple[state_adapter.KlotskiState, list[dict[str, Any]]]:
    """Move one block onto another block's top-left cell to force overlap."""
    block_ids = list(source_state["blocks"].keys())
    if len(block_ids) < 2:
        raise ValueError("Need at least two blocks to generate overlap error")

    shuffled = list(block_ids)
    rng.shuffle(shuffled)

    for actor_block_id in shuffled:
        targets = [bid for bid in shuffled if bid != actor_block_id]
        rng.shuffle(targets)
        for target_block_id in targets:
            invalid_state = copy.deepcopy(source_state)
            target_pos = list(invalid_state["blocks"][target_block_id]["pos"])
            invalid_state["blocks"][actor_block_id]["pos"] = target_pos
            cells, block_ids_involved = _overlap_cells_and_blocks(invalid_state)
            if not cells:
                th = int(invalid_state["blocks"][target_block_id]["shape"][0])
                tw = int(invalid_state["blocks"][target_block_id]["shape"][1])
                if th > 1 or tw > 1:
                    invalid_state["blocks"][actor_block_id]["pos"] = [
                        target_pos[0] + min(1, th - 1),
                        target_pos[1] + min(1, tw - 1),
                    ]
                    cells, block_ids_involved = _overlap_cells_and_blocks(invalid_state)
            if cells and not state_adapter.validate_state(invalid_state):
                return invalid_state, [
                    {
                        "block_id": actor_block_id,
                        "error_type": "overlap",
                        "block_ids": block_ids_involved,
                        "cells": cells,
                    }
                ]

    raise ValueError("Could not generate overlap corruption")


def _corrupt_boundary(
    source_state: state_adapter.KlotskiState,
    rng: random.Random,
) -> tuple[state_adapter.KlotskiState, list[dict[str, Any]]]:
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
    if rng.choice([True, False]):
        invalid_state["blocks"][actor_block_id]["pos"][0] = rows - h + 1
    else:
        invalid_state["blocks"][actor_block_id]["pos"][1] = cols - w + 1

    cells = _out_of_bounds_cells(invalid_state, actor_block_id)
    return invalid_state, [
        {
            "block_id": actor_block_id,
            "error_type": "boundary",
            "cells": cells,
        }
    ]


def _corrupt_missing_block(
    source_state: state_adapter.KlotskiState,
    rng: random.Random,
) -> tuple[state_adapter.KlotskiState, list[dict[str, Any]]]:
    block_ids = list(source_state["blocks"].keys())
    if not block_ids:
        raise ValueError("State has no blocks")

    missing_block_id = rng.choice(block_ids)
    invalid_state = copy.deepcopy(source_state)
    del invalid_state["blocks"][missing_block_id]
    return invalid_state, [
        {
            "error_type": "missing_block",
            "missing_block_id": missing_block_id,
            "block_id": missing_block_id,
        }
    ]


def _corrupt_shape_change(
    source_state: state_adapter.KlotskiState,
    rng: random.Random,
) -> tuple[state_adapter.KlotskiState, list[dict[str, Any]]]:
    block_ids = list(source_state["blocks"].keys())
    if not block_ids:
        raise ValueError("State has no blocks")

    shuffled = list(block_ids)
    rng.shuffle(shuffled)

    for actor_block_id in shuffled:
        block = source_state["blocks"][actor_block_id]
        h, w = int(block["shape"][0]), int(block["shape"][1])
        for nh in range(h, h + 4):
            for nw in range(w, w + 4):
                if (nh, nw) == (h, w):
                    continue
                invalid_state = copy.deepcopy(source_state)
                invalid_state["blocks"][actor_block_id]["shape"] = [nh, nw]
                if not state_adapter.validate_state(invalid_state):
                    return invalid_state, [
                        {
                            "block_id": actor_block_id,
                            "error_type": "shape_change",
                        }
                    ]

    raise ValueError("Could not generate shape_change corruption")


def _corrupt_identity_mismatch(
    source_state: state_adapter.KlotskiState,
    rng: random.Random,
) -> tuple[state_adapter.KlotskiState, list[dict[str, Any]]]:
    """Swap block payloads under two ids so labels no longer match geometry roles."""
    block_ids = list(source_state["blocks"].keys())
    if len(block_ids) < 2:
        raise ValueError("Need at least two blocks for identity_mismatch")

    id_a, id_b = rng.sample(block_ids, 2)
    invalid_state = copy.deepcopy(source_state)
    payload_a = copy.deepcopy(source_state["blocks"][id_a])
    payload_b = copy.deepcopy(source_state["blocks"][id_b])
    invalid_state["blocks"][id_a] = payload_b
    invalid_state["blocks"][id_b] = payload_a
    return invalid_state, [
        {
            "block_id": id_a,
            "error_type": "identity_mismatch",
            "block_ids": [id_a, id_b],
        }
    ]


def _block_cells(
    state: state_adapter.KlotskiState, block_id: str
) -> set[tuple[int, int]]:
    block = state["blocks"][block_id]
    r0, c0 = int(block["pos"][0]), int(block["pos"][1])
    h, w = int(block["shape"][0]), int(block["shape"][1])
    return {(r0 + dr, c0 + dc) for dr in range(h) for dc in range(w)}


def _overlap_cells_and_blocks(
    state: state_adapter.KlotskiState,
) -> tuple[list[list[int]], list[str]]:
    cell_to_blocks: dict[tuple[int, int], list[str]] = {}
    for block_id in state["blocks"]:
        for cell in _block_cells(state, block_id):
            cell_to_blocks.setdefault(cell, []).append(block_id)

    overlap_cells: list[tuple[int, int]] = []
    involved: set[str] = set()
    for cell, bids in cell_to_blocks.items():
        if len(bids) > 1:
            overlap_cells.append(cell)
            involved.update(bids)

    cells_sorted = [[r, c] for r, c in sorted(overlap_cells)]
    return cells_sorted, sorted(involved)


def _out_of_bounds_cells(
    state: state_adapter.KlotskiState, block_id: str
) -> list[list[int]]:
    rows, cols = int(state["grid_size"][0]), int(state["grid_size"][1])
    oob: list[list[int]] = []
    for r, c in _block_cells(state, block_id):
        if r < 0 or c < 0 or r >= rows or c >= cols:
            oob.append([r, c])
    return sorted(oob)


__all__ = [
    "PAPER_STATE_ERROR_TYPES",
    "SUPPORTED_ERROR_TYPES",
    "generate_invalid_state",
    "generate_invalid_state_samples",
    "generate_invalid_state_coverage",
]
