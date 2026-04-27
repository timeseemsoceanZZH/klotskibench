"""Controlled corruption utilities for trajectory task generation."""

from __future__ import annotations

import random

V1_INVALID_TRAJECTORY_TYPES: tuple[str, ...] = (
    "wrong_action",
    "truncated_suffix",
    "extended_noise",
)


def corrupt_trajectory(
    actions: list[dict[str, str]],
    invalid_type: str,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Create controlled invalid trajectory from oracle shortest actions."""
    if invalid_type not in V1_INVALID_TRAJECTORY_TYPES:
        raise ValueError(f"Unsupported invalid trajectory type: {invalid_type}")

    if not actions:
        return [{"block_id": "Caocao", "direction": "up"}]

    corrupted = [dict(a) for a in actions]
    if invalid_type == "wrong_action":
        idx = rng.randrange(len(corrupted))
        corrupted[idx] = {
            "block_id": corrupted[idx]["block_id"],
            "direction": _different_direction(corrupted[idx]["direction"]),
        }
        return corrupted

    if invalid_type == "truncated_suffix":
        if len(corrupted) == 1:
            corrupted.append({"block_id": "Caocao", "direction": "up"})
            return corrupted
        cut = rng.randrange(1, len(corrupted))
        truncated = corrupted[:cut]
        truncated.append({"block_id": "Caocao", "direction": "up"})
        return truncated

    corrupted.append({"block_id": "Caocao", "direction": "up"})
    return corrupted


def _different_direction(direction: str) -> str:
    for candidate in ("up", "down", "left", "right"):
        if candidate != direction:
            return candidate
    return "up"


__all__ = [
    "V1_INVALID_TRAJECTORY_TYPES",
    "corrupt_trajectory",
]
