"""Build a reusable legal-state pool keyed by exact solution distance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core import state_adapter
from src.core import bfs_adapter


@dataclass(frozen=True)
class LegalStatePool:
    """Reusable legal-state pool indexed by exact distance to solution."""

    by_depth: dict[int, list[dict[str, Any]]]
    canonical_keys_by_depth: dict[int, list[state_adapter.CanonicalKey]]

    def items_at_depth(self, depth: int) -> list[dict[str, Any]]:
        """Return pooled legal-state items at one exact distance."""
        return list(self.by_depth.get(depth, []))

    def states_at_depth(self, depth: int) -> list[dict[str, Any]]:
        """Return normalized state payloads at one exact distance."""
        return [item["state"] for item in self.by_depth.get(depth, [])]

    def available_depths(self) -> list[int]:
        """Return sorted depths available in the pool."""
        return sorted(self.by_depth.keys())


def build_legal_state_pool(
    records: dict[state_adapter.CanonicalKey, bfs_adapter.StateRecord],
) -> LegalStatePool:
    """Construct legal-state pool from precomputed reverse-BFS records."""
    by_depth: dict[int, list[dict[str, Any]]] = {}
    canonical_keys_by_depth: dict[int, list[state_adapter.CanonicalKey]] = {}

    for canonical_key, record in records.items():
        depth = _record_depth(record)
        state = _record_state(record)

        by_depth.setdefault(depth, []).append(
            {
                "state": state,
                "optimal_depth": depth,
                "canonical_state_key": canonical_key,
            }
        )
        canonical_keys_by_depth.setdefault(depth, []).append(canonical_key)

    return LegalStatePool(
        by_depth=by_depth,
        canonical_keys_by_depth=canonical_keys_by_depth,
    )


def _record_depth(record: Any) -> int:
    """Extract integer depth from legacy-compatible state record."""
    if isinstance(record, dict):
        return int(record["depth"])
    return int(getattr(record, "depth"))


def _record_state(record: Any) -> state_adapter.KlotskiState:
    """Extract state payload from legacy-compatible state record."""
    if isinstance(record, dict):
        return record["state"]
    return getattr(record, "state")


__all__ = [
    "LegalStatePool",
    "build_legal_state_pool",
]
