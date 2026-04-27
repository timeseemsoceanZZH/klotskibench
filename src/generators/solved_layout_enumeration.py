"""Exhaustive enumeration of valid Hua Rong 5x4 board layouts with Caocao in the goal slot.

Used for reverse-BFS *goal-seed* diversity. Search: always cover the first empty
(row-major) with some remaining labeled piece, until all 9 are placed
(exact cover over labeled pieces; ~Algorithm X in spirit).
"""
from __future__ import annotations

import klotski_state as ks
from src.core import state_adapter

# Labeled block shapes (H1..H4, V1, S1..S4) from the benchmark
_PIECES: list[tuple[str, tuple[int, int]]] = [
    ("H1", (2, 1)),
    ("H2", (2, 1)),
    ("H3", (2, 1)),
    ("H4", (2, 1)),
    ("V1", (1, 2)),
    ("S1", (1, 1)),
    ("S2", (1, 1)),
    ("S3", (1, 1)),
    ("S4", (1, 1)),
]

_ROWS = 5
_COLS = 4
_CCC_POS = (3, 1)
_CCC_SHAPE = (2, 2)

_SOLVED_CACHE: list[ks.KlotskiState] | None = None


def _iter_first_empty(occ: list[list[int]]) -> tuple[int, int] | None:
    for r in range(_ROWS):
        for c in range(_COLS):
            if not occ[r][c]:
                return (r, c)
    return None


def _can_place(occ: list[list[int]], r: int, c: int, h: int, w: int) -> bool:
    if r + h > _ROWS or c + w > _COLS:
        return False
    for i in range(h):
        for j in range(w):
            if occ[r + i][c + j]:
                return False
    return True


def _set_rect(occ: list[list[int]], r: int, c: int, h: int, w: int, v: int) -> None:
    for i in range(h):
        for j in range(w):
            occ[r + i][c + j] = v


def _backtrack(
    rem: list[tuple[str, tuple[int, int]]],
    occ: list[list[int]],
    acc: list[tuple[str, int, int, int, int]],
    out: list[ks.KlotskiState],
) -> None:
    if not rem:
        st: ks.KlotskiState = {
            "grid_size": [_ROWS, _COLS],
            "blocks": {
                "Caocao": {
                    "shape": list(_CCC_SHAPE),
                    "pos": list(_CCC_POS),
                }
            }
            | {bid: {"shape": [h, w], "pos": [r, c]} for (bid, r, c, h, w) in acc},
        }
        if not ks.validate_state(st) or not state_adapter.is_goal_state(st):
            return
        out.append(ks.normalize_state(st))
        return

    fe = _iter_first_empty(occ)
    if fe is None:
        return
    fr, fc = fe

    for j, (bid, (h, w)) in enumerate(rem):
        for dr in range(h):
            for dc in range(w):
                r, c = fr - dr, fc - dc
                if r < 0 or c < 0 or r + h > _ROWS or c + w > _COLS:
                    continue
                if not (r <= fr < r + h and c <= fc < c + w):
                    continue
                if not _can_place(occ, r, c, h, w):
                    continue
                _set_rect(occ, r, c, h, w, 1)
                rest = rem[:j] + rem[j + 1 :]
                acc.append((bid, r, c, h, w))
                _backtrack(rest, occ, acc, out)
                acc.pop()
                _set_rect(occ, r, c, h, w, 0)


def enumerate_solved_huarong_states() -> list[ks.KlotskiState]:
    """All valid, deduplicated solved layouts, Caocao at [3,1]."""
    global _SOLVED_CACHE
    if _SOLVED_CACHE is not None:
        return list(_SOLVED_CACHE)
    occ: list[list[int]] = [[0] * _COLS for _ in range(_ROWS)]
    for i in range(_CCC_SHAPE[0]):
        for j in range(_CCC_SHAPE[1]):
            occ[_CCC_POS[0] + i][_CCC_POS[1] + j] = 1
    rem = list(_PIECES)
    raw: list[ks.KlotskiState] = []
    _backtrack(rem, occ, [], raw)
    seen: set[ks.CanonicalKey] = set()
    out: list[ks.KlotskiState] = []
    for s in raw:
        k = state_adapter.canonicalize_state(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    _SOLVED_CACHE = out
    return list(_SOLVED_CACHE)


__all__ = ["enumerate_solved_huarong_states"]
