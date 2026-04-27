# Legacy reference: normalized Klotski / Hua Rong state helpers for klotskibench.
# Not part of mainline `src/`; imported by `src.core.state_adapter` only.
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Normalized in-memory shape used by the benchmark pipeline.
KlotskiState = Dict[str, Any]
CanonicalKey = Tuple[int, ...]

# Benchmark reference layout (solved) — must match `DEFAULT_GOAL_SEED_CANDIDATES[0]`
# in `src.generators.goal_seed_enumerator` after normalize.
_BENCHMARK_DEFAULT_FLAT: dict[str, Any] = {
    "Caocao": {"shape": [2, 2], "pos": [3, 1]},
    "H1": {"shape": [2, 1], "pos": [0, 0]},
    "H2": {"shape": [2, 1], "pos": [0, 3]},
    "H3": {"shape": [2, 1], "pos": [2, 0]},
    "H4": {"shape": [2, 1], "pos": [2, 3]},
    "V1": {"shape": [1, 2], "pos": [2, 1]},
    "S1": {"shape": [1, 1], "pos": [0, 1]},
    "S2": {"shape": [1, 1], "pos": [0, 2]},
    "S3": {"shape": [1, 1], "pos": [4, 0]},
    "S4": {"shape": [1, 1], "pos": [4, 3]},
    "Grid_Size": [5, 4],
}


def _as_nested(raw: dict[str, Any]) -> dict[str, Any]:
    if "blocks" in raw and "grid_size" in raw:
        out = {
            "grid_size": [int(x) for x in list(raw["grid_size"])],
            "blocks": {},
        }
        for bid, b in raw["blocks"].items():
            out["blocks"][str(bid)] = {
                "shape": [int(b["shape"][0]), int(b["shape"][1])],
                "pos": [int(b["pos"][0]), int(b["pos"][1])],
            }
        return out
    gs_key = "Grid_Size" if "Grid_Size" in raw else "grid_size"
    grid = [int(x) for x in list(raw[gs_key])]
    blocks: dict[str, Any] = {}
    for k, v in raw.items():
        if k in (gs_key, "blocks", "grid_size", "Grid_Size"):
            continue
        if not isinstance(v, dict) or "pos" not in v:
            continue
        blocks[str(k)] = {
            "shape": [int(v["shape"][0]), int(v["shape"][1])],
            "pos": [int(v["pos"][0]), int(v["pos"][1])],
        }
    return {"grid_size": grid, "blocks": blocks}


def normalize_state(state: Any) -> KlotskiState:
    if not isinstance(state, dict):
        raise TypeError("state must be a dict")
    n = _as_nested(state)
    n["grid_size"] = [int(n["grid_size"][0]), int(n["grid_size"][1])]
    return n


def _iter_cells(
    r: int, c: int, h: int, w: int
) -> list[tuple[int, int]]:
    return [(r + i, c + j) for i in range(h) for j in range(w)]


def to_flat_state(state: KlotskiState) -> dict[str, Any]:
    """Back to the flat Hua Rong / benchmark JSON shape."""
    rows, cols = int(state["grid_size"][0]), int(state["grid_size"][1])
    out: dict[str, Any] = {"Grid_Size": [rows, cols]}
    for bid, b in state["blocks"].items():
        out[str(bid)] = {"shape": list(b["shape"]), "pos": list(b["pos"])}
    return out


def _occupied(state: KlotskiState) -> set[tuple[int, int]]:
    o: set[tuple[int, int]] = set()
    for b in state["blocks"].values():
        r, c = int(b["pos"][0]), int(b["pos"][1])
        h, w = int(b["shape"][0]), int(b["shape"][1])
        for (rr, cc) in _iter_cells(r, c, h, w):
            o.add((rr, cc))
    return o


def validate_state(state: KlotskiState) -> bool:
    if not state.get("blocks") or "grid_size" not in state:
        return False
    rows, cols = int(state["grid_size"][0]), int(state["grid_size"][1])
    o: set[tuple[int, int]] = set()
    for b in state["blocks"].values():
        r, c = int(b["pos"][0]), int(b["pos"][1])
        h, w = int(b["shape"][0]), int(b["shape"][1])
        if h <= 0 or w <= 0 or r < 0 or c < 0 or r + h > rows or c + w > cols:
            return False
        for cell in _iter_cells(r, c, h, w):
            if cell in o:
                return False
            o.add(cell)
    return len(o) > 0


def benchmark_reference_goal() -> KlotskiState:
    """The classic benchmark layout, normalized (for tests and R1 exact-goal references)."""
    return normalize_state(_BENCHMARK_DEFAULT_FLAT)


def canonicalize_state(state: KlotskiState) -> CanonicalKey:
    s = state
    if "blocks" not in s or "grid_size" not in s:
        s = normalize_state(s)
    out: list[int] = [int(s["grid_size"][0]) * 1000 + int(s["grid_size"][1])]
    for bid in sorted(s["blocks"].keys(), key=str):
        b = s["blocks"][bid]
        out += [
            int(b["pos"][0]),
            int(b["pos"][1]),
            int(b["shape"][0]),
            int(b["shape"][1]),
        ]
        for ch in str(bid):
            out.append(ord(ch))
    return tuple(int(x) for x in out)


def is_goal_state(state: KlotskiState) -> bool:
    """Solved iff the state is valid and the 2x2 Cao Cao block sits in the exit at [3,1].

    Other block placements may differ; the benchmark R1 success metric still requires an
    exact per-case :func:`canonicalize_state` match to the packaged ``meta.goal_state``.
    """
    s = normalize_state(state) if "blocks" not in state or "grid_size" not in state else state
    if not validate_state(s):
        return False
    cc = s["blocks"].get("Caocao")
    if not cc or "pos" not in cc or "shape" not in cc:
        return False
    r, c = int(cc["pos"][0]), int(cc["pos"][1])
    h, w = int(cc["shape"][0]), int(cc["shape"][1])
    if r != 3 or c != 1 or h != 2 or w != 2:
        return False
    return True
