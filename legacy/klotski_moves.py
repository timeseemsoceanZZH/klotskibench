# Legacy reference: one-step move generation and dynamics for klotskibench.
# Not part of mainline `src/`; imported by `src.core.moves_adapter` only.
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from klotski_state import KlotskiState, normalize_state, validate_state

ActionDict = dict[str, str]

_DIRS = ("up", "down", "left", "right")
_DVEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _norm(s: KlotskiState) -> KlotskiState:
    if "blocks" in s and "grid_size" in s:
        return s
    return normalize_state(s)


def _copy(s: KlotskiState) -> KlotskiState:
    return deepcopy(s)


def _cells(
    r: int, c: int, h: int, w: int
) -> list[tuple[int, int]]:
    return [(r + i, c + j) for i in range(h) for j in range(w)]


def build_occupied_by_id(state: KlotskiState) -> dict[tuple[int, int], str]:
    m: dict[tuple[int, int], str] = {}
    st = _norm(state)
    for bid, b in st["blocks"].items():
        r, c = int(b["pos"][0]), int(b["pos"][1])
        h, w = int(b["shape"][0]), int(b["shape"][1])
        for cell in _cells(r, c, h, w):
            m[cell] = str(bid)
    return m


def get_legal_actions(state: KlotskiState) -> list[ActionDict]:
    s = _norm(state)
    if not validate_state(s):
        return []
    occ = build_occupied_by_id(s)
    rows, cols = int(s["grid_size"][0]), int(s["grid_size"][1])
    out: list[ActionDict] = []
    for bid in sorted(s["blocks"].keys(), key=str):
        b = s["blocks"][bid]
        r, c = int(b["pos"][0]), int(b["pos"][1])
        h, w = int(b["shape"][0]), int(b["shape"][1])
        tr, br = r, r + h - 1
        tc, bcol = c, c + w - 1
        for d in _DIRS:
            dr, dc = _DVEC[d]
            ok = True
            if dr < 0:
                # move up: row above the top edge must be empty
                for cc in range(tc, bcol + 1):
                    p = (tr - 1, cc)
                    if p[0] < 0 or p in occ:
                        ok = False
                        break
            elif dr > 0:
                for cc in range(tc, bcol + 1):
                    p = (br + 1, cc)
                    if p[0] >= rows or p in occ:
                        ok = False
                        break
            elif dc < 0:
                for rr in range(tr, br + 1):
                    p = (rr, tc - 1)
                    if p[1] < 0 or p in occ:
                        ok = False
                        break
            else:
                for rr in range(tr, br + 1):
                    p = (rr, bcol + 1)
                    if p[1] >= cols or p in occ:
                        ok = False
                        break
            if ok:
                out.append({"block_id": str(bid), "direction": d})
    return out


def apply_action(state: KlotskiState, action: ActionDict) -> KlotskiState:
    s = _copy(_norm(state))
    bid = str(action["block_id"])
    d = str(action["direction"])
    if bid not in s["blocks"] or d not in _DVEC:
        return s
    dr, dc = _DVEC[d]
    s["blocks"][bid]["pos"][0] += dr
    s["blocks"][bid]["pos"][1] += dc
    return s


def enumerate_predecessor_states(p_state: KlotskiState) -> list[KlotskiState]:
    """States S for which a single legal action yields `p_state`."""
    from klotski_state import canonicalize_state

    p = _norm(p_state)
    if not validate_state(p):
        return []
    res: list[KlotskiState] = []
    seen: set = set()
    for bid in p["blocks"].keys():
        for d in _DIRS:
            dr, dc = _DVEC[d]
            a: ActionDict = {"block_id": str(bid), "direction": d}
            s = _copy(p)
            s["blocks"][str(bid)]["pos"][0] -= dr
            s["blocks"][str(bid)]["pos"][1] -= dc
            s = _norm(s)
            if not validate_state(s):
                continue
            if a not in get_legal_actions(s):
                continue
            if _blocks_equal(apply_action(s, a), p):
                ckey = canonicalize_state(s)
                if ckey in seen:
                    continue
                seen.add(ckey)
                res.append(s)
    return res


def _blocks_equal(
    a: KlotskiState, b: KlotskiState
) -> bool:
    x, y = _norm(a), _norm(b)
    return x["blocks"] == y["blocks"] and x["grid_size"] == y["grid_size"]
