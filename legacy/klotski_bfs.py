# Legacy reference: shortest-path BFS over the Klotski state graph for klotskibench.
# Not part of mainline `src/`; imported by `src.core.bfs_adapter` only.
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import klotski_moves as km
import klotski_state as ks

StateRecord = dict[str, Any]
ActionDict = dict[str, str]
CanonicalKey = ks.CanonicalKey
KlotskiState = ks.KlotskiState


def _n(s: KlotskiState) -> KlotskiState:
    if "blocks" in s and "grid_size" in s:
        return s
    return ks.normalize_state(s)


def _beq(x: KlotskiState, y: KlotskiState) -> bool:
    a, b = _n(x), _n(y)
    return a["blocks"] == b["blocks"] and a["grid_size"] == b["grid_size"]


def _action_from_to(s_from: KlotskiState, s_to: KlotskiState) -> Optional[ActionDict]:
    a0 = _n(s_from)
    t0 = _n(s_to)
    for a in km.get_legal_actions(a0):
        if _beq(km.apply_action(a0, a), t0):
            return a
    return None


def bfs_shortest_paths(
    start_state: KlotskiState,
    max_depth: Optional[int] = None,
) -> Dict[CanonicalKey, StateRecord]:
    s0 = _n(start_state)
    if not ks.validate_state(s0):
        return {}
    c0 = ks.canonicalize_state(s0)
    to_parent: dict[CanonicalKey, Tuple[CanonicalKey, ActionDict, KlotskiState]] = {}
    dist: dict[CanonicalKey, int] = {c0: 0}
    q: deque[Tuple[CanonicalKey, KlotskiState, int]] = deque([(c0, s0, 0)])
    seen: set[CanonicalKey] = {c0}
    while q:
        ccur, cur, d0 = q.popleft()
        for a in km.get_legal_actions(cur):
            n0 = _n(km.apply_action(cur, a))
            if not ks.validate_state(n0):
                continue
            kn = ks.canonicalize_state(n0)
            if kn in seen:
                continue
            nd = d0 + 1
            if max_depth is not None and nd > max_depth:
                continue
            to_parent[kn] = (ccur, a, cur)
            dist[kn] = nd
            seen.add(kn)
            q.append((kn, n0, nd))
    s0n = _n(s0)
    out: dict[CanonicalKey, StateRecord] = {}
    for k, d in dist.items():
        ps, pa = _rebuild_path_forward(c0, k, to_parent, s0n)
        if not ps and k != c0:
            continue
        psl = ps or [s0n]
        st = _n(psl[-1])
        out[k] = {
            "depth": d,
            "state": st,
            "path_actions": list(pa),
            "path_states": [_n(x) for x in psl],
        }
    return out


def _rebuild_path_forward(
    c0: CanonicalKey,
    k_end: CanonicalKey,
    to_parent: dict[CanonicalKey, Tuple[CanonicalKey, ActionDict, KlotskiState]],
    s0n: KlotskiState,
) -> Tuple[List[KlotskiState], List[ActionDict]]:
    if k_end == c0:
        return [s0n], []
    k = k_end
    rev_a: list[ActionDict] = []
    while k != c0:
        pr = to_parent.get(k)
        if pr is None:
            return [], []
        pcan, a, pstate0 = pr
        rev_a.append(a)
        k = pcan
    s_cur: KlotskiState = s0n
    ps2: list[KlotskiState] = [s_cur]
    pa2: list[ActionDict] = []
    for a in reversed(rev_a):
        s_cur2 = _n(km.apply_action(s_cur, a))
        pa2.append(a)
        s_cur = s_cur2
        ps2.append(s_cur)
    return ps2, pa2


def reverse_bfs_shortest_paths(
    goal_seed_state: KlotskiState,
    is_goal_fn: Callable[[KlotskiState], bool],
    max_depth: Optional[int] = None,
) -> Dict[CanonicalKey, StateRecord]:
    g0 = _n(goal_seed_state)
    if not is_goal_fn(g0):
        return {}
    return reverse_bfs_shortest_paths_multi([g0], is_goal_fn, max_depth)


def reverse_bfs_shortest_paths_multi(
    goal_seed_states: List[KlotskiState],
    is_goal_fn: Callable[[KlotskiState], bool],
    max_depth: Optional[int] = None,
) -> Dict[CanonicalKey, StateRecord]:
    gset: set[CanonicalKey] = set()
    goals0: list[KlotskiState] = []
    for s in goal_seed_states:
        g0 = _n(s)
        if not is_goal_fn(g0) or not ks.validate_state(g0):
            continue
        ck = ks.canonicalize_state(g0)
        if ck in gset:
            continue
        gset.add(ck)
        goals0.append(g0)
    if not goals0:
        return {}
    key_state: dict[CanonicalKey, KlotskiState] = {ks.canonicalize_state(_n(g)): _n(g) for g in goals0}
    to_next: dict[CanonicalKey, Tuple[ActionDict, CanonicalKey]] = {}
    dist: dict[CanonicalKey, int] = {k: 0 for k in key_state}
    q: deque[Tuple[CanonicalKey, KlotskiState, int]] = deque((k, st, 0) for k, st in key_state.items())
    seen: set[CanonicalKey] = set(key_state)
    gkeys: set[CanonicalKey] = set(key_state)
    while q:
        c_p, p_n, d0 = q.popleft()
        if max_depth is not None and d0 >= max_depth:
            continue
        p_state = _n(p_n)
        for s_cand in km.enumerate_predecessor_states(p_state):
            s0_ = _n(s_cand)
            a = _action_from_to(s0_, p_state)
            if a is None or not ks.validate_state(s0_):
                continue
            sk = ks.canonicalize_state(s0_)
            if sk in seen:
                continue
            nd = d0 + 1
            if max_depth is not None and nd > max_depth:
                continue
            to_next[sk] = (a, c_p)
            dist[sk] = nd
            key_state[sk] = s0_
            seen.add(sk)
            q.append((sk, s0_, nd))
    out: dict[CanonicalKey, StateRecord] = {}
    for gk in gkeys:
        stg = key_state[gk]
        out[gk] = {
            "depth": 0,
            "state": stg,
            "path_actions": [],
            "path_states": [stg],
        }
    for k, d in dist.items():
        if k in gkeys:
            continue
        s0_ = key_state.get(k)
        if s0_ is None:
            continue
        ppath, pacts = _walk_toward_goal(
            s0_, k, gkeys, to_next, key_state
        )
        if not ppath:
            continue
        st0 = _n(ppath[0])
        out[k] = {
            "depth": d,
            "state": st0,
            "path_actions": list(pacts),
            "path_states": [_n(s) for s in ppath],
        }
    return out


def _walk_toward_goal(
    s0: KlotskiState,
    s_key: CanonicalKey,
    gset: set[CanonicalKey],
    to_next: dict[CanonicalKey, Tuple[ActionDict, CanonicalKey]],
    key_state: dict[CanonicalKey, KlotskiState],
) -> Tuple[List[KlotskiState], List[ActionDict]]:
    ps: list[KlotskiState] = []
    pa: list[ActionDict] = []
    cur = _n(s0)
    ck = s_key
    ps.append(cur)
    for _ in range(100_000):
        if ck in gset:
            return ps, pa
        t = to_next.get(ck)
        if t is None:
            return ps, pa
        a, c_parent = t
        nxt = _n(km.apply_action(cur, a))
        c_n = ks.canonicalize_state(nxt)
        p_ref = key_state.get(c_parent)
        nxt2 = p_ref if p_ref is not None and c_n == ks.canonicalize_state(p_ref) else nxt
        nxt2 = _n(nxt2)
        pa.append(a)
        cur = nxt2
        ps.append(cur)
        ck = c_parent
    return ps, pa


def states_at_depth(
    records: Dict[CanonicalKey, StateRecord], depth: int
) -> List[StateRecord]:
    return [r for r in records.values() if int(r["depth"]) == int(depth)]


def states_in_depth_range(
    records: Dict[CanonicalKey, StateRecord], depth_min: int, depth_max: int
) -> List[StateRecord]:
    return [r for r in records.values() if depth_min <= int(r["depth"]) <= depth_max]
