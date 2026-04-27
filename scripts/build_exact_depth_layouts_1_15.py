#!/usr/bin/env python3
"""
Build benchmark-ready exact-depth layout cases: for each depth N, sample K distinct
states at exactly N reverse-BFS steps from a solved state (current mainline pipeline).

The reverse-BFS front is built from *all* valid solved-state goal seeds (Caocao at
``[3,1]``; see :mod:`src.generators.goal_seed_enumerator`). The per-depth
``candidate_count`` is still finite; if ``--per-depth`` exceeds it for a depth, the
script fails after printing an availability table. Use ``--print-availability`` or
``scripts/report_exact_depth_availability.py`` to inspect caps.

Usage (from repository root):
  python scripts/build_exact_depth_layouts_1_15.py --depth-min 1 --depth-max 15 --per-depth 10 --seed 42
  python scripts/build_exact_depth_layouts_1_15.py --print-availability
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap_paths

_bootstrap_paths.ensure_repo_sys_path()

from src.core import moves_adapter, state_adapter
from src.generators.exact_depth_stats import candidate_count_at_depth
from src.generators.goal_seed_enumerator import (
    DEFAULT_GOAL_SEED_CANDIDATES,
    enumerate_goal_seed_states,
)
from src.generators.reverse_bfs_dataset_builder import (
    build_exact_depth_cases_with_stats,
    build_reverse_bfs_records,
)


def _print_availability_from_records(
    rec: dict[Any, Any] | None,
    depth_min: int,
    depth_max: int,
) -> int:
    """Print a table; return the minimum candidate_count in the inclusive depth range."""
    print("exact-depth candidate_count (valid non-goal states) per depth:")
    low = 10**9
    for d in range(depth_min, depth_max + 1):
        c = candidate_count_at_depth(rec, d) if rec else 0
        if c < low:
            low = c
        print(f"  depth {d:2d}: {c}")
    if rec is not None and low < 10**9:
        print(
            f"(minimum in [{depth_min},{depth_max}] is {low} — use --per-depth <= {low} "
            "to export that full range, or reduce --depth-min/--depth-max)\n"
        )
    return low if rec else 0


def _enumerate_goal_seeds(
    no_predecessor_filter: bool, max_goal_seeds: int | None
) -> list[dict]:
    if no_predecessor_filter:
        goal_seeds = enumerate_goal_seed_states(
            max_goals=max_goal_seeds, require_predecessor_space=False
        )
    else:
        goal_seeds = enumerate_goal_seed_states(
            max_goals=max_goal_seeds, require_predecessor_space=True
        )
    if not goal_seeds:
        print(
            "Warning: no goal seeds; using DEFAULT_GOAL_SEED_CANDIDATES.",
            file=sys.stderr,
        )
        goal_seeds = [dict(s) for s in DEFAULT_GOAL_SEED_CANDIDATES]
    return goal_seeds


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    return str(obj)


def _normalize_case_dict(c: dict[str, Any], depth: int) -> dict[str, Any]:
    out = dict(c)
    out["initial_state"] = state_adapter.normalize_state(c["initial_state"])
    out["goal_state"] = state_adapter.normalize_state(c["goal_state"])
    out["optimal_depth"] = int(c["optimal_depth"])
    out["shortest_solution_actions"] = [dict(a) for a in c["shortest_solution_actions"]]
    out["shortest_solution_states"] = [
        state_adapter.normalize_state(s) for s in c["shortest_solution_states"]
    ]
    if "meta" in c:
        m = dict(c["meta"])
        m["canonical_state_key"] = state_adapter.canonicalize_state(out["initial_state"])
        m["difficulty_bucket"] = f"depth_{depth}"
        out["meta"] = m
    return out


def _forward_reach_goal(c: dict[str, Any], init: Any, goal: Any, depth: int) -> str | None:
    """Full forward re-simulation of the shortest-solution action chain (cross-check)."""
    acts = c.get("shortest_solution_actions") or []
    if depth == 0:
        if state_adapter.canonicalize_state(init) != state_adapter.canonicalize_state(goal):
            return "depth0: init != goal canonical"
        return None
    cur = init
    for i, a in enumerate(acts):
        try:
            cur = moves_adapter.apply_action(cur, dict(a))
        except Exception as e:  # noqa: BLE001
            return f"forward step {i}: {e!r}"
        if not state_adapter.validate_state(cur):
            return f"forward step {i}: invalid state after action"
    if state_adapter.canonicalize_state(cur) != state_adapter.canonicalize_state(goal):
        return "forward re-sim final state != goal (canonical)"
    return None


def _validate_case(
    c: dict[str, Any], expected_depth: int, do_forward: bool
) -> list[str]:
    """Return list of error strings; empty if ok."""
    errs: list[str] = []
    if c.get("optimal_depth") != expected_depth:
        errs.append(f"optimal_depth {c.get('optimal_depth')} != {expected_depth}")
    try:
        init = state_adapter.normalize_state(c["initial_state"])
    except Exception as e:  # noqa: BLE001
        return [f"initial_state normalize: {e!r}"]
    if not state_adapter.validate_state(init):
        errs.append("initial_state not validate_state-ok")
    try:
        goal = state_adapter.normalize_state(c["goal_state"])
    except Exception as e:  # noqa: BLE001
        return [f"goal_state normalize: {e!r}"]
    if not state_adapter.is_goal_state(goal):
        errs.append("goal_state is not a solved/goal state per is_goal_state")
    acts = c.get("shortest_solution_actions") or []
    if len(acts) != expected_depth:
        errs.append(f"len(actions)={len(acts)} != depth {expected_depth}")
    pss = c.get("shortest_solution_states") or []
    if len(pss) != expected_depth + 1:
        errs.append(f"len(path_states)={len(pss)} != depth+1={expected_depth+1}")
    if pss and state_adapter.canonicalize_state(pss[0]) != state_adapter.canonicalize_state(
        init
    ):
        errs.append("path_states[0] != initial_state (canonical)")
    if pss and state_adapter.canonicalize_state(pss[-1]) != state_adapter.canonicalize_state(
        goal
    ):
        errs.append("path_states[-1] != goal_state (canonical)")
    if not errs and do_forward:
        m = _forward_reach_goal(c, init, goal, expected_depth)
        if m:
            errs.append(m)
    return errs


def _validate_depth_file(
    cases: list[dict[str, Any]], depth: int, per_depth: int, forward_first_k: int
) -> list[str]:
    out: list[str] = []
    if len(cases) != per_depth:
        out.append(f"expected {per_depth} cases, got {len(cases)}")
    seen: set = set()
    for i, c in enumerate(cases):
        e = c.get("meta", {}).get("canonical_state_key")
        if e in seen:
            out.append(f"duplicate canonical_state_key at index {i}")
        seen.add(e)
        do_f = i < forward_first_k
        for msg in _validate_case(c, depth, do_f):
            out.append(f"case[{i}]: {msg}")
    return out


@dataclass
class BuildConfig:
    depth_min: int
    depth_max: int
    per_depth: int
    seed: int
    out_dir: Path
    no_predecessor_filter: bool
    write_combined: bool
    forward_full_reach_goal_first_k: int
    max_goal_seeds: int | None


def _run(cfg: BuildConfig) -> list[Path]:
    if cfg.depth_min < 1 or cfg.depth_max < cfg.depth_min:
        raise SystemExit("require 1 <= depth_min <= depth_max")
    if cfg.per_depth < 1:
        raise SystemExit("per_depth must be >= 1")

    goal_seeds = _enumerate_goal_seeds(
        cfg.no_predecessor_filter, cfg.max_goal_seeds
    )

    records = build_reverse_bfs_records(
        goal_seed_states=goal_seeds,
        max_depth=cfg.depth_max,
    )
    if not records:
        raise SystemExit("reverse-BFS produced no states; check legacy/ engine.")

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    by_depth: dict[str, list[dict[str, Any]]] = {}

    for depth in range(cfg.depth_min, cfg.depth_max + 1):
        per_seed = cfg.seed * 1000 + depth
        raw_cases, stats = build_exact_depth_cases_with_stats(
            records=records,
            depth=depth,
            num_samples=cfg.per_depth,
            seed=per_seed,
        )
        cand = stats["candidate_count"]
        if cand < cfg.per_depth:
            print("Availability for this BFS (same goal seeds):", file=sys.stderr)
            _print_availability_from_records(records, cfg.depth_min, cfg.depth_max)
            raise SystemExit(
                f"depth {depth}: need {cfg.per_depth} distinct exact-depth candidates, "
                f"only {cand} available. See table above, or use --print-availability, "
                f"or lower --per-depth / --depth-max."
            )
        if len(raw_cases) != cfg.per_depth:
            raise SystemExit(
                f"depth {depth}: exported {len(raw_cases)} cases, expected {cfg.per_depth}."
            )
        cases: list[dict[str, Any]] = []
        for i, c in enumerate(raw_cases):
            nc = _normalize_case_dict(c, depth)
            nc["case_id"] = f"depth{depth:02d}_case{i:02d}"
            cases.append(nc)

        v_errs = _validate_depth_file(
            cases, depth, cfg.per_depth, cfg.forward_full_reach_goal_first_k
        )
        if v_errs:
            raise SystemExit(f"validation failed depth {depth}:\n" + "\n".join(v_errs))

        path = out_dir / f"dataset_N{depth}.json"
        payload = {
            "schema": "klotskibench.exact_depth_layouts_per_depth_v1",
            "depth": depth,
            "per_depth": cfg.per_depth,
            "seed": cfg.seed,
            "sampling_subseed": per_seed,
            "n_cases": len(cases),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "goal_seed_count": len(goal_seeds),
            "max_goal_seeds": cfg.max_goal_seeds,
            "cases": cases,
        }
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
        by_depth[str(depth)] = cases
        print(f"depth {depth}: exported {len(cases)} cases -> {path}")

    if cfg.write_combined:
        comb = out_dir / f"dataset_N{cfg.depth_min}_{cfg.depth_max}_{cfg.per_depth}each.json"
        comb_payload = {
            "schema": "klotskibench.exact_depth_layouts_combined_v1",
            "depth_min": cfg.depth_min,
            "depth_max": cfg.depth_max,
            "per_depth": cfg.per_depth,
            "seed": cfg.seed,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "goal_seed_count": len(goal_seeds),
            "max_goal_seeds": cfg.max_goal_seeds,
            "per_depth_files": {str(d): f"dataset_N{d}.json" for d in range(cfg.depth_min, cfg.depth_max + 1)},
            "cases_by_depth": by_depth,
        }
        comb.write_text(
            json.dumps(_jsonable(comb_payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(comb)
        print(f"combined -> {comb}")

    print()
    print("Output paths:")
    for p in written:
        print(f"  {p.resolve()}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build exact-depth Klotski layout sets (reverse-BFS, mainline only)."
    )
    p.add_argument("--depth-min", type=int, default=1)
    p.add_argument("--depth-max", type=int, default=15)
    p.add_argument(
        "--per-depth",
        type=int,
        default=10,
        help="How many cases to sample per depth (fails with availability table if insufficient).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; per-depth sampling uses seed*1000+depth (same as mini_bench).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/exact_depth"),
        help="Directory for dataset_N{depth}.json and optional combined file",
    )
    p.add_argument(
        "--no-predecessor-filter",
        action="store_true",
        help="Use all enumerated goal seeds (see goal_seed_enumerator).",
    )
    p.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not write the combined all-depth JSON.",
    )
    p.add_argument(
        "--print-availability",
        action="store_true",
        help="Build reverse-BFS and print per-depth exact-depth candidate_count, then exit.",
    )
    p.add_argument(
        "--max-goal-seeds",
        type=int,
        default=2_000,
        help="Cap solved goal seeds (after canonical sort; 0 = no cap, use all ~10^5). Higher (e.g. 8000) can improve diversity but multiplies reverse-BFS time.",
    )
    p.add_argument(
        "--forward-cases",
        type=int,
        default=2,
        help="For the first K cases per depth, run a full forward re-sim of shortest_solution_actions to reach goal (0 to disable).",
    )
    args = p.parse_args()
    mgs: int | None
    if int(args.max_goal_seeds) == 0:
        mgs = None
    else:
        mgs = int(args.max_goal_seeds)
    if args.print_availability:
        if args.depth_min < 1 or args.depth_max < args.depth_min:
            raise SystemExit("require 1 <= depth_min <= depth_max")
        gs = _enumerate_goal_seeds(bool(args.no_predecessor_filter), mgs)
        rec = build_reverse_bfs_records(
            goal_seed_states=gs, max_depth=int(args.depth_max)
        )
        if not rec:
            raise SystemExit("reverse-BFS produced no states; check legacy/ engine.")
        _print_availability_from_records(rec, int(args.depth_min), int(args.depth_max))
        return
    _run(
        BuildConfig(
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            per_depth=args.per_depth,
            seed=args.seed,
            out_dir=args.out_dir,
            no_predecessor_filter=bool(args.no_predecessor_filter),
            write_combined=not args.no_combined,
            forward_full_reach_goal_first_k=max(0, int(args.forward_cases)),
            max_goal_seeds=mgs,
        )
    )


if __name__ == "__main__":
    main()
