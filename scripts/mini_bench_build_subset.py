#!/usr/bin/env python3
"""
Build a small R1 (trajectory generation) mini-benchmark subset: depths 1..D,
K samples per depth, from reverse-BFS on benchmark goal seeds.

Usage:
  python scripts/mini_bench_build_subset.py --out output/mini_r1/subset_r1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap_paths

_bootstrap_paths.ensure_repo_sys_path()

from src.generators.goal_seed_enumerator import (
    DEFAULT_GOAL_SEED_CANDIDATES,
    enumerate_goal_seed_states,
)
from src.generators.reverse_bfs_dataset_builder import (
    build_exact_depth_cases,
    build_reverse_bfs_records,
)
from src.tasks.r1_builder import build_r1_cases


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


def run_build(
    out: Path,
    min_depth: int,
    max_depth: int,
    per_depth: int,
    seed: int,
    no_predecessor_filter: bool,
) -> int:
    """Build R1 subset file; return number of cases written."""
    goal_seeds: list[dict]
    if no_predecessor_filter:
        goal_seeds = enumerate_goal_seed_states(
            max_goals=None, require_predecessor_space=False
        )
    else:
        goal_seeds = enumerate_goal_seed_states(
            max_goals=None, require_predecessor_space=True
        )
    if not goal_seeds:
        print(
            "Warning: no goal seeds from enumerator; using DEFAULT_GOAL_SEED_CANDIDATES.",
            file=sys.stderr,
        )
        goal_seeds = [dict(s) for s in DEFAULT_GOAL_SEED_CANDIDATES]

    records = build_reverse_bfs_records(
        goal_seed_states=goal_seeds,
        max_depth=max_depth,
    )
    if not records:
        raise SystemExit("reverse-BFS produced no states; check legacy klotski imports.")

    all_exact: list[dict] = []
    for depth in range(min_depth, max_depth + 1):
        per_seed = seed * 1000 + depth
        batch = build_exact_depth_cases(
            records=records,
            depth=depth,
            num_samples=per_depth,
            seed=per_seed,
        )
        all_exact.extend(batch)

    r1_cases = build_r1_cases(all_exact)
    payload = {
        "schema": "klotskibench.mini_r1_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "min_depth": min_depth,
        "max_depth": max_depth,
        "per_depth": per_depth,
        "seed": seed,
        "n_cases": len(r1_cases),
        "goal_seed_count": len(goal_seeds),
        "cases": r1_cases,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)

    return len(r1_cases)


def main() -> None:
    p = argparse.ArgumentParser(description="Build mini R1 evaluation subset (exact-depth).")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("output/mini_r1/mini_r1_subset.json"),
        help="Output JSON path",
    )
    p.add_argument("--min-depth", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--per-depth", type=int, default=10)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; per-depth uses seed*1000+depth for sampling stability.",
    )
    p.add_argument(
        "--no-predecessor-filter",
        action="store_true",
        help="Use all enumerated goal seeds (faster; default uses predecessor filter on seeds).",
    )
    args = p.parse_args()
    n = run_build(
        out=args.out,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        per_depth=args.per_depth,
        seed=args.seed,
        no_predecessor_filter=bool(args.no_predecessor_filter),
    )
    print(f"Wrote {n} R1 cases to {args.out}")


if __name__ == "__main__":
    main()
