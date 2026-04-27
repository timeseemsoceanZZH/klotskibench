#!/usr/bin/env python3
"""
Print per-depth exact-depth candidate_count for the current mainline reverse-BFS
configuration (goal seeds from :func:`enumerate_goal_seed_states`).

  python scripts/report_exact_depth_availability.py
  python scripts/report_exact_depth_availability.py --depth-min 1 --depth-max 15
  python scripts/report_exact_depth_availability.py --no-predecessor-filter
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap_paths

_bootstrap_paths.ensure_repo_sys_path()

from src.generators.exact_depth_stats import candidate_counts_in_range
from src.generators.goal_seed_enumerator import enumerate_goal_seed_states
from src.generators.reverse_bfs_dataset_builder import build_reverse_bfs_records


def main() -> None:
    p = argparse.ArgumentParser(
        description="Show exact-depth candidate_count per depth (multi-goal reverse BFS)."
    )
    p.add_argument("--depth-min", type=int, default=1)
    p.add_argument("--depth-max", type=int, default=15)
    p.add_argument(
        "--no-predecessor-filter",
        action="store_true",
        help="Use all valid solved goal encodings, not only those with a reverse predecessor.",
    )
    p.add_argument(
        "--max-goal-seeds",
        type=int,
        default=2_000,
        help="Cap goal seeds (0 = no cap; larger = slower BFS to --depth-max).",
    )
    args = p.parse_args()
    if args.depth_min < 1 or args.depth_max < args.depth_min:
        raise SystemExit("require 1 <= depth_min <= depth_max")

    mgs = None if int(args.max_goal_seeds) == 0 else int(args.max_goal_seeds)
    gs = enumerate_goal_seed_states(
        max_goals=mgs,
        require_predecessor_space=not bool(args.no_predecessor_filter),
    )
    print(f"goal_seed_count: {len(gs)} (require_predecessor_space={not bool(args.no_predecessor_filter)})")
    if not gs:
        print("No goal seeds; nothing to measure.", file=sys.stderr)
        raise SystemExit(1)
    rec = build_reverse_bfs_records(
        goal_seed_states=gs, max_depth=int(args.depth_max)
    )
    if not rec:
        print("reverse-BFS produced no records.", file=sys.stderr)
        raise SystemExit(1)
    print("exact-depth candidate_count (valid non-goal) per depth:")
    rows = candidate_counts_in_range(rec, int(args.depth_min), int(args.depth_max))
    for d, c in rows:
        print(f"  depth {d:2d}: {c}")
    lo = min(c for _, c in rows)
    print(f"minimum in [{args.depth_min},{args.depth_max}]: {lo}")


if __name__ == "__main__":
    main()
