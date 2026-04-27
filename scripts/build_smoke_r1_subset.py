#!/usr/bin/env python3
"""Build a fixed 6-case R1 smoke subset: depths 1..3, 2 per depth, seed 42 (default)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap_paths

_bootstrap_paths.ensure_repo_sys_path()

# Same directory as this script; needed so `import mini_bench_build_subset` resolves.
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import mini_bench_build_subset as mbs

SMOKE_OUT_DEFAULT = Path("output/smoke_r1/smoke_r1_subset.json")
SMOKE_MIN_DEPTH = 1
SMOKE_MAX_DEPTH = 3
SMOKE_PER_DEPTH = 2
SMOKE_SEED_DEFAULT = 42


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            f"Smoke R1 subset: depth {SMOKE_MIN_DEPTH}..{SMOKE_MAX_DEPTH}, "
            f"{SMOKE_PER_DEPTH} per depth, seed (default) {SMOKE_SEED_DEFAULT} → 6 cases."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=SMOKE_OUT_DEFAULT,
        help="Output JSON path (default: output/smoke_r1/smoke_r1_subset.json).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SMOKE_SEED_DEFAULT,
        help="Base random seed (per-depth: seed*1000+depth).",
    )
    p.add_argument(
        "--no-predecessor-filter",
        action="store_true",
        help="Forward to builder: skip predecessor filter on goal seeds.",
    )
    args = p.parse_args()
    n = mbs.run_build(
        out=args.out,
        min_depth=SMOKE_MIN_DEPTH,
        max_depth=SMOKE_MAX_DEPTH,
        per_depth=SMOKE_PER_DEPTH,
        seed=args.seed,
        no_predecessor_filter=bool(args.no_predecessor_filter),
    )
    print(f"Wrote {n} R1 smoke cases to {args.out}")


if __name__ == "__main__":
    main()
