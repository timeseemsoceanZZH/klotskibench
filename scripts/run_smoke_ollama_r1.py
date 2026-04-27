#!/usr/bin/env python3
"""Run the smoke R1 subset on Ollama (default: phi4-mini:latest). Reuses mini_bench exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap_paths

_bootstrap_paths.ensure_repo_sys_path()

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import mini_bench_run_ollama as mor

SMOKE_SUBSET_DEFAULT = Path("output/smoke_r1/smoke_r1_subset.json")
SMOKE_OUT_DIR_DEFAULT = Path("output/smoke_r1/run_phi4-mini")
SMOKE_MODEL_DEFAULT = "phi4-mini:latest"
SMOKE_MAX_CASES = 6


def main() -> None:
    p = argparse.ArgumentParser(
        description="Smoke run: 6 R1 cases via Ollama; writes eval JSON under --out-dir."
    )
    p.add_argument(
        "--subset",
        type=Path,
        default=SMOKE_SUBSET_DEFAULT,
        help="Subset JSON (default: output/smoke_r1/smoke_r1_subset.json).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=SMOKE_MODEL_DEFAULT,
        help="Ollama model tag (default: phi4-mini:latest).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=SMOKE_OUT_DIR_DEFAULT,
        help="Result directory (default: output/smoke_r1/run_phi4-mini).",
    )
    p.add_argument(
        "--ollama-base",
        default="http://127.0.0.1:11434",
        help="Ollama HTTP base URL",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=SMOKE_MAX_CASES,
        help="Cap case count (default: 6 for this smoke).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No Ollama HTTP; still writes all report files with empty trajectories.",
    )
    args = p.parse_args()
    mor.run_ollama_benchmark(
        subset=args.subset,
        model=args.model,
        out_dir=args.out_dir,
        ollama_base=args.ollama_base,
        max_cases=args.max_cases,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
