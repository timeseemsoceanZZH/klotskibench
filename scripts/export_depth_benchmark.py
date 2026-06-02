#!/usr/bin/env python3
"""
Build a depth-stratified seven-task benchmark preserving the task-specific
directory structure expected by scripts/run_mini_ollama_all_tasks.py.

Usage (smoke — 10 cases per depth per task, 700 total):
  python scripts/export_depth_benchmark.py \
      --out-dir output/seven_task_depth1_10_10_per_depth \
      --depth-min 1 --depth-max 10 --cases-per-depth-per-task 10

Usage (full — 100 cases per depth per task, 7000 total):
  python scripts/export_depth_benchmark.py \
      --out-dir output/seven_task_depth1_10_100_per_depth \
      --depth-min 1 --depth-max 10 --cases-per-depth-per-task 100

Coverage is a hard requirement by default.  S1 and T3 each need ≥6 cases per
depth (one per required bucket), S2 and R2 need ≥5.  Pass
--allow-partial-coverage to disable the strict check and export whatever is
available (useful for debugging or low-depth exploratory runs).

Output layout:
  <out-dir>/
    s1/cases.json
    s2/cases.json
    t1/cases.json
    t2/cases.json
    t3/cases.json
    r1/cases.json
    r2/cases.json
    coverage_summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import _bootstrap_paths  # noqa: E402

_bootstrap_paths.ensure_repo_sys_path()

from src.generators.goal_seed_enumerator import (  # noqa: E402
    DEFAULT_GOAL_SEED_CANDIDATES,
    enumerate_goal_seed_states,
)
from src.generators.invalid_state_generator import PAPER_STATE_ERROR_TYPES  # noqa: E402
from src.generators.legal_state_pool import build_legal_state_pool  # noqa: E402
from src.generators.reverse_bfs_dataset_builder import (  # noqa: E402
    build_exact_depth_cases,
    build_reverse_bfs_records,
)
from src.generators.trajectory_corruptor import R2_PAPER_INVALID_TRAJECTORY_TYPES  # noqa: E402
from src.generators.transition_oracle import T3_PAPER_INVALID_REASONS  # noqa: E402
from src.tasks.r1_builder import build_r1_cases  # noqa: E402
from src.tasks.r2_builder import build_r2_cases  # noqa: E402
from src.tasks.s1_builder import build_s1_cases  # noqa: E402
from src.tasks.s2_builder import build_s2_cases  # noqa: E402
from src.tasks.t1_builder import build_t1_cases  # noqa: E402
from src.tasks.t2_builder import build_t2_cases  # noqa: E402
from src.tasks.t3_builder import V1_INVALID_REASONS, build_t3_cases  # noqa: E402

ALL_TASKS: tuple[str, ...] = ("s1", "s2", "t1", "t2", "t3", "r1", "r2")
_TASK_IDX: dict[str, int] = {t: i for i, t in enumerate(ALL_TASKS)}

# Coverage guarantee buckets for smart-sampling per task
_S1_BUCKETS: list[str] = ["valid"] + list(PAPER_STATE_ERROR_TYPES)
_S2_BUCKETS: list[str] = list(PAPER_STATE_ERROR_TYPES)
_T3_BUCKETS: list[str] = ["valid"] + list(T3_PAPER_INVALID_REASONS)
_R2_BUCKETS: list[str] = ["oracle_shortest"] + list(R2_PAPER_INVALID_TRAJECTORY_TYPES)

# Minimum cases-per-depth-per-task required to guarantee full bucket coverage.
# Tasks without required buckets (t1, t2, r1) use 1 as the practical floor.
_MIN_STRICT_TARGET: dict[str, int] = {
    "s1": len(_S1_BUCKETS),   # valid + 5 invalid types  → 6
    "s2": len(_S2_BUCKETS),   # 5 error types            → 5
    "t1": 1,
    "t2": 1,
    "t3": len(_T3_BUCKETS),   # valid + 5 invalid reasons → 6
    "r1": 1,
    "r2": len(_R2_BUCKETS),   # oracle_shortest + 4 types → 5
}
# Human-readable bucket labels for error messages
_BUCKET_LABELS: dict[str, list[str]] = {
    "s1": _S1_BUCKETS,
    "s2": _S2_BUCKETS,
    "t3": _T3_BUCKETS,
    "r2": _R2_BUCKETS,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _task_depth_seed(base: int, task: str, depth: int) -> int:
    return base * 10000 + _TASK_IDX[task] * 1000 + depth


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    return str(obj)


def _smart_sample(
    cases: list[dict[str, Any]],
    target: int,
    coverage_fn: Callable[[dict[str, Any]], str | None],
    required_buckets: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Return up to target cases, guaranteeing ≥1 per required bucket where available."""
    if len(cases) <= target:
        return list(cases)

    pool = list(cases)
    rng.shuffle(pool)

    by_bucket: dict[str, list[int]] = {}
    for i, c in enumerate(pool):
        b = coverage_fn(c)
        if b is not None:
            by_bucket.setdefault(b, []).append(i)

    used: set[int] = set()
    selected: list[dict[str, Any]] = []

    for bucket in required_buckets:
        for idx in by_bucket.get(bucket, []):
            if idx not in used:
                used.add(idx)
                selected.append(pool[idx])
                break

    fill = target - len(selected)
    if fill > 0:
        leftover = [pool[i] for i in range(len(pool)) if i not in used]
        selected.extend(rng.sample(leftover, min(fill, len(leftover))))

    return selected


def _inject_depth(cases: list[dict[str, Any]], depth: int) -> None:
    """Add optimal_depth to meta where absent (e.g. S2 cases, S1 invalid cases)."""
    for c in cases:
        m = c.setdefault("meta", {})
        if "optimal_depth" not in m:
            m["optimal_depth"] = depth


def _base_count(task: str, target: int) -> int:
    """Estimate how many base states/exact-cases to fetch before generating output cases."""
    estimates: dict[str, int] = {
        "s1": max(10, (target // 5 + 1) * 2),
        "s2": max(10, (target // 5 + 1) * 2),
        "t1": target,
        "t2": max(10, (target // 3 + 1) * 2),
        "t3": max(20, (target // 15 + 1) * 2),
        "r1": target,
        "r2": max(10, (target // 4 + 1) * 2),
    }
    return estimates[task]


# ---------------------------------------------------------------------------
# Per-task depth builders
# ---------------------------------------------------------------------------

def _build_s1_at_depth(
    items: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    warnings: list[str] = []
    if not items:
        return [], ["s1: no pool items available at this depth"]

    pool = list(items)
    rng.shuffle(pool)
    base_items = pool[: min(_base_count("s1", target), len(pool))]

    all_cases: list[dict[str, Any]] = []
    for item in base_items:
        try:
            batch = build_s1_cases(
                legal_state_items=[item],
                error_types=PAPER_STATE_ERROR_TYPES,
                max_invalid_per_state=5,
                seed=rng.randint(0, 2**31 - 1),
            )
            all_cases.extend(batch)
        except Exception:  # noqa: BLE001
            pass

    if not all_cases:
        return [], warnings + ["s1: no cases generated at this depth"]

    def _cov(c: dict[str, Any]) -> str | None:
        if c["gold"]["label"] == "valid":
            return "valid"
        return c["meta"].get("error_type")

    result = _smart_sample(all_cases, target, _cov, _S1_BUCKETS, rng)
    if len(result) < target:
        warnings.append(f"s1: {len(result)}/{target} cases available at this depth")
    return result, warnings


def _build_s2_at_depth(
    items: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    warnings: list[str] = []
    if not items:
        return [], ["s2: no pool items available at this depth"]

    pool = list(items)
    rng.shuffle(pool)
    base_items = pool[: min(_base_count("s2", target), len(pool))]

    all_cases: list[dict[str, Any]] = []
    for item in base_items:
        try:
            batch = build_s2_cases(
                source_states=[item["state"]],
                error_types=PAPER_STATE_ERROR_TYPES,
                max_invalid_per_state=5,
                seed=rng.randint(0, 2**31 - 1),
            )
            all_cases.extend(batch)
        except Exception:  # noqa: BLE001
            pass

    if not all_cases:
        return [], warnings + ["s2: no cases generated at this depth"]

    def _cov(c: dict[str, Any]) -> str | None:
        return c["meta"].get("error_type")

    result = _smart_sample(all_cases, target, _cov, _S2_BUCKETS, rng)
    if len(result) < target:
        warnings.append(f"s2: {len(result)}/{target} cases available at this depth")
    return result, warnings


def _build_t1_at_depth(
    items: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    warnings: list[str] = []
    if not items:
        return [], ["t1: no pool items available at this depth"]

    pool = list(items)
    rng.shuffle(pool)
    base_items = pool[: min(target, len(pool))]

    cases = build_t1_cases(base_items)
    if len(cases) < target:
        warnings.append(f"t1: {len(cases)}/{target} cases available at this depth")
    return cases, warnings


def _build_t2_at_depth(
    items: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    warnings: list[str] = []
    if not items:
        return [], ["t2: no pool items available at this depth"]

    pool = list(items)
    rng.shuffle(pool)
    base_items = pool[: min(_base_count("t2", target), len(pool))]

    cases = build_t2_cases(base_items)
    result = cases if len(cases) <= target else rng.sample(cases, target)
    if len(result) < target:
        warnings.append(f"t2: {len(result)}/{target} cases available at this depth")
    return result, warnings


def _build_t3_at_depth(
    items: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    warnings: list[str] = []
    if not items:
        return [], ["t3: no pool items available at this depth"]

    pool = list(items)
    rng.shuffle(pool)
    base_items = pool[: min(_base_count("t3", target), len(pool))]

    try:
        cases = build_t3_cases(
            legal_state_items=base_items,
            include_valid=True,
            invalid_reasons=V1_INVALID_REASONS,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"t3: build_t3_cases raised: {exc!r}")
        return [], warnings

    def _cov(c: dict[str, Any]) -> str | None:
        if c["gold"]["label"] == "valid":
            return "valid"
        return c["gold"].get("reason")

    result = _smart_sample(cases, target, _cov, _T3_BUCKETS, rng)
    if len(result) < target:
        warnings.append(f"t3: {len(result)}/{target} cases available at this depth")
    return result, warnings


def _build_r1_at_depth(
    exact_pool: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    warnings: list[str] = []
    if not exact_pool:
        return [], ["r1: no exact-depth cases available at this depth"]

    pool = list(exact_pool)
    rng.shuffle(pool)
    base = pool[: min(target, len(pool))]

    cases = build_r1_cases(base)
    if len(cases) < target:
        warnings.append(f"r1: {len(cases)}/{target} cases available at this depth")
    return cases, warnings


def _build_r2_at_depth(
    exact_pool: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    warnings: list[str] = []
    if not exact_pool:
        return [], ["r2: no exact-depth cases available at this depth"]

    pool = list(exact_pool)
    rng.shuffle(pool)
    base_count = min(_base_count("r2", target), len(pool))
    base = pool[:base_count]

    try:
        cases = build_r2_cases(
            exact_depth_cases=base,
            include_valid=True,
            invalid_types=R2_PAPER_INVALID_TRAJECTORY_TYPES,
            seed=rng.randint(0, 2**31 - 1),
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"r2: build_r2_cases raised: {exc!r}")
        return [], warnings

    def _cov(c: dict[str, Any]) -> str | None:
        return c["meta"].get("trajectory_type")

    result = _smart_sample(cases, target, _cov, _R2_BUCKETS, rng)
    if len(result) < target:
        warnings.append(f"r2: {len(result)}/{target} cases available at this depth")
    return result, warnings


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------

def _build_coverage_summary(
    by_task: dict[str, list[dict[str, Any]]],
    depth_counts: dict[str, dict[int, int]],
    warnings: list[str],
    cfg: "_DepthBenchConfig",
    goal_seed_count: int,
) -> dict[str, Any]:
    total = sum(len(v) for v in by_task.values())

    s1_label: dict[str, int] = defaultdict(int)
    s1_itype: dict[str, int] = defaultdict(int)
    for c in by_task["s1"]:
        s1_label[c["gold"]["label"]] += 1
        if c["gold"]["label"] == "invalid":
            s1_itype[c["meta"].get("error_type", "unknown")] += 1

    s2_etype: dict[str, int] = defaultdict(int)
    for c in by_task["s2"]:
        s2_etype[c["meta"].get("error_type", "unknown")] += 1

    t3_reason: dict[str, int] = defaultdict(int)
    t3_req: dict[str, int] = defaultdict(int)
    for c in by_task["t3"]:
        if c["gold"]["label"] == "valid":
            t3_reason["valid"] += 1
        else:
            t3_reason[c["gold"].get("reason", "unknown")] += 1
            t3_req[c["meta"].get("requested_invalid_reason", "unknown")] += 1

    r2_ttype: dict[str, int] = defaultdict(int)
    for c in by_task["r2"]:
        r2_ttype[c["meta"].get("trajectory_type", "unknown")] += 1

    return {
        "schema": "klotskibench.depth_benchmark_coverage_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "depth_min": cfg.depth_min,
            "depth_max": cfg.depth_max,
            "cases_per_depth_per_task": cfg.cases_per_depth,
            "seed": cfg.seed,
            "goal_seed_count": goal_seed_count,
            "max_goal_seeds": cfg.max_goal_seeds,
        },
        "total_cases": total,
        "cases_per_task": {t: len(by_task[t]) for t in ALL_TASKS},
        "cases_per_task_per_depth": {
            t: {str(d): n for d, n in depth_counts[t].items()} for t in ALL_TASKS
        },
        "distribution": {
            "s1": {"label": dict(s1_label), "invalid_type": dict(s1_itype)},
            "s2": {"error_type": dict(s2_etype)},
            "t3": {
                "gold_reason": dict(t3_reason),
                "requested_invalid_reason": dict(t3_req),
            },
            "r2": {"trajectory_type": dict(r2_ttype)},
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class _DepthBenchConfig:
    out_dir: Path
    depth_min: int
    depth_max: int
    cases_per_depth: int
    seed: int
    max_goal_seeds: int | None
    no_predecessor_filter: bool
    strict_coverage: bool = True


def _run(cfg: _DepthBenchConfig) -> None:
    if cfg.depth_min < 1 or cfg.depth_max < cfg.depth_min:
        raise SystemExit("require 1 <= depth_min <= depth_max")
    if cfg.cases_per_depth < 1:
        raise SystemExit("cases_per_depth_per_task must be >= 1")

    if cfg.strict_coverage:
        failing = [
            t for t in ALL_TASKS if cfg.cases_per_depth < _MIN_STRICT_TARGET[t]
        ]
        if failing:
            lines = [
                f"Error: --cases-per-depth-per-task={cfg.cases_per_depth} is too small "
                "for strict coverage mode.",
                "Minimum required to guarantee all required buckets are covered:",
            ]
            for t in failing:
                buckets = _BUCKET_LABELS.get(t)
                if buckets:
                    lines.append(
                        f"  {t}: {_MIN_STRICT_TARGET[t]} "
                        f"(buckets: {', '.join(buckets)})"
                    )
                else:
                    lines.append(f"  {t}: {_MIN_STRICT_TARGET[t]}")
            lines.append(
                "Use --allow-partial-coverage to disable this check "
                "and export whatever is available."
            )
            raise SystemExit("\n".join(lines))

    # Step 1: goal seeds
    if cfg.no_predecessor_filter:
        goal_seeds = enumerate_goal_seed_states(
            max_goals=cfg.max_goal_seeds, require_predecessor_space=False
        )
    else:
        goal_seeds = enumerate_goal_seed_states(
            max_goals=cfg.max_goal_seeds, require_predecessor_space=True
        )
    if not goal_seeds:
        print(
            "Warning: no goal seeds from enumerator; using DEFAULT_GOAL_SEED_CANDIDATES.",
            file=sys.stderr,
        )
        goal_seeds = [dict(s) for s in DEFAULT_GOAL_SEED_CANDIDATES]
    print(f"Goal seeds: {len(goal_seeds)}")

    # Step 2: reverse-BFS records
    print(f"Building reverse-BFS records up to depth {cfg.depth_max} ...")
    records = build_reverse_bfs_records(goal_seed_states=goal_seeds, max_depth=cfg.depth_max)
    if not records:
        raise SystemExit("reverse-BFS produced no states; check legacy engine.")
    print(f"Records built: {len(records)} states.")

    # Step 3: legal state pool (for S/T tasks)
    pool = build_legal_state_pool(records=records)

    # Step 4: per-depth per-task generation
    all_by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in ALL_TASKS}
    depth_counts: dict[str, dict[int, int]] = {t: {} for t in ALL_TASKS}
    all_warnings: list[str] = []

    target = cfg.cases_per_depth
    # How many exact-depth cases to fetch per depth (serves both R1 and R2)
    exact_need = max(target, _base_count("r2", target))

    for depth in range(cfg.depth_min, cfg.depth_max + 1):
        print(f"  depth {depth} ...", end="  ", flush=True)

        pool_items = pool.items_at_depth(depth)

        # Shared exact-case pool for R1/R2 at this depth
        base_fetch_seed = cfg.seed * 1000 + depth
        exact_pool = build_exact_depth_cases(
            records=records,
            depth=depth,
            num_samples=exact_need,
            seed=base_fetch_seed,
        )

        depth_task_counts: list[str] = []
        for task in ALL_TASKS:
            task_seed = _task_depth_seed(cfg.seed, task, depth)

            if task == "s1":
                cases, warns = _build_s1_at_depth(pool_items, target, task_seed)
            elif task == "s2":
                cases, warns = _build_s2_at_depth(pool_items, target, task_seed)
            elif task == "t1":
                cases, warns = _build_t1_at_depth(pool_items, target, task_seed)
            elif task == "t2":
                cases, warns = _build_t2_at_depth(pool_items, target, task_seed)
            elif task == "t3":
                cases, warns = _build_t3_at_depth(pool_items, target, task_seed)
            elif task == "r1":
                cases, warns = _build_r1_at_depth(exact_pool, target, task_seed)
            else:
                cases, warns = _build_r2_at_depth(exact_pool, target, task_seed)

            _inject_depth(cases, depth)
            all_by_task[task].extend(cases)
            depth_counts[task][depth] = len(cases)
            depth_task_counts.append(f"{task}={len(cases)}")
            for w in warns:
                all_warnings.append(f"depth {depth} {w}")

        print(" ".join(depth_task_counts))

    # Step 5: write output files
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print()
    for task in ALL_TASKS:
        task_dir = cfg.out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / "cases.json"
        path.write_text(
            json.dumps(_jsonable(all_by_task[task]), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  {task}: {len(all_by_task[task])} cases -> {path}")

    # Step 6: coverage summary
    summary = _build_coverage_summary(
        all_by_task, depth_counts, all_warnings, cfg, len(goal_seeds)
    )
    summary_path = cfg.out_dir / "coverage_summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nTotal: {summary['total_cases']} cases")
    print(f"Coverage summary -> {summary_path}")
    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  {w}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build a depth-stratified seven-task benchmark "
            "(task-dir structure compatible with run_mini_ollama_all_tasks.py)."
        )
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--depth-min", type=int, default=1)
    p.add_argument("--depth-max", type=int, default=10)
    p.add_argument(
        "--cases-per-depth-per-task",
        type=int,
        default=100,
        help="Target final exported cases per task per depth (default: 100).",
    )
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument(
        "--max-goal-seeds",
        type=int,
        default=2_000,
        help="Cap on goal seeds used for reverse-BFS (0 = no cap).",
    )
    p.add_argument(
        "--no-predecessor-filter",
        action="store_true",
        help="Use all goal seeds without predecessor-space filter.",
    )
    p.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help=(
            "Disable the strict coverage check.  By default the script fails fast "
            "if --cases-per-depth-per-task is smaller than the number of required "
            "buckets for any task (S1/T3 need ≥6, S2/R2 need ≥5).  Pass this flag "
            "to export whatever is available instead."
        ),
    )
    args = p.parse_args()
    mgs: int | None = None if args.max_goal_seeds == 0 else args.max_goal_seeds
    _run(
        _DepthBenchConfig(
            out_dir=args.out_dir,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            cases_per_depth=args.cases_per_depth_per_task,
            seed=args.seed,
            max_goal_seeds=mgs,
            no_predecessor_filter=bool(args.no_predecessor_filter),
            strict_coverage=not args.allow_partial_coverage,
        )
    )


if __name__ == "__main__":
    main()
