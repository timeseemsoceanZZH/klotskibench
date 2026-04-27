"""Structured report tables (dict rows) for integration and pretty-printing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from src.metrics import r1_metrics, r2_metrics

GroupKey = Literal["optimal_depth", "task", "trajectory_type"]


def build_main_task_table(
    evaluation_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    One row per task: ``n``, status, full ``metrics`` payload (or reason if skipped).
    """
    rows: list[dict[str, Any]] = []
    for task, block in sorted(evaluation_result.get("results_by_task", {}).items()):
        row: dict[str, Any] = {
            "task": task,
            "n": block.get("n", 0),
        }
        metrics = block.get("metrics")
        if metrics is None:
            row["status"] = block.get("reason", "no_metrics")
            row["metrics"] = None
        else:
            row["status"] = "ok"
            row["metrics"] = metrics
        rows.append(row)
    return rows


def build_conditioned_metrics_table(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    group_by: GroupKey = "optimal_depth",
) -> list[dict[str, Any]]:
    """
    Re-run evaluation within each stratum of ``group_by`` (from case meta / gold).

    - ``optimal_depth``: from ``case["gold"].get("optimal_depth")`` or
      ``case["meta"].get("optimal_depth")`` (R1, R2, T1 pool items).
    - ``task``: group by ``case["task"]`` (per-task full metric dict on one row).
    - ``trajectory_type``: from ``case["meta"].get("trajectory_type")`` (R2).
    """
    from src.evaluate import evaluate_benchmark
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must be the same length")

    buckets: dict[str | int | None, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    for case, pred in zip(cases, predictions):
        key: Any
        if group_by == "optimal_depth":
            g = case.get("gold", {})
            key = g.get("optimal_depth", case.get("meta", {}).get("optimal_depth"))
        elif group_by == "task":
            key = case.get("task")
        elif group_by == "trajectory_type":
            key = case.get("meta", {}).get("trajectory_type")
        else:
            key = None
        buckets[key].append((case, pred))

    rows: list[dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda x: (x is None, str(x))):
        pair_list = buckets[key]
        c_sub = [p[0] for p in pair_list]
        p_sub = [p[1] for p in pair_list]
        ev = evaluate_benchmark(c_sub, p_sub)
        for task, block in ev["results_by_task"].items():
            m = block.get("metrics")
            if m is None:
                continue
            row: dict[str, Any] = {
                str(group_by): key,
                "task": task,
                "n": block.get("n", 0),
            }
            if isinstance(m, dict):
                row.update(m)
            rows.append(row)
    return rows


def build_oracle_state_table(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Per-case oracle metadata for reproducibility (no predictions).

    Pulls ``canonical_state_key`` and ``optimal_depth`` when present.
    """
    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        meta = case.get("meta", {})
        gold = case.get("gold", {})
        row = {
            "index": i,
            "task": case.get("task"),
            "case_id": case.get("case_id") or f"row_{i}",
            "canonical_state_key": meta.get("canonical_state_key"),
            "optimal_depth": gold.get("optimal_depth", meta.get("optimal_depth")),
            "trajectory_type": meta.get("trajectory_type"),
        }
        rows.append(row)
    return rows


def build_depth_wise_trajectory_report(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    trajectory_tasks: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Depth-stratified metrics for trajectory tracks (R1, R2 by default).

    For each depth, runs the appropriate metric module on the slice of cases.
    """
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must be the same length")
    if trajectory_tasks is None:
        trajectory_tasks = frozenset({"r1", "r2"})

    depth_buckets: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for case, pred in zip(cases, predictions):
        task = case.get("task")
        if task not in trajectory_tasks:
            continue
        d = case.get("gold", {}).get("optimal_depth", case.get("meta", {}).get("optimal_depth"))
        if d is None:
            continue
        depth_buckets[int(d)].append((case, pred))

    rows: list[dict[str, Any]] = []
    for depth in sorted(depth_buckets.keys()):
        pair_list = depth_buckets[depth]
        c_sub = [p[0] for p in pair_list]
        p_sub = [p[1] for p in pair_list]
        by_t: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for c, p in pair_list:
            t = c.get("task", "unknown")
            by_t[t].append((c, p))
        for task, plist in sorted(by_t.items()):
            c_t = [x[0] for x in plist]
            p_t = [x[1] for x in plist]
            if task == "r1":
                m = r1_metrics.compute_r1_metrics(c_t, p_t)
            elif task == "r2":
                m = r2_metrics.compute_r2_metrics(c_t, p_t)
            else:
                continue
            row: dict[str, Any] = {"optimal_depth": depth, "task": task, "n": len(plist)}
            if isinstance(m, dict):
                row.update(m)
            rows.append(row)
    return rows


__all__ = [
    "build_main_task_table",
    "build_conditioned_metrics_table",
    "build_oracle_state_table",
    "build_depth_wise_trajectory_report",
]
