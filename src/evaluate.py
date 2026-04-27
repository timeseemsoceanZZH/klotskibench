"""Benchmark evaluator: cases + predictions -> unified results by task."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from src.metrics import r1_metrics, r2_metrics, s1_metrics, s2_metrics, t1_metrics, t2_metrics, t3_metrics

_TASKS_WITH_METRICS: frozenset[str] = frozenset({"s1", "s2", "t1", "t2", "t3", "r1", "r2"})


def evaluate_benchmark(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate aligned (case, prediction) rows and dispatch to metric modules by ``task``.

    Returns a single unified result object. All seven task tracks
    (s1, s2, t1, t2, t3, r1, r2) have metrics modules; unknown ``task``
    values get ``metrics=None`` and ``reason='no_metrics_module_for_task'``.
    """
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")

    by_task: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for case, pred in zip(cases, predictions):
        task = case.get("task", "unknown")
        by_task[str(task)].append((case, pred))

    results_by_task: dict[str, Any] = {}
    for task, pairs in sorted(by_task.items(), key=lambda x: x[0]):
        c_list = [p[0] for p in pairs]
        p_list = [p[1] for p in pairs]
        if task not in _TASKS_WITH_METRICS:
            results_by_task[task] = {
                "n": len(pairs),
                "metrics": None,
                "reason": "no_metrics_module_for_task",
            }
            continue
        if task == "s1":
            m = s1_metrics.compute_s1_metrics(c_list, p_list)
        elif task == "s2":
            m = s2_metrics.compute_s2_metrics(c_list, p_list)
        elif task == "t1":
            m = t1_metrics.compute_t1_metrics(c_list, p_list)
        elif task == "t2":
            m = t2_metrics.compute_t2_metrics(c_list, p_list)
        elif task == "t3":
            m = t3_metrics.compute_t3_metrics(c_list, p_list)
        elif task == "r1":
            m = r1_metrics.compute_r1_metrics(c_list, p_list)
        elif task == "r2":
            m = r2_metrics.compute_r2_metrics(c_list, p_list)
        else:
            m = None
        results_by_task[task] = {"n": len(pairs), "metrics": m}

    return {
        "version": 1,
        "summary": {
            "total_cases": len(cases),
            "by_task": {t: len(v) for t, v in by_task.items()},
        },
        "results_by_task": results_by_task,
    }


TASKS_WITH_METRICS: frozenset[str] = _TASKS_WITH_METRICS

__all__ = [
    "evaluate_benchmark",
    "TASKS_WITH_METRICS",
]
