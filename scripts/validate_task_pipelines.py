#!/usr/bin/env python3
"""
Local pipeline validation (builders + metrics + evaluate_benchmark) for all
seven task tracks. Run from repository root:

  python scripts/validate_task_pipelines.py

Requires the in-repo legacy engine (legacy/ on sys.path via _bootstrap_paths).
This script does not load pytest conftest; it exercises real adapters/legacy.
"""
from __future__ import annotations

import math
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import _bootstrap_paths  # noqa: E402

_bootstrap_paths.ensure_repo_sys_path()

from src.evaluate import evaluate_benchmark  # noqa: E402
from src.generators.goal_seed_enumerator import (  # noqa: E402
    enumerate_goal_seed_states,
)
from src.generators.legal_state_pool import build_legal_state_pool  # noqa: E402
from src.generators.reverse_bfs_dataset_builder import (  # noqa: E402
    build_exact_depth_cases,
    build_reverse_bfs_records,
)
from src.metrics import (  # noqa: E402
    r1_metrics,
    r2_metrics,
    s1_metrics,
    s2_metrics,
    t1_metrics,
    t2_metrics,
    t3_metrics,
)
from src.tasks.r1_builder import build_r1_cases  # noqa: E402
from src.tasks.r2_builder import build_r2_cases  # noqa: E402
from src.generators.invalid_state_generator import PAPER_STATE_ERROR_TYPES  # noqa: E402
from src.tasks.s1_builder import build_s1_validation_sample  # noqa: E402
from src.tasks.s2_builder import build_s2_coverage_cases  # noqa: E402
from src.tasks.t1_builder import build_t1_cases  # noqa: E402
from src.tasks.t2_builder import build_t2_cases  # noqa: E402
from src.tasks.t3_builder import build_t3_cases, V1_INVALID_REASONS  # noqa: E402


# --- tiny deterministic corpus: reverse BFS + exact-depth cases ----------------


def _build_tiny_corpus() -> tuple[Any, list[dict[str, Any]]]:
    goal_seeds = enumerate_goal_seed_states(
        max_goals=2_000, require_predecessor_space=True
    )
    if not goal_seeds:
        raise RuntimeError("No goal seeds; check legacy engine / enumerator.")
    records = build_reverse_bfs_records(goal_seed_states=goal_seeds, max_depth=4)
    if not records:
        raise RuntimeError("No reverse-BFS records.")
    all_exact: list[dict] = []
    for depth, seed, n in ((1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1), (3, 1, 1)):
        all_exact.extend(
            build_exact_depth_cases(records=records, depth=depth, num_samples=n, seed=1000 * seed + depth)
        )
    if len(all_exact) < 3:
        raise RuntimeError("Exact-depth sampler returned too few cases; raise max_depth?")
    exact_cases = all_exact[:5]
    return records, exact_cases


def _legal_item_dicts(
    records: dict[Any, Any], exact_cases: list[dict], max_items: int
) -> list[dict[str, Any]]:
    pool = build_legal_state_pool(records=records)
    out: list[dict[str, Any]] = []
    for d in sorted(pool.available_depths())[:3]:
        out.extend(pool.items_at_depth(d))
        if len(out) >= max_items:
            return out[:max_items]
    for ec in exact_cases:
        st = ec["initial_state"]
        if isinstance(st, dict) and st:
            out.append(
                {
                    "state": st,
                    "optimal_depth": int(ec.get("optimal_depth", 0)),
                    "canonical_state_key": ec.get("meta", {}).get("canonical_state_key"),
                }
            )
        if len(out) >= max_items:
            break
    if not out:
        raise RuntimeError("Could not obtain legal state items for S/T tasks.")
    return out[:max_items]


@dataclass
class Row:
    task: str
    n_cases: int
    build_ok: bool
    oracle_ok: bool
    wrong_degrades: bool
    e2e_ok: bool
    details: str = ""


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    error: str | None = None


def _fclose(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def _check_s1(
    items: list[dict[str, Any]], _exact_cases: list, report: Report
) -> None:
    cases = build_s1_validation_sample(items, seed=42)
    if not cases or any(c.get("task") != "s1" for c in cases):
        report.rows.append(Row("s1", 0, False, False, False, False, "no cases or bad task"))
        return
    labels = {c["gold"]["label"] for c in cases}
    invalid_types = {
        c["meta"]["error_type"]
        for c in cases
        if c["gold"]["label"] == "invalid" and "error_type" in c.get("meta", {})
    }
    build_ok = "valid" in labels and invalid_types >= set(PAPER_STATE_ERROR_TYPES)
    o = [{"label": c["gold"]["label"]} for c in cases]
    w = []
    for c in cases:
        w.append({"label": "invalid" if c["gold"]["label"] == "valid" else "valid"})
    mo = s1_metrics.compute_s1_metrics(cases, o)
    mw = s1_metrics.compute_s1_metrics(cases, w)
    oracle_ok = _fclose(mo["accuracy"], 1.0)
    wrong_degrades = mo["accuracy"] > mw["accuracy"] and not _fclose(mw["accuracy"], mo["accuracy"])
    e2e_ok = True
    try:
        evaluate_benchmark(cases, o)
    except Exception as e:  # noqa: BLE001
        e2e_ok = False
        report.error = f"s1 e2e: {e!r}" if not report.error else report.error
    d = (
        f"acc_oracle={mo['accuracy']:.2f} acc_wrong={mw['accuracy']:.2f} "
        f"invalid_types={len(invalid_types)}/{len(PAPER_STATE_ERROR_TYPES)}"
    )
    if not (oracle_ok and wrong_degrades):
        d += " [CHECK: oracle=1, wrong<oracle]"
    report.rows.append(
        Row("s1", len(cases), build_ok, oracle_ok, wrong_degrades, e2e_ok, d)
    )


def _check_s2(
    source_states: list[dict[str, Any]], report: Report
) -> None:
    cases = build_s2_coverage_cases(source_states=source_states, seed=42)
    if not cases:
        report.rows.append(Row("s2", 0, False, False, False, False, "no s2 cases"))
        return
    covered_types = {c["meta"]["error_type"] for c in cases}
    build_ok = covered_types >= set(PAPER_STATE_ERROR_TYPES)
    o = [{"errors": list(c["gold"]["errors"])} for c in cases]
    w = [{"errors": [{"block_id": "WRONG", "error_type": "boundary"}]} for c in cases]
    mo = s2_metrics.compute_s2_metrics(cases, o)
    mw = s2_metrics.compute_s2_metrics(cases, w)
    oracle_ok = _fclose(mo["exact_error_match"], 1.0)
    wrong_degrades = mo["exact_error_match"] > mw["exact_error_match"] + 1e-6
    e2e_ok = True
    try:
        evaluate_benchmark(cases, o)
    except Exception as e:  # noqa: BLE001
        e2e_ok = False
        report.error = f"s2 e2e: {e!r}" if not report.error else report.error
    report.rows.append(
        Row(
            "s2",
            len(cases),
            build_ok,
            oracle_ok,
            wrong_degrades,
            e2e_ok,
            (
                f"eem_or={mo['exact_error_match']:.2f} eem_wr={mw['exact_error_match']:.2f} "
                f"types={len(covered_types)}/{len(PAPER_STATE_ERROR_TYPES)}"
            ),
        )
    )


def _check_t1(
    items: list[dict[str, Any]], report: Report
) -> None:
    cases = build_t1_cases(items[:3])
    if not cases:
        report.rows.append(Row("t1", 0, False, False, False, False, "no t1 cases"))
        return
    o = [{"legal_actions": [dict(x) for x in c["gold"]["legal_actions"]]} for c in cases]
    w: list[dict[str, Any]] = []
    for c in cases:
        w.append({"legal_actions": []} if c["gold"]["legal_actions"] else {"legal_actions": c["gold"]["legal_actions"]})
    if all(len(c["gold"]["legal_actions"]) == 0 for c in cases):
        w[0] = o[0]  # avoid degenerate equal
    mo = t1_metrics.compute_t1_metrics(cases, o)
    mw = t1_metrics.compute_t1_metrics(cases, w)
    if all(len(c["gold"]["legal_actions"]) == 0 for c in cases) and w == o:
        wrong_degrades = True  # degenerate: both perfect
    else:
        wrong_degrades = not _fclose(mo["exact_set_match"], mw["exact_set_match"])
    build_ok = True
    oracle_ok = _fclose(mo["exact_set_match"], 1.0)
    e2e_ok = True
    try:
        evaluate_benchmark(cases, o)
    except Exception as e:  # noqa: BLE001
        e2e_ok = False
    report.rows.append(
        Row(
            "t1",
            len(cases),
            build_ok,
            oracle_ok,
            wrong_degrades,
            e2e_ok,
            f"esm_or={mo['exact_set_match']:.2f} esm_wr={mw['exact_set_match']:.2f}",
        )
    )


def _check_t2(
    items: list[dict[str, Any]], report: Report
) -> None:
    cases = build_t2_cases(items[:1])[:4]
    if not cases:
        report.rows.append(Row("t2", 0, False, False, False, False, "no t2 cases"))
        return
    o = []
    w = []
    for c in cases:
        o.append({"next_state": c["gold"]["next_state"]})
        bad = deepcopy(c["gold"]["next_state"])
        if "blocks" in bad and bad["blocks"]:
            b0 = next(iter(bad["blocks"].values()))
            b0["pos"][0] = 1 - b0["pos"][0]
        w.append({"next_state": bad})
    mo = t2_metrics.compute_t2_metrics(cases, o)
    mw = t2_metrics.compute_t2_metrics(cases, w)
    build_ok = True
    oracle_ok = _fclose(mo["next_state_exact_match"], 1.0)
    wrong_degrades = mw["next_state_exact_match"] < mo["next_state_exact_match"] - 1e-6
    e2e_ok = True
    try:
        evaluate_benchmark(cases, o)
    except Exception:  # noqa: BLE001
        e2e_ok = False
    report.rows.append(
        Row(
            "t2",
            len(cases),
            build_ok,
            oracle_ok,
            wrong_degrades,
            e2e_ok,
            f"nem_or={mo['next_state_exact_match']:.2f} nem_wr={mw['next_state_exact_match']:.2f}",
        )
    )


def _check_t3(
    items: list[dict[str, Any]], report: Report
) -> None:
    reasons: tuple[str, ...] = tuple(
        r for r in (V1_INVALID_REASONS) if r in V1_INVALID_REASONS
    )[:2]
    cases = build_t3_cases(items[:1], include_valid=True, invalid_reasons=reasons)[:5]
    if not cases:
        report.rows.append(Row("t3", 0, False, False, False, False, "no t3 cases"))
        return
    o = []
    w = []
    for c in cases:
        g = c["gold"]
        p = {"label": g["label"]}
        if g.get("label") == "invalid" and "reason" in g:
            p["reason"] = g["reason"]
        o.append(p)
        w.append(
            {
                "label": "valid" if g.get("label") == "invalid" else "invalid",
                "reason": g.get("reason", "wrong"),
            }
        )
    mo = t3_metrics.compute_t3_metrics(cases, o)
    mw = t3_metrics.compute_t3_metrics(cases, w)
    build_ok = True
    oracle_ok = _fclose(mo["label_accuracy"], 1.0) and _fclose(
        mo["joint_verification_reason_accuracy"], 1.0
    )
    wrong_degrades = (
        mo["label_accuracy"] > mw["label_accuracy"] + 1e-6
        or mo["joint_verification_reason_accuracy"] > mw["joint_verification_reason_accuracy"] + 1e-6
    )
    e2e_ok = True
    try:
        evaluate_benchmark(cases, o)
    except Exception:  # noqa: BLE001
        e2e_ok = False
    report.rows.append(
        Row(
            "t3",
            len(cases),
            build_ok,
            oracle_ok,
            wrong_degrades,
            e2e_ok,
            f"la_or={mo['label_accuracy']:.2f} la_wr={mw['label_accuracy']:.2f}",
        )
    )


def _check_r1(
    exact_cases: list[dict[str, Any]], report: Report
) -> None:
    cases = build_r1_cases(exact_cases[:4])
    if not cases or any(c.get("task") != "r1" for c in cases):
        report.rows.append(Row("r1", 0, False, False, False, False, "r1 build failed"))
        return
    o: list[dict] = [
        {
            "predicted_trajectory": [dict(m) for m in c["meta"]["shortest_solution_actions"]]
        }
        for c in cases
    ]
    w: list[dict] = [{"predicted_trajectory": []} for c in cases]
    mo = r1_metrics.compute_r1_metrics(cases, o)
    mw = r1_metrics.compute_r1_metrics(cases, w)
    build_ok = True
    sr_o: dict[str | int, float] = mo.get("sr_at_n", {})  # type: ignore[assignment]
    sr_w: dict[str | int, float] = mw.get("sr_at_n", {})  # type: ignore[assignment]
    depths = {int(c["gold"]["optimal_depth"]) for c in cases}
    oracle_ok = all(float(sr_o.get(d) or sr_o.get(str(d)) or 0.0) >= 0.99 for d in depths)
    sum_o = sum(float(x) for x in sr_o.values() if x is not None)
    sum_w = sum(float(x) for x in sr_w.values() if x is not None)
    wrong_degrades = sum_w < sum_o - 1e-6
    e2e_ok = True
    try:
        evaluate_benchmark(cases, o)
    except Exception:  # noqa: BLE001
        e2e_ok = False
    report.rows.append(
        Row(
            "r1",
            len(cases),
            build_ok,
            oracle_ok,
            wrong_degrades,
            e2e_ok,
            f"sr@n_or eff={mo.get('efficiency', 0):.2f} eff_wr={mw.get('efficiency', 0):.2f}",
        )
    )


def _check_r2(
    exact_cases: list[dict[str, Any]], report: Report
) -> None:
    if not exact_cases:
        report.rows.append(Row("r2", 0, False, False, False, False, "no exact cases for r2"))
        return
    try:
        cases = build_r2_cases(
            exact_cases[:3],
            include_valid=True,
            invalid_types=("wrong_action", "truncated_suffix"),
            seed=0,
        )[:5]
    except Exception as e:  # noqa: BLE001
        report.rows.append(Row("r2", 0, False, False, False, False, f"build: {e!r}"))
        return
    if not cases:
        report.rows.append(Row("r2", 0, False, False, False, False, "r2: empty case list"))
        return
    o, w = [], []
    for c in cases:
        g = c["gold"]
        if g["label"] == "valid":
            o.append({"label": "valid"})
            w.append({"label": "invalid"})
        else:
            p = {
                "label": g["label"],
                "first_error_step": g.get("first_error_step"),
                "step_validity_sequence": list(g.get("step_validity_sequence", [])),
            }
            o.append(p)
            w.append({"label": "valid"})
    mo = r2_metrics.compute_r2_metrics(cases, o)
    mw = r2_metrics.compute_r2_metrics(cases, w)
    build_ok = True
    oracle_ok = _fclose(mo["trajectory_verification_accuracy"], 1.0)
    wrong_degrades = mo["trajectory_verification_accuracy"] > mw["trajectory_verification_accuracy"] + 1e-6
    e2e_ok = True
    try:
        evaluate_benchmark(cases, o)
    except Exception:  # noqa: BLE001
        e2e_ok = False
    report.rows.append(
        Row(
            "r2",
            len(cases),
            build_ok,
            oracle_ok,
            wrong_degrades,
            e2e_ok,
            f"lab_or={mo['trajectory_verification_accuracy']:.2f} lab_wr={mw['trajectory_verification_accuracy']:.2f}",
        )
    )


def run_validation() -> Report:
    rep = Report()
    try:
        records, exact_cases = _build_tiny_corpus()
    except Exception as e:  # noqa: BLE001
        rep.error = str(e)
        return rep
    items = _legal_item_dicts(records, exact_cases, 5)
    source_states: list[dict] = [it["state"] for it in items[:2]]

    order: list[tuple[str, Callable[[], None]]] = [
        ("s1", lambda: _check_s1(items, exact_cases, rep)),
        ("s2", lambda: _check_s2(source_states, rep)),
        ("t1", lambda: _check_t1(items, rep)),
        ("t2", lambda: _check_t2(items, rep)),
        ("t3", lambda: _check_t3(items, rep)),
        ("r1", lambda: _check_r1(exact_cases, rep)),
        ("r2", lambda: _check_r2(exact_cases, rep)),
    ]
    for _name, fn in order:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            rep.rows.append(Row(_name, 0, False, False, False, False, f"EXC: {e!r}"))
    return rep


def _print_table(rep: Report) -> None:
    w = 7
    head = f"| {'task':^{w}} | build | oracle | wrong↓ |  e2e  | n | details |"
    _order = ["s1", "s2", "t1", "t2", "t3", "r1", "r2"]
    print("## Task pipeline validation (local)")
    if rep.error:
        print("**FATAL (corpus or global):**", rep.error)
    print()
    print(head)
    print(f"| {'-'*w} |-------|--------|--------|--------|---|---------|")
    for r in sorted(
        rep.rows,
        key=lambda x: _order.index(x.task) if x.task in _order else 99,
    ):
        b = "PASS" if r.build_ok else "FAIL"
        o = "PASS" if r.oracle_ok else "FAIL"
        wd = "PASS" if r.wrong_degrades else "FAIL"
        e2 = "PASS" if r.e2e_ok else "FAIL"
        print(
            f"| {r.task:^{w}} |  {b:^5} |  {o:^5} |  {wd:^5} |  {e2:^5} | {r.n_cases} | {r.details} |"
        )
    all_pass = not rep.error and all(
        r.build_ok and r.oracle_ok and r.wrong_degrades and r.e2e_ok for r in rep.rows
    )
    print()
    if all_pass and len(rep.rows) == 7:
        print("**Overall: all seven tasks PASS (oracle perfect, wrong strictly worse, e2e no raise).**")
    else:
        print(
            "**Overall: at least one task did not meet checks — see FAIL cells (may indicate fragile generator/coverage or a real bug).**"
        )


def main() -> int:
    rep = run_validation()
    _print_table(rep)
    ok = (
        not rep.error
        and len(rep.rows) == 7
        and all(r.build_ok and r.oracle_ok and r.wrong_degrades and r.e2e_ok for r in rep.rows)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
