#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path("scripts").resolve()))
import validate_task_pipelines as v  # noqa: E402

from src.generators.invalid_state_generator import PAPER_STATE_ERROR_TYPES  # noqa: E402
from src.tasks.s1_builder import build_s1_validation_sample  # noqa: E402
from src.tasks.s2_builder import build_s2_coverage_cases  # noqa: E402
from src.tasks.t1_builder import build_t1_cases  # noqa: E402
from src.tasks.t2_builder import build_t2_cases  # noqa: E402
from src.tasks.t3_builder import build_t3_cases, V1_INVALID_REASONS  # noqa: E402
from src.tasks.r1_builder import build_r1_cases  # noqa: E402
from src.tasks.r2_builder import build_r2_coverage_cases  # noqa: E402

OUT_ROOT = Path("output/seven_task_validation")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(val) for k, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2, ensure_ascii=False)


def build_task_outputs() -> dict[str, dict[str, Any]]:
    records, exact_cases = v._build_tiny_corpus()
    items = v._legal_item_dicts(records, exact_cases, 5)
    source_states = [it["state"] for it in items[:2]]

    outputs: dict[str, dict[str, Any]] = {}

    # S1: State Validity Judgment
    s1_cases = build_s1_validation_sample(items, seed=42)
    s1_oracle = [{"label": c["gold"]["label"]} for c in s1_cases]
    s1_wrong = [
        {"label": "invalid" if c["gold"]["label"] == "valid" else "valid"}
        for c in s1_cases
    ]
    outputs["s1"] = {
        "task_name": "S1 State Validity Judgment",
        "headline_metric": "accuracy",
        "coverage": {"required_invalid_types": list(PAPER_STATE_ERROR_TYPES)},
        "cases": s1_cases,
        "oracle_predictions": s1_oracle,
        "wrong_predictions": s1_wrong,
    }

    # S2: State Error Localization
    s2_cases = build_s2_coverage_cases(source_states=source_states, seed=42)
    s2_oracle = [{"errors": list(c["gold"]["errors"])} for c in s2_cases]
    s2_wrong = [
        {"errors": [{"block_id": "WRONG", "error_type": "boundary"}]}
        for _ in s2_cases
    ]
    outputs["s2"] = {
        "task_name": "S2 State Error Localization",
        "headline_metric": "exact_error_match",
        "coverage": {"required_error_types": list(PAPER_STATE_ERROR_TYPES)},
        "cases": s2_cases,
        "oracle_predictions": s2_oracle,
        "wrong_predictions": s2_wrong,
    }

    # T1: Legal Action Prediction
    t1_cases = build_t1_cases(items[:3])
    t1_oracle = [
        {"legal_actions": [dict(x) for x in c["gold"]["legal_actions"]]}
        for c in t1_cases
    ]
    t1_wrong = []
    for c in t1_cases:
        if c["gold"]["legal_actions"]:
            t1_wrong.append({"legal_actions": []})
        else:
            t1_wrong.append({"legal_actions": c["gold"]["legal_actions"]})
    outputs["t1"] = {
        "task_name": "T1 Legal Action Prediction",
        "headline_metric": "exact_set_match",
        "cases": t1_cases,
        "oracle_predictions": t1_oracle,
        "wrong_predictions": t1_wrong,
    }

    # T2: Next-State Prediction
    t2_cases = build_t2_cases(items[:1])[:4]
    t2_oracle = []
    t2_wrong = []
    for c in t2_cases:
        t2_oracle.append({"next_state": c["gold"]["next_state"]})
        bad = deepcopy(c["gold"]["next_state"])
        if "blocks" in bad and bad["blocks"]:
            b0 = next(iter(bad["blocks"].values()))
            b0["pos"][0] = 1 - b0["pos"][0]
        t2_wrong.append({"next_state": bad})
    outputs["t2"] = {
        "task_name": "T2 Next-State Prediction",
        "headline_metric": "next_state_exact_match",
        "cases": t2_cases,
        "oracle_predictions": t2_oracle,
        "wrong_predictions": t2_wrong,
    }

    # T3: One-Step Transition Verification
    t3_cases = build_t3_cases(
        items[:1],
        include_valid=True,
        invalid_reasons=V1_INVALID_REASONS,
    )
    t3_oracle = []
    t3_wrong = []
    for c in t3_cases:
        g = c["gold"]
        p = {"label": g["label"]}
        if g.get("label") == "invalid" and "reason" in g:
            p["reason"] = g["reason"]
        t3_oracle.append(p)
        t3_wrong.append({
            "label": "valid" if g.get("label") == "invalid" else "invalid",
            "reason": g.get("reason", "wrong"),
        })
    outputs["t3"] = {
        "task_name": "T3 One-Step Transition Verification",
        "headline_metric": "verification_accuracy",
        "auxiliary_metric": "joint_verification_reason_accuracy",
        "coverage": {"required_invalid_reasons": list(V1_INVALID_REASONS)},
        "cases": t3_cases,
        "oracle_predictions": t3_oracle,
        "wrong_predictions": t3_wrong,
    }

    # R1: Trajectory Generation
    r1_cases = build_r1_cases(exact_cases[:4])
    r1_oracle = [
        {
            "predicted_trajectory": [
                dict(m) for m in c["meta"]["shortest_solution_actions"]
            ]
        }
        for c in r1_cases
    ]
    r1_wrong = [{"predicted_trajectory": []} for _ in r1_cases]
    outputs["r1"] = {
        "task_name": "R1 Trajectory Generation",
        "headline_metric": "sr_at_n",
        "auxiliary_metrics": ["alive_t", "tcr_t", "efficiency"],
        "cases": r1_cases,
        "oracle_predictions": r1_oracle,
        "wrong_predictions": r1_wrong,
    }

    # R2: Trajectory Verification
    r2_cases = build_r2_coverage_cases(exact_cases[:4], seed=0)
    r2_oracle = []
    r2_wrong = []
    for c in r2_cases:
        g = c["gold"]
        if g["label"] == "valid":
            r2_oracle.append({"label": "valid"})
            r2_wrong.append({"label": "invalid"})
        else:
            r2_oracle.append({
                "label": g["label"],
                "first_error_step": g.get("first_error_step"),
                "step_validity_sequence": list(g.get("step_validity_sequence", [])),
            })
            r2_wrong.append({"label": "valid"})
    outputs["r2"] = {
        "task_name": "R2 Trajectory Verification",
        "headline_metric": "trajectory_verification_accuracy",
        "coverage": {
            "invalid_types": sorted({
                c.get("meta", {}).get("trajectory_type")
                for c in r2_cases
                if c["gold"].get("label") == "invalid"
            })
        },
        "cases": r2_cases,
        "oracle_predictions": r2_oracle,
        "wrong_predictions": r2_wrong,
    }

    return outputs


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    outputs = build_task_outputs()

    all_cases = []
    all_oracle_predictions = []
    all_wrong_predictions = []
    summary_rows = []

    for task, obj in outputs.items():
        task_dir = OUT_ROOT / task
        cases = obj["cases"]
        oracle = obj["oracle_predictions"]
        wrong = obj["wrong_predictions"]

        oracle_eval = v.evaluate_benchmark(cases, oracle)
        wrong_eval = v.evaluate_benchmark(cases, wrong)

        write_json(task_dir / "cases.json", cases)
        write_json(task_dir / "oracle_predictions.json", oracle)
        write_json(task_dir / "wrong_predictions.json", wrong)
        write_json(task_dir / "oracle_eval_summary.json", oracle_eval)
        write_json(task_dir / "wrong_eval_summary.json", wrong_eval)
        write_json(task_dir / "task_metadata.json", {
            k: val for k, val in obj.items()
            if k not in {"cases", "oracle_predictions", "wrong_predictions"}
        })

        all_cases.extend(cases)
        all_oracle_predictions.extend(oracle)
        all_wrong_predictions.extend(wrong)

        summary_rows.append({
            "task": task,
            "task_name": obj.get("task_name"),
            "n_cases": len(cases),
            "headline_metric": obj.get("headline_metric"),
            "oracle_metrics": oracle_eval["results_by_task"][task]["metrics"],
            "wrong_metrics": wrong_eval["results_by_task"][task]["metrics"],
            "coverage": obj.get("coverage", {}),
        })

    write_json(OUT_ROOT / "all_cases.json", all_cases)
    write_json(OUT_ROOT / "all_oracle_predictions.json", all_oracle_predictions)
    write_json(OUT_ROOT / "all_wrong_predictions.json", all_wrong_predictions)
    write_json(
        OUT_ROOT / "all_oracle_eval_summary.json",
        v.evaluate_benchmark(all_cases, all_oracle_predictions),
    )
    write_json(
        OUT_ROOT / "all_wrong_eval_summary.json",
        v.evaluate_benchmark(all_cases, all_wrong_predictions),
    )
    write_json(OUT_ROOT / "summary_rows.json", summary_rows)

    print(f"Wrote seven-task validation outputs under {OUT_ROOT}")
    print()
    print("| task | n_cases | headline_metric | coverage |")
    print("| ---- | ------- | --------------- | -------- |")
    for row in summary_rows:
        print(
            f"| {row['task']} | {row['n_cases']} | "
            f"{row['headline_metric']} | {row['coverage']} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())