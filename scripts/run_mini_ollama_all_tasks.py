#!/usr/bin/env python3
"""
Run a small Ollama model on the seven-task Klotski-Bench mini validation set.

Example:
  python scripts/run_mini_ollama_all_tasks.py --model qwen3:4b --dry-run
  python scripts/run_mini_ollama_all_tasks.py --model qwen3:4b --out-dir output/mini_all_tasks/run_qwen3_4b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import _bootstrap_paths  # noqa: E402

_bootstrap_paths.ensure_repo_sys_path()

from src.evaluate import evaluate_benchmark  # noqa: E402
from src.reporting import build_main_task_table  # noqa: E402

from mini_bench_run_ollama import ollama_chat  # noqa: E402

ALL_TASKS: tuple[str, ...] = ("s1", "s2", "t1", "t2", "t3", "r1", "r2")
EXPORT_ROOT = Path("output/seven_task_validation")
DEFAULT_OUT_BASE = Path("output/mini_all_tasks")


def model_slug(model: str) -> str:
    return re.sub(r"[^\w.\-]+", "-", model).strip("-") or "model"


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return str(obj)


def extract_json_text(raw: str) -> str | None:
    """Return a JSON object/array substring from model text."""
    text = (raw or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        a = text.find(opener)
        b = text.rfind(closer)
        if 0 <= a < b:
            return text[a : b + 1]
    return None


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a JSON object; return (obj, error_message)."""
    candidate = extract_json_text(raw)
    if candidate is None:
        return None, "no_json_found"
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(obj, dict):
        return None, "root_not_object"
    return obj, None


def _normalize_prediction(task: str, obj: dict[str, Any] | None) -> dict[str, Any]:
    """Map parsed JSON to the benchmark prediction schema for ``task``."""
    if obj is None:
        return _empty_prediction(task)

    if task == "s1":
        label = obj.get("label")
        if label in ("valid", "invalid"):
            return {"label": str(label)}
        return _empty_prediction(task)

    if task == "s2":
        errors = obj.get("errors")
        if isinstance(errors, list):
            cleaned: list[dict[str, Any]] = []
            for item in errors:
                if isinstance(item, dict) and item.get("error_type"):
                    cleaned.append(dict(item))
            return {"errors": cleaned}
        return {"errors": []}

    if task == "t1":
        acts = obj.get("legal_actions") or obj.get("predicted_legal_actions")
        if isinstance(acts, list):
            out: list[dict[str, str]] = []
            for a in acts:
                if isinstance(a, dict) and "block_id" in a and "direction" in a:
                    out.append(
                        {
                            "block_id": str(a["block_id"]),
                            "direction": str(a["direction"]),
                        }
                    )
            return {"legal_actions": out}
        return {"legal_actions": []}

    if task == "t2":
        ns = obj.get("next_state")
        if isinstance(ns, dict):
            return {"next_state": ns}
        return _empty_prediction(task)

    if task == "t3":
        label = obj.get("label")
        if label not in ("valid", "invalid"):
            return _empty_prediction(task)
        pred: dict[str, Any] = {"label": str(label)}
        if label == "invalid" and "reason" in obj:
            pred["reason"] = str(obj["reason"])
        return pred

    if task == "r1":
        traj = (
            obj.get("predicted_trajectory")
            or obj.get("trajectory")
            or obj.get("actions")
            or obj.get("moves")
        )
        if isinstance(traj, list):
            out_moves: list[dict[str, str]] = []
            for m in traj:
                if isinstance(m, dict) and "block_id" in m and "direction" in m:
                    out_moves.append(
                        {
                            "block_id": str(m["block_id"]),
                            "direction": str(m["direction"]),
                        }
                    )
            return {"predicted_trajectory": out_moves}
        return {"predicted_trajectory": []}

    if task == "r2":
        label = obj.get("label")
        if label not in ("valid", "invalid"):
            return _empty_prediction(task)
        pred = {"label": str(label)}
        if label == "invalid":
            if "first_error_step" in obj:
                pred["first_error_step"] = obj["first_error_step"]
            if "step_validity_sequence" in obj:
                pred["step_validity_sequence"] = list(obj["step_validity_sequence"])
        return pred

    return _empty_prediction(task)


T2_SAFE_NEXT_STATE: dict[str, Any] = {"grid_size": [5, 4], "blocks": {}}


def _t2_next_state_is_eval_safe(next_state: Any) -> bool:
    return (
        isinstance(next_state, dict)
        and "grid_size" in next_state
        and "blocks" in next_state
        and isinstance(next_state.get("blocks"), dict)
    )


def sanitize_prediction_for_task(
    task: str,
    prediction: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return a prediction shape safe for the existing evaluator (may count as wrong)."""
    if task == "s1":
        if prediction.get("label") in ("valid", "invalid"):
            return prediction, False
        return {"label": "__invalid_prediction__"}, True

    if task == "s2":
        if isinstance(prediction.get("errors"), list):
            return prediction, False
        return {"errors": []}, True

    if task == "t1":
        if isinstance(prediction.get("legal_actions"), list):
            return prediction, False
        return {"legal_actions": []}, True

    if task == "t2":
        next_state = prediction.get("next_state")
        if _t2_next_state_is_eval_safe(next_state):
            return prediction, False
        return {"next_state": dict(T2_SAFE_NEXT_STATE)}, True

    if task == "t3":
        label = prediction.get("label")
        if label in ("valid", "invalid"):
            return prediction, False
        return {
            "label": "__invalid_prediction__",
            "reason": "__invalid_prediction__",
        }, True

    if task == "r1":
        if isinstance(prediction.get("predicted_trajectory"), list):
            return prediction, False
        return {"predicted_trajectory": []}, True

    if task == "r2":
        label = prediction.get("label")
        if label == "valid":
            return prediction, False
        if label == "invalid":
            has_step = isinstance(prediction.get("step_validity_sequence"), list)
            has_first = "first_error_step" in prediction
            if has_step and has_first:
                return prediction, False
        return {
            "label": "__invalid_prediction__",
            "first_error_step": 0,
            "step_validity_sequence": [],
        }, True

    return prediction, False


def _empty_prediction(task: str) -> dict[str, Any]:
    if task == "s1":
        return {"label": "valid"}
    if task == "s2":
        return {"errors": []}
    if task == "t1":
        return {"legal_actions": []}
    if task == "t2":
        return {"next_state": {}}
    if task == "t3":
        return {"label": "valid"}
    if task == "r1":
        return {"predicted_trajectory": []}
    if task == "r2":
        return {"label": "valid"}
    return {}


def dry_run_raw_response(task: str) -> str:
    """Placeholder model output for pipeline checks (not oracle-quality)."""
    placeholders = {
        "s1": {"label": "valid"},
        "s2": {"errors": [{"error_type": "boundary", "block_id": "PLACEHOLDER"}]},
        "t1": {"legal_actions": []},
        "t2": {"next_state": {"blocks": {}, "grid_size": [5, 4]}},
        "t3": {"label": "valid"},
        "r1": {"predicted_trajectory": []},
        "r2": {"label": "valid"},
    }
    return json.dumps(placeholders[task], ensure_ascii=False)


KLOTSKI_RULES = """\
Klotski rules (apply to all tasks below):
- Board size is 5 rows by 4 columns.
- Coordinates are [row, col], zero-indexed from the top-left.
- Each block has shape [height, width] and pos [row, col] (top-left cell).
- A primitive action moves exactly one block by one cell: up, down, left, or right.
- A move is legal only if the moved block stays within the board and does not overlap any other block.
- Blocks that are not moved must keep the same id, shape, and position.
- A state is valid if all blocks are within bounds, non-overlapping, and preserve their declared identities and shapes.
- Solved-state predicate G: the 2×2 Caocao block reaches top-left position [3, 1].\
"""

TASK_NAMES = {
    "s1": "S1 State Validity Judgment",
    "s2": "S2 State Error Localization",
    "t1": "T1 Legal Action Prediction",
    "t2": "T2 Next-State Prediction",
    "t3": "T3 One-Step Transition Verification",
    "r1": "R1 Trajectory Generation",
    "r2": "R2 Trajectory Verification",
}

OUTPUT_SCHEMAS = {
    "s1": '{"label":"valid"} OR {"label":"invalid"}',
    "s2": '{"errors":[{"error_type":"overlap|boundary|missing_block|identity_mismatch|shape_change", ...}]}',
    "t1": '{"legal_actions":[{"block_id":"<id>","direction":"up|down|left|right"}, ...]}',
    "t2": '{"next_state":{"blocks":{...},"grid_size":[rows,cols]}}',
    "t3": '{"label":"valid"} OR {"label":"invalid","reason":"overlap|boundary|stationary_object_drift|shape_change|wrong_moved_block_position"}',
    "r1": '{"predicted_trajectory":[{"block_id":"<id>","direction":"up|down|left|right"}, ...]}',
    "r2": (
        '{"label":"valid"} OR {"label":"invalid","first_error_step":<int>,'
        '"step_validity_sequence":[0|1,...]}'
    ),
}

TASK_INSTRUCTIONS = {
    "s1": """\
Task: Decide whether the given state is valid.
Check: in-bounds, no overlap, no missing block, no identity mismatch, no shape change.
Output JSON only with label "valid" or "invalid".""",
    "s2": """\
Task: Localize all state errors in the given state.
For S2: Do not output only error_type. You must include localization fields required by the error type.

Required fields:
- overlap:
  {"error_type": "overlap", "block_ids": ["<id1>", "<id2>"], "cells": [[row, col], ...]}
- boundary:
  {"error_type": "boundary", "block_id": "<id>", "cells": [[row, col], ...]}
- missing_block:
  {"error_type": "missing_block", "missing_block_id": "<id>", "block_id": "<id>"}
- shape_change:
  {"error_type": "shape_change", "block_id": "<id>"}
- identity_mismatch:
  {"error_type": "identity_mismatch", "block_id": "<id>"}

If multiple errors are present, output all detected errors in the errors list.
If the state is invalid, errors must not be empty.
Output JSON only with an "errors" list (empty only if the state is fully valid).""",
    "t1": """\
Task: From current_state, list the complete unordered set of legal primitive moves.
Each move is one block moved one cell in one direction.
Output JSON only with every legal action (block_id + direction).""",
    "t2": """\
Task: Apply the given legal action to current_state and predict the complete full next_state.
The next_state must include "grid_size": [5, 4] and all blocks (not only the moved block).
Each block must include shape and pos; unchanged blocks keep the same id, shape, and position.
Output JSON only with the resulting next_state.""",
    "t3": """\
Task: Verify whether candidate_next_state exactly matches the simulator-defined result of applying action to current_state.
If invalid, give one reason: overlap, boundary, stationary_object_drift, shape_change, or wrong_moved_block_position.
Output JSON only with label "valid", or label "invalid" plus reason.""",
    "r1": """\
Task: Given initial_state, generate a complete sequence of primitive actions.
The sequence should solve the puzzle by reaching a state in G (Caocao 2×2 at [3, 1]).
Legal detours are allowed, but the final state after executing the complete sequence must be solved.
Output JSON only:
{"predicted_trajectory":[{"block_id":"<id>","direction":"up|down|left|right"}, ...]}""",
    "r2": """\
Task: Given initial_state and candidate_trajectory, judge whether the complete trajectory is valid.
Valid only if every action is legal, every intermediate state is valid, and the final state satisfies G.
The trajectory does not need to be shortest.

For R2:
- step_validity_sequence is a list of 1/0 values, one value per candidate action.
- 1 means the corresponding step is valid; 0 means the corresponding step is invalid.
- first_error_step is 1-indexed.
- first_error_step must be the index of the first 0 in step_validity_sequence.
- Example: if step_validity_sequence is [1, 0], then first_error_step must be 2.
- Example: if step_validity_sequence is [1, 1, 0], then first_error_step must be 3.
- If all action steps are valid but the final state is not solved, then:
  first_error_step = len(candidate_trajectory) + 1
  step_validity_sequence = [1] * len(candidate_trajectory)
- If the trajectory is valid, output only: {"label": "valid"}
- If invalid, output:
  {"label": "invalid", "first_error_step": <int>, "step_validity_sequence": [...]}

Output JSON only.""",
}

PROMPT_LEAKAGE_TERMS = (
    "gold",
    "optimal_depth",
    "shortest_solution_actions",
)

PROMPT_MODES: frozenset[str] = frozenset({"zero_shot_rules_v3", "one_shot_rules_v3"})


def build_ollama_options(
    temperature: float = 0.0,
    seed: int = 42,
    top_p: float | None = None,
    num_predict: int | None = None,
) -> dict[str, Any]:
    """Build Ollama ``options`` for deterministic decoding."""
    opts: dict[str, Any] = {"temperature": temperature, "seed": seed}
    if top_p is not None:
        opts["top_p"] = top_p
    if num_predict is not None:
        opts["num_predict"] = num_predict
    return opts


def decoding_settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "seed": args.seed,
        "top_p": args.top_p,
        "num_predict": args.num_predict,
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument(
        "--ollama-base",
        default="http://127.0.0.1:11434",
        help="Ollama HTTP base URL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/mini_all_tasks/run_<model_slug>)",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(ALL_TASKS),
        help="Comma-separated task ids (default: all seven)",
    )
    parser.add_argument(
        "--max-cases-per-task",
        type=int,
        default=None,
        help="Limit cases per task",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Ollama HTTP; write placeholder predictions and still evaluate",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between Ollama calls",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild cases via export builder instead of loading all_cases.json",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ollama sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ollama random seed (default: 42)",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Optional Ollama top_p (included in request only when set)",
    )
    parser.add_argument(
        "--num-predict",
        dest="num_predict",
        type=int,
        default=None,
        help="Optional Ollama num_predict (included in request only when set)",
    )
    parser.add_argument(
        "--case-root",
        dest="case_root",
        type=Path,
        default=None,
        help=(
            "Load cases from a depth-benchmark directory produced by "
            "scripts/export_depth_benchmark.py. Expects <case-root>/<task>/cases.json "
            "for each task. Overrides --force-rebuild and the default all_cases.json path."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        dest="prompt_mode",
        default="zero_shot_rules_v3",
        choices=sorted(PROMPT_MODES),
        help="Prompt strategy (default: zero_shot_rules_v3).",
    )
    parser.add_argument(
        "--exemplar-root",
        dest="exemplar_root",
        type=Path,
        default=None,
        help=(
            "Directory containing one-shot exemplars produced by "
            "scripts/export_one_shot_exemplars.py. "
            "Required when --prompt-mode one_shot_rules_v3 is used."
        ),
    )


def parse_runner_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ollama on the seven-task Klotski-Bench mini validation set.",
    )
    configure_parser(parser)
    return parser.parse_args(argv)


def build_prompt(case: dict[str, Any]) -> str:
    task = str(case["task"])
    payload = json.dumps(jsonable(case.get("input", {})), ensure_ascii=False, indent=2)
    return (
        f"You are solving Klotski-Bench task {task.upper()} ({TASK_NAMES[task]}).\n\n"
        f"{KLOTSKI_RULES}\n\n"
        f"{TASK_INSTRUCTIONS[task]}\n\n"
        "Respond with JSON only (no markdown fences, no commentary).\n"
        f"Required output schema: {OUTPUT_SCHEMAS[task]}\n\n"
        f"Input JSON:\n{payload}\n"
    )


def load_exemplars(exemplar_root: Path) -> dict[str, dict[str, Any]]:
    """Load per-task one-shot exemplar files from exemplar_root/<task>/case.json."""
    result: dict[str, dict[str, Any]] = {}
    for task in ALL_TASKS:
        path = exemplar_root / task / "case.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"One-shot exemplar not found: {path}\n"
                f"Run: python scripts/export_one_shot_exemplars.py --out-dir {exemplar_root}"
            )
        result[task] = json.loads(path.read_text(encoding="utf-8"))
    return result


def build_one_shot_prompt(case: dict[str, Any], exemplar: dict[str, Any]) -> str:
    """Build a one-shot prompt by prepending a fixed task exemplar to the zero-shot prompt."""
    task = str(case["task"])
    payload = json.dumps(jsonable(case.get("input", {})), ensure_ascii=False, indent=2)
    ex_input = json.dumps(jsonable(exemplar.get("input", {})), ensure_ascii=False, indent=2)
    ex_output = json.dumps(
        jsonable(exemplar.get("exemplar_output", {})), ensure_ascii=False, indent=2
    )
    description = exemplar.get("description", "")
    example_block = (
        "--- Example ---\n"
        f"{description}\n"
        f"Input JSON:\n{ex_input}\n\n"
        f"Output:\n{ex_output}\n"
        "--- End of Example ---"
    )
    return (
        f"You are solving Klotski-Bench task {task.upper()} ({TASK_NAMES[task]}).\n\n"
        f"{KLOTSKI_RULES}\n\n"
        f"{TASK_INSTRUCTIONS[task]}\n\n"
        "Respond with JSON only (no markdown fences, no commentary).\n"
        f"Required output schema: {OUTPUT_SCHEMAS[task]}\n\n"
        f"{example_block}\n\n"
        "Now solve the following test case. Output only the final JSON object.\n\n"
        f"Input JSON:\n{payload}\n"
    )


def _load_from_case_root(case_root: Path) -> dict[str, dict[str, Any]]:
    """Load per-task cases.json files from a depth-benchmark export directory."""
    result: dict[str, dict[str, Any]] = {}
    for task in ALL_TASKS:
        task_file = case_root / task / "cases.json"
        if task_file.is_file():
            cases = json.loads(task_file.read_text(encoding="utf-8"))
            if not isinstance(cases, list):
                print(
                    f"Warning: {task_file} is not a JSON list; skipping task {task}.",
                    file=__import__("sys").stderr,
                )
                cases = []
            result[task] = {"cases": cases}
        else:
            result[task] = {"cases": []}
    return result


def acquire_task_outputs(
    force_rebuild: bool,
    case_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build cases or load from output/seven_task_validation if present.

    If case_root is provided, load per-task cases.json files from that directory
    (produced by scripts/export_depth_benchmark.py) and ignore force_rebuild.
    """
    if case_root is not None:
        return _load_from_case_root(case_root)

    if not force_rebuild:
        all_path = EXPORT_ROOT / "all_cases.json"
        if all_path.is_file():
            cases = json.loads(all_path.read_text(encoding="utf-8"))
            by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in ALL_TASKS}
            for case in cases:
                by_task.setdefault(str(case.get("task")), []).append(case)
            return {
                t: {"cases": by_task.get(t, [])}
                for t in ALL_TASKS
            }

    from export_seven_task_validation_outputs import build_task_outputs  # noqa: WPS433

    return build_task_outputs()


def select_cases(
    task_outputs: dict[str, dict[str, Any]],
    tasks: Sequence[str],
    max_per_task: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for task in tasks:
        pool = list(task_outputs.get(task, {}).get("cases", []))
        if max_per_task is not None:
            pool = pool[: int(max_per_task)]
        counts[task] = len(pool)
        selected.extend(pool)
    return selected, counts


def run_benchmark(args: argparse.Namespace) -> int:
    tasks = tuple(t.strip().lower() for t in args.tasks.split(","))
    for t in tasks:
        if t not in ALL_TASKS:
            raise SystemExit(f"Unknown task: {t!r}; expected one of {ALL_TASKS}")

    prompt_mode: str = getattr(args, "prompt_mode", "zero_shot_rules_v3")
    exemplar_root: Path | None = getattr(args, "exemplar_root", None)

    if prompt_mode == "one_shot_rules_v3":
        if exemplar_root is None:
            raise SystemExit(
                "Error: --exemplar-root is required when --prompt-mode one_shot_rules_v3 is used.\n"
                "Run: python scripts/export_one_shot_exemplars.py --out-dir <exemplar-root>"
            )
        exemplars = load_exemplars(exemplar_root)
    else:
        exemplars = {}

    slug = model_slug(args.model)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_BASE / f"run_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    task_outputs = acquire_task_outputs(
        force_rebuild=args.force_rebuild,
        case_root=getattr(args, "case_root", None),
    )
    cases, counts_by_task = select_cases(task_outputs, tasks, args.max_cases_per_task)
    if not cases:
        raise SystemExit("No cases selected for benchmark run.")

    raw_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    ollama_options = build_ollama_options(
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        num_predict=args.num_predict,
    )
    decoding_settings = decoding_settings_from_args(args)

    for index, case in enumerate(cases):
        task = str(case["task"])
        case_id = case.get("case_id", f"{task}_{index}")
        if prompt_mode == "one_shot_rules_v3":
            prompt = build_one_shot_prompt(case, exemplars[task])
        else:
            prompt = build_prompt(case)

        if args.dry_run:
            raw = dry_run_raw_response(task)
            parse_error = None
        else:
            try:
                raw = ollama_chat(
                    args.ollama_base,
                    args.model,
                    prompt,
                    options=ollama_options,
                )
                parse_error = None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                raw = f"[error: {exc!r}]"
                parse_error = f"ollama_error: {exc!r}"

        obj, json_err = parse_json_object(raw)
        if parse_error is None and json_err is not None:
            parse_error = json_err

        prediction = _normalize_prediction(task, obj)
        prediction, sanitized = sanitize_prediction_for_task(task, prediction)
        row: dict[str, Any] = {
            "index": index,
            "case_id": case_id,
            "task": task,
            "model": args.model,
            "raw_response": raw,
            "parsed_object": obj,
            "prediction": prediction,
            "parse_error": parse_error,
        }
        if sanitized:
            row["sanitized"] = True
        raw_rows.append(row)
        predictions.append(prediction)

        if not args.dry_run and args.sleep > 0 and index + 1 < len(cases):
            time.sleep(args.sleep)

    if len(predictions) != len(cases):
        raise RuntimeError("Prediction/case length mismatch (internal error).")

    eval_all = evaluate_benchmark(cases, predictions)
    per_task_eval = {
        task: evaluate_benchmark(
            [c for c in cases if c["task"] == task],
            [p for c, p in zip(cases, predictions) if c["task"] == task],
        )
        for task in tasks
    }

    table_rows = build_main_task_table(eval_all)
    md_lines = [
        "# Mini Ollama seven-task run",
        "",
        f"- model: `{args.model}`",
        f"- dry_run: {args.dry_run}",
        f"- cases: {len(cases)}",
        "",
        "| task | n | status | headline metrics |",
        "| ---- | - | ------ | ---------------- |",
    ]
    for row in table_rows:
        md_lines.append(
            f"| {row['task']} | {row['n']} | {row['status']} | "
            f"{json.dumps(row.get('metrics'), ensure_ascii=False)} |"
        )

    write_json = lambda name, obj: (out_dir / name).write_text(
        json.dumps(jsonable(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_json(f"predictions_raw_{slug}.json", {
        "model": args.model,
        "ollama_base": args.ollama_base,
        "dry_run": args.dry_run,
        "n_cases": len(cases),
        "rows": raw_rows,
    })
    write_json(f"predictions_{slug}.json", {
        "model": args.model,
        "format": "aligned with cases order",
        "predictions": predictions,
    })
    write_json(f"eval_summary_{slug}.json", eval_all)
    write_json(f"per_task_eval_{slug}.json", per_task_eval)
    write_json(f"run_metadata_{slug}.json", {
        "model": args.model,
        "ollama_base": args.ollama_base,
        "dry_run": args.dry_run,
        "tasks": list(tasks),
        "counts_by_task": counts_by_task,
        "n_cases": len(cases),
        "out_dir": str(out_dir),
        "force_rebuild_cases": args.force_rebuild,
        "max_cases_per_task": args.max_cases_per_task,
        "sleep_seconds": args.sleep,
        "prompt_mode": prompt_mode,
        "exemplar_root": str(exemplar_root) if exemplar_root is not None else None,
        "num_shots": 1 if prompt_mode == "one_shot_rules_v3" else 0,
        **decoding_settings,
    })
    write_json(f"main_task_table_{slug}.json", table_rows)
    (out_dir / f"summary_{slug}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote mini benchmark outputs under {out_dir}")
    print(f"Cases: {len(cases)} | tasks: {counts_by_task}")
    return 0


def main() -> None:
    args = parse_runner_args()
    raise SystemExit(run_benchmark(args))


if __name__ == "__main__":
    main()
