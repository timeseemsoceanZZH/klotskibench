#!/usr/bin/env python3
"""
List local Ollama models and run one model on a mini R1 JSON subset; write predictions
and evaluation exports (summary, main task table, depth-wise R1 report).

Usage:
  python scripts/mini_bench_run_ollama.py list --ollama-base http://127.0.0.1:11434
  python scripts/mini_bench_run_ollama.py run --subset output/mini_r1/mini_r1_subset.json \\
      --model llama3.2 --out-dir output/mini_r1/run1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import _bootstrap_paths

_bootstrap_paths.ensure_repo_sys_path()

from src.evaluate import evaluate_benchmark
from src.reporting import build_main_task_table, build_depth_wise_trajectory_report


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ollama_tags(ollama_base: str) -> dict[str, Any]:
    base = ollama_base.rstrip("/")
    url = f"{base}/api/tags"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def build_ollama_chat_request_body(
    model: str,
    user: str,
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "stream": False,
    }
    if options:
        body["options"] = options
    return body


def ollama_chat(
    ollama_base: str,
    model: str,
    user: str,
    *,
    options: dict[str, Any] | None = None,
) -> str:
    base = ollama_base.rstrip("/")
    body = json.dumps(
        build_ollama_chat_request_body(model, user, options=options)
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("message", {}).get("content", "") or ""


def _extract_json_block(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    a = text.find("{")
    b = text.rfind("}")
    if 0 <= a < b:
        return text[a : b + 1]
    return None


def _parse_moves(raw: str) -> list[dict[str, str]]:
    jtxt = _extract_json_block(raw) or raw
    try:
        obj = json.loads(jtxt)
    except json.JSONDecodeError:
        return []
    moves = obj.get("moves") or obj.get("actions") or obj.get("predicted_trajectory")
    if not isinstance(moves, list):
        return []
    out: list[dict[str, str]] = []
    for m in moves:
        if isinstance(m, dict) and "block_id" in m and "direction" in m:
            out.append(
                {
                    "block_id": str(m["block_id"]),
                    "direction": str(m["direction"]),
                }
            )
    return out


def _build_prompt(case: dict[str, Any]) -> str:
    initial = case["input"]["initial_state"]
    return (
        "You are solving a Klotski (Huarong Road) sliding block puzzle on a 5x4 grid.\n"
        "The goal (reference) is encoded in the benchmark metadata; you must move blocks "
        "by one cell at a time (up, down, left, right) where legal. Output ONLY valid JSON, "
        "no markdown, with this exact shape:\n"
        '{"moves":[{"block_id":"<string>","direction":"up|down|left|right"},...]}\n\n'
        "Current puzzle state (normalized JSON, blocks with shape and pos [row, col]):\n"
        f"{json.dumps(initial, ensure_ascii=False)}\n"
    )


def cmd_list(args: argparse.Namespace) -> None:
    try:
        data = ollama_tags(args.ollama_base)
    except (urllib.error.URLError, TimeoutError) as e:
        raise SystemExit(f"Ollama not reachable at {args.ollama_base}: {e}")
    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    if not models:
        print("No models reported by Ollama.")
        return
    for name in sorted(models):
        print(name)


def cmd_run(args: argparse.Namespace) -> None:
    subset = _read_json(args.subset)
    cases: list[dict] = list(subset.get("cases", []))
    if not cases:
        raise SystemExit("No cases in subset file.")

    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    ollama_base = args.ollama_base
    model = args.model

    raw_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"case_{i}")
        user = _build_prompt(case)
        if args.dry_run:
            raw, moves = "[dry-run]", []
        else:
            try:
                raw = ollama_chat(ollama_base, model, user)
                moves = _parse_moves(raw)
            except (urllib.error.URLError, TimeoutError, Exception) as e:  # noqa: BLE001
                raw, moves = f"[error: {e!r}]", []
        raw_rows.append(
            {
                "index": i,
                "case_id": case_id,
                "model": model,
                "raw_response": raw,
                "parsed_moves": moves,
            }
        )
        predictions.append({"predicted_trajectory": moves})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-]+", "-", model).strip("-") or "model"
    (out / f"predictions_raw_{safe}.json").write_text(
        json.dumps(
            {
                "ollama_base": ollama_base,
                "model": model,
                "subset": str(args.subset),
                "n": len(cases),
                "rows": raw_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out / f"predictions_{safe}.json").write_text(
        json.dumps(
            {
                "format": "aligned with cases order",
                "model": model,
                "predictions": predictions,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ev = evaluate_benchmark(cases, predictions)
    (out / f"eval_summary_{safe}.json").write_text(
        json.dumps(ev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    mtable = build_main_task_table(ev)
    (out / f"main_task_table_{safe}.json").write_text(
        json.dumps(mtable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dwr = build_depth_wise_trajectory_report(cases, predictions)
    (out / f"depth_wise_report_{safe}.json").write_text(
        json.dumps(dwr, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote results under {out} (model slug: {safe})")


def run_ollama_benchmark(
    subset: Path,
    model: str,
    out_dir: Path,
    *,
    ollama_base: str = "http://127.0.0.1:11434",
    max_cases: int | None = None,
    dry_run: bool = False,
) -> None:
    """Run evaluation for one model; used by CLIs and smoke entrypoints."""
    args = argparse.Namespace(
        subset=subset,
        model=model,
        out_dir=out_dir,
        ollama_base=ollama_base,
        max_cases=max_cases,
        dry_run=dry_run,
    )
    cmd_run(args)


def main() -> None:
    p = argparse.ArgumentParser(description="Ollama mini-bench list/run")
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List local Ollama models (GET /api/tags).")
    p_list.add_argument(
        "--ollama-base",
        default="http://127.0.0.1:11434",
        help="Ollama HTTP base URL (default: http://127.0.0.1:11434).",
    )
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="Run one model on mini subset and export results.")
    p_run.add_argument("--subset", type=Path, required=True, help="JSON from mini_bench_build_subset.py")
    p_run.add_argument("--model", type=str, required=True, help="Ollama model name, e.g. llama3.2")
    p_run.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/mini_r1/run_ollama"),
        help="Output directory for predictions and evaluation",
    )
    p_run.add_argument(
        "--ollama-base",
        default="http://127.0.0.1:11434",
        help="Ollama HTTP base URL",
    )
    p_run.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Run only the first N cases (smoke test).",
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="No HTTP calls; empty trajectories, still write exports",
    )
    p_run.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
