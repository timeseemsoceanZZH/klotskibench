#!/usr/bin/env python3
"""Print paths to smoke JSON artifacts and a short eval_summary preview."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap_paths

_bootstrap_paths.ensure_repo_sys_path()

_DEFAULT_RUN_DIR = Path("output/smoke_r1/run_phi4-mini")
_DEFAULT_MODEL = "phi4-mini:latest"


def _model_slug(model: str) -> str:
    s = re.sub(r"[^\w.\-]+", "-", str(model)).strip("-")
    return s or "model"


def _slug_from_eval_in_dir(d: Path) -> str | None:
    pats = sorted(d.glob("eval_summary_*.json"))
    if not pats:
        return None
    name = pats[0].stem
    if name.startswith("eval_summary_"):
        return name[len("eval_summary_") :]
    return None


def _infer_slug_from_run_dir_name(d: Path) -> str | None:
    n = d.name
    if not n.startswith("run_") or n == "run" or n == "run_":
        return None
    tail = n[4:].replace("_", "-")
    # Ollama smoke runs use names like "qwen2.5-14b:latest" → file slug "qwen2.5-14b-latest";
    # folder names often omit the tag, so we align with that common case.
    if ":" not in tail:
        return _model_slug(f"{tail}:latest")
    return _model_slug(tail)


def main() -> None:
    p = argparse.ArgumentParser(
        description="List smoke R1 result JSON files (default: phi4-mini under output/smoke_r1/run_phi4-mini)."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=_DEFAULT_RUN_DIR,
        help="Directory from run_smoke_ollama_r1.py (default: output/smoke_r1/run_phi4-mini).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Ollama model name as passed to the run (same as --model there, e.g. qwen2.5-14b:latest). "
            "If omitted: read suffix from eval_summary_*.json in --run-dir when it exists, "
            "else from run_FOO directory name, assuming an implicit :latest tag (FOO -> FOO-latest)."
        ),
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not print eval summary JSON to stdout",
    )
    args = p.parse_args()
    d = args.run_dir
    if args.model is not None:
        safe = _model_slug(args.model)
        suffix_note = f"suffix from --model {args.model!r}"
    elif d.is_dir() and (g := _slug_from_eval_in_dir(d)) is not None:
        safe = g
        suffix_note = "suffix from eval_summary_*.json in --run-dir"
    elif d.resolve() == _DEFAULT_RUN_DIR.resolve():
        safe = _model_slug(_DEFAULT_MODEL)
        suffix_note = f"default ({_DEFAULT_MODEL!r})"
    elif (inf := _infer_slug_from_run_dir_name(d)) is not None:
        safe = inf
        suffix_note = (
            f"inferred from directory name {d.name!r} (assumes :latest tag, same as `mini_bench_run_ollama`"
            f" slugs; use --model if you used a different Ollama tag)"
        )
    else:
        safe = _model_slug(_DEFAULT_MODEL)
        suffix_note = f"fallback default ({_DEFAULT_MODEL!r}); pass --model to match your run"
    order = [
        f"predictions_raw_{safe}.json",
        f"predictions_{safe}.json",
        f"eval_summary_{safe}.json",
        f"main_task_table_{safe}.json",
        f"depth_wise_report_{safe}.json",
    ]
    print(f"File name suffix: {safe!r} ({suffix_note})")
    if not d.is_dir():
        dr = d.resolve()
        print(f"Run output directory not found (smoke run not done yet or different --out-dir): {dr}")
        for name in order:
            print(f"expected  (missing)  {dr / name}")
        raise SystemExit(0)
    for name in order:
        pth = d / name
        status = "ok" if pth.is_file() else "missing"
        print(f"{status}  {pth}")
    eval_path = d / f"eval_summary_{safe}.json"
    if not args.no_preview and eval_path.is_file():
        print()
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        print("eval_summary (full JSON at path above):")
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
