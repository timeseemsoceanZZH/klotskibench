#!/usr/bin/env python3
"""Validate exact-depth per-depth JSON (and optional combined) produced by
``build_exact_depth_layouts_1_15.py``."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import _bootstrap_paths

_bootstrap_paths.ensure_repo_sys_path()

_REQUIRED_CASE_KEYS = frozenset(
    {
        "case_id",
        "initial_state",
        "goal_state",
        "optimal_depth",
        "shortest_solution_actions",
        "shortest_solution_states",
        "meta",
    }
)
_LEGACY_ALIAS_KEYS = frozenset({"state", "depth", "number_of_moves"})


def _load_build_validators() -> Any:
    """Reuse validation from the exact-depth build script (single source of truth)."""
    here = Path(__file__).resolve().parent
    p = here / "build_exact_depth_layouts_1_15.py"
    spec = importlib.util.spec_from_file_location("klotskibench_ed_build", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load build_exact_depth_layouts_1_15")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _find_per_depth_files(root: Path) -> list[Path]:
    pats = sorted(
        root.glob("dataset_N*.json"),
        key=lambda x: (len(x.name), x.name),
    )
    return [p for p in pats if re.match(r"^dataset_N\d+\.json$", p.name)]


def _check_no_legacy(c: dict[str, Any], path: str) -> list[str]:
    out: list[str] = []
    for k in c:
        if k in _LEGACY_ALIAS_KEYS:
            out.append(f"{path}: disallowed legacy key {k!r}")
    m = c.get("meta")
    if isinstance(m, dict):
        for k in m:
            if k in _LEGACY_ALIAS_KEYS:
                out.append(f"{path}.meta: disallowed legacy key {k!r}")
    return out


def _validate_file(
    m: Any,
    path: Path,
    forward_all: bool,
) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("schema", "")
    if "exact_depth" not in str(schema) and "exact_depth_layouts" not in str(schema):
        return [f"{path}: missing expected schema tag (got {schema!r})"]
    if "cases" not in data:
        return [f"{path}: missing top-level 'cases'"]
    depth = int(data["depth"])
    per = int(data.get("per_depth", 0) or 0)
    n_cases = len(data.get("cases") or [])
    if n_cases != per:
        return [f"{path}: n_cases {n_cases} != per_depth {per}"]

    errs: list[str] = []
    cases = data["cases"]
    fw_k = 10**9 if forward_all else 2
    errs.extend(
        m._validate_depth_file(  # type: ignore[attr-defined]
            cases, depth, per, forward_first_k=fw_k
        )
    )
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            errs.append(f"case {i} not a dict")
            continue
        missing = _REQUIRED_CASE_KEYS - c.keys()
        if missing:
            errs.append(f"case {i}: missing keys: {sorted(missing)}")
        for bad in _check_no_legacy(c, f"case[{i}]"):
            errs.append(bad)
    return errs


def _validate_combined(
    m: Any,
    path: Path,
    forward_all: bool,
) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "cases_by_depth" not in data:
        return [f"{path}: expected combined layout with 'cases_by_depth'"]
    errs: list[str] = []
    for ks, v in data["cases_by_depth"].items():
        d = int(ks)
        per = int(data.get("per_depth", 0) or 0)
        if not isinstance(v, list) or not v:
            errs.append(f"depth {d}: not a non-empty list")
            continue
        depth = d
        fw_k = 10**9 if forward_all else 2
        sub = f"{path} depth {d}"
        e2 = m._validate_depth_file(  # type: ignore[attr-defined]
            v, depth, per, forward_first_k=fw_k
        )
        for e in e2:
            errs.append(f"{sub}: {e}")
    return errs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate exact-depth dataset_*.json under a directory (default: output/exact_depth)."
    )
    p.add_argument(
        "--dir",
        type=Path,
        default=Path("output/exact_depth"),
        help="Directory with dataset_N{depth}.json (default: output/exact_depth).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Stricter check: full forward re-sim on all cases, reject legacy key aliases.",
    )
    p.add_argument(
        "--include-combined",
        action="store_true",
        help="Also validate dataset_N*_L_*each.json (combined) if present.",
    )
    args = p.parse_args()
    root: Path = args.dir
    if not root.is_dir():
        print(
            f"ERROR: directory does not exist: {root}\n"
            f"  Generate with: python scripts/build_exact_depth_layouts_1_15.py",
            file=sys.stderr,
        )
        raise SystemExit(2)

    m = _load_build_validators()
    all_errs: list[str] = []
    per_depth = _find_per_depth_files(root)
    if not per_depth:
        print(
            f"ERROR: no dataset_N<depth>.json in {root}\n"
            f"  Run: python scripts/build_exact_depth_layouts_1_15.py --out-dir {root}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    for fp in per_depth:
        all_errs.extend(_validate_file(m, fp, args.strict))

    if args.include_combined:
        for fp in sorted(root.glob("dataset_*.json")):
            if re.match(
                r"^dataset_N\d+_\d+_.*each\.json$", fp.name
            ) and fp not in per_depth:
                all_errs.extend(_validate_combined(m, fp, args.strict))

    if all_errs:
        print("Validation failed:", file=sys.stderr)
        for e in all_errs[:200]:
            print(f"  {e}", file=sys.stderr)
        if len(all_errs) > 200:
            print(f"  ... and {len(all_errs) - 200} more", file=sys.stderr)
        raise SystemExit(1)

    print(f"OK: {len(per_depth)} per-depth file(s) under {root.resolve()}")
    for fp in per_depth:
        print(f"  {fp.name}")


if __name__ == "__main__":
    main()
