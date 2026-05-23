from __future__ import annotations

import pathlib
import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.generators.invalid_state_generator import (
    PAPER_STATE_ERROR_TYPES,
    generate_invalid_state,
    generate_invalid_state_coverage,
)
from src.tasks.s1_builder import build_s1_cases, build_s1_validation_sample
from src.tasks.s2_builder import build_s2_cases, build_s2_coverage_cases


def _multi_block_state() -> dict:
    return {
        "blocks": {
            "A": {"shape": [1, 1], "pos": [0, 0]},
            "B": {"shape": [1, 1], "pos": [2, 1]},
            "C": {"shape": [2, 1], "pos": [3, 0]},
        },
        "grid_size": [5, 4],
    }


def _realistic_validate(state: dict) -> bool:
    if not state.get("blocks") or "grid_size" not in state:
        return False
    rows, cols = int(state["grid_size"][0]), int(state["grid_size"][1])
    occupied: set[tuple[int, int]] = set()
    for block in state["blocks"].values():
        r, c = int(block["pos"][0]), int(block["pos"][1])
        h, w = int(block["shape"][0]), int(block["shape"][1])
        if h <= 0 or w <= 0 or r < 0 or c < 0 or r + h > rows or c + w > cols:
            return False
        for dr in range(h):
            for dc in range(w):
                cell = (r + dr, c + dc)
                if cell in occupied:
                    return False
                occupied.add(cell)
    return len(occupied) > 0


def _adapter_patch_targets() -> list[str]:
    modules = (
        "src.core.state_adapter",
        "src.generators.invalid_state_generator.state_adapter",
        "src.tasks.s1_builder.state_adapter",
        "src.tasks.s2_builder.state_adapter",
    )
    targets: list[str] = []
    for mod in modules:
        targets.append(f"{mod}.validate_state")
        targets.append(f"{mod}.normalize_state")
        targets.append(f"{mod}.canonicalize_state")
    return targets


@pytest.fixture(autouse=True)
def state_adapter_patch():
    patches = [
        patch(target, side_effect=_realistic_validate)
        if target.endswith("validate_state")
        else patch(target, side_effect=lambda s: s)
        if target.endswith("normalize_state")
        else patch(target, side_effect=lambda s: ("test_key", len(s.get("blocks", {}))))
        for target in _adapter_patch_targets()
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@pytest.mark.parametrize("error_type", PAPER_STATE_ERROR_TYPES)
def test_generate_invalid_state_covers_all_types(
    error_type: str, state_adapter_patch
):
    sample = generate_invalid_state(_multi_block_state(), error_type, seed=1)
    assert sample["meta"]["error_type"] == error_type
    assert sample["errors"]
    assert sample["errors"][0]["error_type"] == error_type


def test_overlap_error_has_richer_fields(state_adapter_patch):
    sample = generate_invalid_state(_multi_block_state(), "overlap", seed=2)
    err = sample["errors"][0]
    assert "block_ids" in err
    assert "cells" in err
    assert len(err["cells"]) >= 1


def test_missing_block_error_fields(state_adapter_patch):
    sample = generate_invalid_state(_multi_block_state(), "missing_block", seed=3)
    err = sample["errors"][0]
    assert err["missing_block_id"] in _multi_block_state()["blocks"]
    assert err["block_id"] == err["missing_block_id"]


def test_coverage_produces_all_five_types(state_adapter_patch):
    samples = generate_invalid_state_coverage([_multi_block_state()], seed=99)
    types = {s["meta"]["error_type"] for s in samples}
    assert types == set(PAPER_STATE_ERROR_TYPES)


def test_build_s1_cases_valid_and_invalid_labels(state_adapter_patch):
    items = [{"state": _multi_block_state(), "optimal_depth": 0}]
    cases = build_s1_cases(items, seed=7)
    labels = {c["gold"]["label"] for c in cases}
    assert "valid" in labels
    assert "invalid" in labels
    invalid_types = {c["meta"]["error_type"] for c in cases if c["gold"]["label"] == "invalid"}
    assert invalid_types == set(PAPER_STATE_ERROR_TYPES)


def test_build_s1_validation_sample_stratified(state_adapter_patch):
    items = [{"state": _multi_block_state(), "optimal_depth": 0}]
    cases = build_s1_validation_sample(items, seed=8)
    assert len(cases) == 1 + len(PAPER_STATE_ERROR_TYPES)
    assert cases[0]["gold"]["label"] == "valid"
    invalid_types = {c["meta"]["error_type"] for c in cases[1:]}
    assert invalid_types == set(PAPER_STATE_ERROR_TYPES)


def test_build_s2_coverage_all_types(state_adapter_patch):
    cases = build_s2_coverage_cases([_multi_block_state()], seed=10)
    assert len(cases) == len(PAPER_STATE_ERROR_TYPES)
    types = {c["meta"]["error_type"] for c in cases}
    assert types == set(PAPER_STATE_ERROR_TYPES)


def test_build_s2_cases_all_types_per_source(state_adapter_patch):
    cases = build_s2_cases([_multi_block_state()], seed=11)
    types = {c["meta"]["error_type"] for c in cases}
    assert types == set(PAPER_STATE_ERROR_TYPES)


def test_s1_s2_metrics_oracle_and_wrong(state_adapter_patch):
    from src.metrics.s1_metrics import compute_s1_metrics
    from src.metrics.s2_metrics import compute_s2_metrics

    items = [{"state": _multi_block_state()}]
    s1_cases = build_s1_validation_sample(items, seed=12)
    s1_oracle = [{"label": c["gold"]["label"]} for c in s1_cases]
    s1_wrong = [
        {"label": "valid" if c["gold"]["label"] == "invalid" else "invalid"} for c in s1_cases
    ]
    mo = compute_s1_metrics(s1_cases, s1_oracle)
    mw = compute_s1_metrics(s1_cases, s1_wrong)
    assert mo["accuracy"] == 1.0
    assert mw["accuracy"] < mo["accuracy"]

    s2_cases = build_s2_coverage_cases([_multi_block_state()], seed=13)
    s2_oracle = [{"errors": list(c["gold"]["errors"])} for c in s2_cases]
    s2_wrong = [{"errors": [{"block_id": "WRONG", "error_type": "boundary"}]} for c in s2_cases]
    mo2 = compute_s2_metrics(s2_cases, s2_oracle)
    mw2 = compute_s2_metrics(s2_cases, s2_wrong)
    assert mo2["exact_error_match"] == 1.0
    assert mw2["exact_error_match"] < mo2["exact_error_match"]
