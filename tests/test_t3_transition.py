from __future__ import annotations

import pathlib
import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.generators import transition_oracle
from src.generators.transition_oracle import T3_PAPER_INVALID_REASONS
from src.metrics.t3_metrics import compute_t3_metrics
from src.tasks.t3_builder import V1_INVALID_REASONS, build_t3_cases


def _multi_block_state() -> dict:
    return {
        "blocks": {
            "A": {"shape": [1, 1], "pos": [0, 1]},
            "B": {"shape": [1, 1], "pos": [2, 3]},
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


def _legal_actions_stub(state: dict) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for block_id, block in state.get("blocks", {}).items():
        c = int(block["pos"][1])
        if block_id == "A":
            if c < 3:
                actions.append({"block_id": "A", "direction": "right"})
            if c > 0:
                actions.append({"block_id": "A", "direction": "left"})
    return actions


def _apply_action_checked_stub(state: dict, action: dict[str, str]) -> dict:
    legal = _legal_actions_stub(state)
    if not any(
        a["block_id"] == action["block_id"] and a["direction"] == action["direction"]
        for a in legal
    ):
        raise ValueError(f"Illegal action for state: {action}")
    out = {
        "blocks": {
            bid: {"shape": list(b["shape"]), "pos": list(b["pos"])}
            for bid, b in state["blocks"].items()
        },
        "grid_size": list(state["grid_size"]),
    }
    block = out["blocks"][action["block_id"]]
    if action["direction"] == "right":
        block["pos"][1] += 1
    elif action["direction"] == "left":
        block["pos"][1] -= 1
    return out


@pytest.fixture(autouse=True)
def adapter_patch():
    targets = (
        "src.core.state_adapter.validate_state",
        "src.core.state_adapter.normalize_state",
        "src.core.state_adapter.canonicalize_state",
        "src.generators.transition_oracle.state_adapter.validate_state",
        "src.generators.transition_oracle.state_adapter.normalize_state",
        "src.generators.transition_oracle.state_adapter.canonicalize_state",
        "src.core.moves_adapter.get_legal_actions",
        "src.core.moves_adapter.apply_action_checked",
        "src.core.moves_adapter.apply_action",
        "src.generators.transition_oracle.moves_adapter.get_legal_actions",
        "src.generators.transition_oracle.moves_adapter.apply_action_checked",
        "src.generators.transition_oracle.moves_adapter.apply_action",
        "src.tasks.t3_builder.moves_adapter.get_legal_actions",
        "src.tasks.t3_builder.moves_adapter.apply_action_checked",
    )
    with ExitStack() as stack:
        stack.enter_context(patch("src.core.state_adapter.validate_state", _realistic_validate))
        stack.enter_context(patch("src.generators.transition_oracle.state_adapter.validate_state", _realistic_validate))
        stack.enter_context(patch("src.core.state_adapter.normalize_state", lambda s: s))
        stack.enter_context(patch("src.generators.transition_oracle.state_adapter.normalize_state", lambda s: s))
        stack.enter_context(
            patch(
                "src.core.state_adapter.canonicalize_state",
                lambda s: ("key", tuple(sorted((bid, *b["pos"], *b["shape"]) for bid, b in s["blocks"].items()))),
            )
        )
        stack.enter_context(
            patch(
                "src.generators.transition_oracle.state_adapter.canonicalize_state",
                lambda s: ("key", tuple(sorted((bid, *b["pos"], *b["shape"]) for bid, b in s["blocks"].items()))),
            )
        )
        stack.enter_context(patch("src.core.moves_adapter.get_legal_actions", _legal_actions_stub))
        stack.enter_context(patch("src.core.moves_adapter.apply_action_checked", _apply_action_checked_stub))
        stack.enter_context(patch("src.core.moves_adapter.apply_action", _apply_action_checked_stub))
        stack.enter_context(
            patch("src.generators.transition_oracle.moves_adapter.get_legal_actions", _legal_actions_stub)
        )
        stack.enter_context(
            patch(
                "src.generators.transition_oracle.moves_adapter.apply_action_checked",
                _apply_action_checked_stub,
            )
        )
        stack.enter_context(
            patch(
                "src.generators.transition_oracle.moves_adapter.apply_action",
                _apply_action_checked_stub,
            )
        )
        stack.enter_context(patch("src.tasks.t3_builder.moves_adapter.get_legal_actions", _legal_actions_stub))
        stack.enter_context(
            patch("src.tasks.t3_builder.moves_adapter.apply_action_checked", _apply_action_checked_stub)
        )
        yield


def test_wrong_transition_alias_normalizes():
    assert (
        transition_oracle.normalize_t3_invalid_reason("wrong_transition")
        == "wrong_moved_block_position"
    )
    assert transition_oracle.is_supported_t3_invalid_reason("wrong_transition")


def test_shape_change_priority_over_stationary_drift(adapter_patch):
    state = _multi_block_state()
    action = {"block_id": "A", "direction": "right"}
    expected = transition_oracle.get_next_state(state, action)
    candidate = transition_oracle._successor_layout(state, expected, "A")  # noqa: SLF001
    candidate["blocks"]["B"]["shape"] = [2, 1]
    gold = transition_oracle.evaluate_candidate_transition(state, action, candidate)
    assert gold == {"label": "invalid", "reason": "shape_change"}
    assert not transition_oracle._has_stationary_position_drift(state, "A", candidate)  # noqa: SLF001


@pytest.mark.parametrize("reason", T3_PAPER_INVALID_REASONS)
def test_make_invalid_candidate_matches_paper_reason(reason: str, adapter_patch):
    state = _multi_block_state()
    action = {"block_id": "A", "direction": "right"}
    candidate = transition_oracle.make_invalid_candidate_next_state(state, action, reason)
    gold = transition_oracle.evaluate_candidate_transition(state, action, candidate)
    assert gold == {"label": "invalid", "reason": reason}


def test_build_t3_cases_all_paper_reasons(adapter_patch):
    items = [{"state": _multi_block_state()}]
    cases = build_t3_cases(items, include_valid=True, invalid_reasons=T3_PAPER_INVALID_REASONS)
    reasons = {
        c["gold"]["reason"]
        for c in cases
        if c["gold"].get("label") == "invalid"
    }
    assert any(c["gold"]["label"] == "valid" for c in cases)
    assert reasons == set(T3_PAPER_INVALID_REASONS)


def test_build_t3_cases_accepts_wrong_transition_alias(adapter_patch):
    items = [{"state": _multi_block_state()}]
    cases = build_t3_cases(
        items,
        include_valid=False,
        invalid_reasons=("wrong_transition",),
    )
    assert cases
    assert all(c["gold"]["reason"] == "wrong_moved_block_position" for c in cases)


def test_t3_metrics_oracle_and_wrong(adapter_patch):
    items = [{"state": _multi_block_state()}]
    cases = build_t3_cases(items, include_valid=True, invalid_reasons=T3_PAPER_INVALID_REASONS)
    oracle: list[dict] = []
    for c in cases:
        pred = {"label": c["gold"]["label"]}
        if c["gold"].get("reason"):
            pred["reason"] = c["gold"]["reason"]
        oracle.append(pred)
    wrong = [
        {
            "label": "valid" if c["gold"]["label"] == "invalid" else "invalid",
            "reason": c["gold"].get("reason", "overlap"),
        }
        for c in cases
    ]
    mo = compute_t3_metrics(cases, oracle)
    mw = compute_t3_metrics(cases, wrong)
    assert mo["label_accuracy"] == 1.0
    assert mo["joint_verification_reason_accuracy"] == 1.0
    assert mw["label_accuracy"] < mo["label_accuracy"]


def test_t3_metrics_accepts_wrong_transition_prediction_alias(adapter_patch):
    cases = [{"gold": {"label": "invalid", "reason": "wrong_moved_block_position"}}]
    preds = [{"label": "invalid", "reason": "wrong_transition"}]
    m = compute_t3_metrics(cases, preds)
    assert m["joint_verification_reason_accuracy"] == 1.0


def test_v1_invalid_reasons_includes_legacy_alias():
    assert "wrong_transition" in V1_INVALID_REASONS
    assert "wrong_moved_block_position" in T3_PAPER_INVALID_REASONS
