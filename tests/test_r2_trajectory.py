from __future__ import annotations

import pathlib
import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.generators.trajectory_corruptor import R2_PAPER_INVALID_TRAJECTORY_TYPES
from src.metrics.r2_metrics import compute_r2_metrics
from src.tasks.r2_builder import build_r2_cases, evaluate_candidate_trajectory


def _exact_case() -> dict:
    return {
        "initial_state": {
            "blocks": {"A": {"shape": [1, 1], "pos": [0, 0]}},
            "grid_size": [5, 4],
        },
        "goal_state": {
            "blocks": {"A": {"shape": [1, 1], "pos": [0, 1]}},
            "grid_size": [5, 4],
        },
        "optimal_depth": 1,
        "shortest_solution_actions": [{"block_id": "A", "direction": "right"}],
        "meta": {"canonical_state_key": ("A", 0, 0)},
    }


def _is_solved_stub(state: dict) -> bool:
    return int(state["blocks"]["A"]["pos"][1]) >= 1


def _legal_actions_stub(state: dict) -> list[dict[str, str]]:
    block = state["blocks"]["A"]
    c = int(block["pos"][1])
    actions: list[dict[str, str]] = []
    if c < 3:
        actions.append({"block_id": "A", "direction": "right"})
    if c > 0:
        actions.append({"block_id": "A", "direction": "left"})
    return actions


def _apply_action_stub(state: dict, action: dict[str, str]) -> dict:
    out = {
        "blocks": {"A": {"shape": [1, 1], "pos": list(state["blocks"]["A"]["pos"])}},
        "grid_size": [5, 4],
    }
    if action["direction"] == "right":
        out["blocks"]["A"]["pos"][1] += 1
    elif action["direction"] == "left":
        out["blocks"]["A"]["pos"][1] -= 1
    return out


@pytest.fixture(autouse=True)
def moves_patch():
    with ExitStack() as stack:
        stack.enter_context(
            patch("src.core.goal.is_solved_state", side_effect=_is_solved_stub)
        )
        stack.enter_context(
            patch("src.tasks.r2_builder.is_solved_state", side_effect=_is_solved_stub)
        )
        for mod in (
            "src.core.moves_adapter",
            "src.tasks.r2_builder.moves_adapter",
        ):
            stack.enter_context(
                patch(f"{mod}.get_legal_actions", side_effect=_legal_actions_stub)
            )
            stack.enter_context(
                patch(f"{mod}.apply_action", side_effect=_apply_action_stub)
            )
        yield


def test_valid_shortest_trajectory(moves_patch):
    case = _exact_case()
    gold = evaluate_candidate_trajectory(
        case["initial_state"], case["shortest_solution_actions"]
    )
    assert gold == {"label": "valid"}


def test_legal_detour_valid(moves_patch):
    detour = [
        {"block_id": "A", "direction": "right"},
        {"block_id": "A", "direction": "left"},
        {"block_id": "A", "direction": "right"},
    ]
    gold = evaluate_candidate_trajectory(_exact_case()["initial_state"], detour)
    assert gold == {"label": "valid"}


def test_wrong_action_invalid(moves_patch):
    case = _exact_case()
    gold = evaluate_candidate_trajectory(
        case["initial_state"],
        [{"block_id": "A", "direction": "left"}],
    )
    assert gold["label"] == "invalid"
    assert gold["first_error_step"] == 1
    assert gold["step_validity_sequence"] == [0]


def test_goal_not_reached_convention(moves_patch):
    case = _exact_case()
    gold = evaluate_candidate_trajectory(case["initial_state"], [])
    assert gold["label"] == "invalid"
    assert gold["first_error_step"] == 1
    assert gold["step_validity_sequence"] == []


def test_goal_not_reached_prefix(moves_patch):
    cases = build_r2_cases(
        [_exact_case()],
        include_valid=False,
        invalid_types=("goal_not_reached",),
        seed=0,
    )
    assert cases
    invalid = cases[0]["gold"]
    assert invalid["label"] == "invalid"
    traj_len = len(cases[0]["input"]["candidate_trajectory"])
    assert invalid["first_error_step"] == traj_len + 1
    assert invalid["step_validity_sequence"] == [1] * traj_len


def test_all_legal_steps_not_in_g_is_goal_not_reached(moves_patch):
    gold = evaluate_candidate_trajectory(
        _exact_case()["initial_state"],
        [
            {"block_id": "A", "direction": "right"},
            {"block_id": "A", "direction": "left"},
        ],
    )
    assert gold["label"] == "invalid"
    assert gold["first_error_step"] == 3
    assert gold["step_validity_sequence"] == [1, 1]


@pytest.mark.parametrize("invalid_type", R2_PAPER_INVALID_TRAJECTORY_TYPES)
def test_build_r2_invalid_types(invalid_type: str, moves_patch):
    built = build_r2_cases(
        [_exact_case()],
        include_valid=True,
        invalid_types=(invalid_type,),
        seed=1,
    )
    invalid_cases = [c for c in built if c["gold"]["label"] == "invalid"]
    assert invalid_cases
    assert invalid_cases[0]["meta"]["trajectory_type"] == invalid_type


def test_r2_metrics_oracle_and_wrong(moves_patch):
    built = build_r2_cases([_exact_case()], seed=2)
    oracle = []
    for c in built:
        g = c["gold"]
        if g["label"] == "valid":
            oracle.append({"label": "valid"})
        else:
            oracle.append(
                {
                    "label": g["label"],
                    "first_error_step": g.get("first_error_step"),
                    "step_validity_sequence": list(g.get("step_validity_sequence", [])),
                }
            )
    wrong = [{"label": "valid" if o["label"] == "invalid" else "invalid"} for o in oracle]
    mo = compute_r2_metrics(built, oracle)
    mw = compute_r2_metrics(built, wrong)
    assert mo["trajectory_verification_accuracy"] == 1.0
    assert mw["trajectory_verification_accuracy"] < mo["trajectory_verification_accuracy"]
