"""Metrics for R1 trajectory-generation tasks."""

from __future__ import annotations

from typing import Any, Sequence

from src.core import moves_adapter, state_adapter
from src.core.goal import is_solved_state


def compute_r1_metrics(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute SR@N, Alive(t), TCR(t), and Efficiency for R1.

    Success (SR@N): simulate the full predicted trajectory; every step must be a
    legal action on a valid state, and the final state must satisfy
    ``is_solved_state``. Exact match to ``meta.goal_state`` is not required.
    Legal detours (extra legal moves) can succeed if the full execution ends in G;
    efficiency is ``optimal_depth / len(predicted_actions)`` for successful runs.

    TCR(t) compares step t to the packaged shortest reference prefix.
    Alive(t) tracks prefix legality and state validity only.
    """
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return {
            "sr_at_n": {},
            "alive_t": {},
            "tcr_t": {},
            "efficiency": 0.0,
        }

    sr_counts: dict[int, dict[str, int]] = {}
    alive_counts: dict[int, dict[str, int]] = {}
    tcr_counts: dict[int, dict[str, int]] = {}
    efficiency_values: list[float] = []

    for case, pred in zip(cases, predictions):
        initial_state = state_adapter.normalize_state(case["input"]["initial_state"])
        optimal_depth = int(case["gold"]["optimal_depth"])
        step_budget_n = case.get("meta", {}).get("step_budget_n")
        if step_budget_n is not None:
            step_budget_n = int(step_budget_n)
        ref_actions = list(case["meta"]["shortest_solution_actions"])
        pred_actions = _extract_predicted_actions(pred)

        eval_out = _evaluate_predicted_trajectory(
            initial_state=initial_state,
            optimal_depth=optimal_depth,
            step_budget_n=step_budget_n,
            predicted_actions=pred_actions,
            reference_actions=ref_actions,
        )

        bucket = sr_counts.setdefault(optimal_depth, {"success": 0, "total": 0})
        bucket["total"] += 1
        if eval_out["strict_success"]:
            bucket["success"] += 1

        for t, alive in eval_out["alive_t"].items():
            c = alive_counts.setdefault(t, {"ok": 0, "total": 0})
            c["total"] += 1
            if alive:
                c["ok"] += 1

        for t, tcr in eval_out["tcr_t"].items():
            c = tcr_counts.setdefault(t, {"ok": 0, "total": 0})
            c["total"] += 1
            if tcr:
                c["ok"] += 1

        if eval_out["strict_success"]:
            efficiency_values.append(
                float(optimal_depth) / float(eval_out["successful_length"])
            )

    sr_at_n = {
        depth: (counts["success"] / counts["total"] if counts["total"] else 0.0)
        for depth, counts in sorted(sr_counts.items())
    }
    alive_t = {
        t: (counts["ok"] / counts["total"] if counts["total"] else 0.0)
        for t, counts in sorted(alive_counts.items())
    }
    tcr_t = {
        t: (counts["ok"] / counts["total"] if counts["total"] else 0.0)
        for t, counts in sorted(tcr_counts.items())
    }
    efficiency = sum(efficiency_values) / len(efficiency_values) if efficiency_values else 0.0

    return {
        "sr_at_n": sr_at_n,
        "alive_t": alive_t,
        "tcr_t": tcr_t,
        "efficiency": efficiency,
    }


def _evaluate_predicted_trajectory(
    initial_state: state_adapter.KlotskiState,
    optimal_depth: int,
    step_budget_n: int | None,
    predicted_actions: list[dict[str, str]],
    reference_actions: list[dict[str, str]],
) -> dict[str, Any]:
    current = state_adapter.normalize_state(initial_state)
    alive_t: dict[int, bool] = {}
    tcr_t: dict[int, bool] = {}

    alive_prefix = True
    tcr_prefix = True

    for t, action in enumerate(predicted_actions, start=1):
        if alive_prefix:
            legal_actions = moves_adapter.get_legal_actions(current)
            if _action_matches_any(action, legal_actions):
                current = moves_adapter.apply_action(current, action)
                alive_prefix = bool(state_adapter.validate_state(current))
            else:
                alive_prefix = False
        alive_t[t] = alive_prefix

        if tcr_prefix:
            if t > len(reference_actions):
                tcr_prefix = False
            else:
                tcr_prefix = _action_matches_any(action, [reference_actions[t - 1]])
        tcr_t[t] = tcr_prefix

    within_budget = (
        len(predicted_actions) <= step_budget_n
        if step_budget_n is not None
        else True
    )
    final_solved = is_solved_state(current)
    strict_success = (
        (all(alive_t.values()) if alive_t else True)
        and final_solved
        and within_budget
    )

    return {
        "alive_t": alive_t,
        "tcr_t": tcr_t,
        "strict_success": strict_success,
        "successful_length": len(predicted_actions),
    }


def _action_matches_any(
    action: dict[str, str], legal_actions: list[dict[str, str]]
) -> bool:
    return any(
        str(action.get("block_id")) == str(legal.get("block_id"))
        and str(action.get("direction")) == str(legal.get("direction"))
        for legal in legal_actions
    )


def _extract_predicted_actions(prediction: dict[str, Any]) -> list[dict[str, str]]:
    if "predicted_trajectory" in prediction:
        return list(prediction["predicted_trajectory"])
    if "trajectory" in prediction:
        return list(prediction["trajectory"])
    if "actions" in prediction:
        return list(prediction["actions"])
    raise ValueError("Prediction must include one of: predicted_trajectory, trajectory, actions")


__all__ = [
    "compute_r1_metrics",
]
