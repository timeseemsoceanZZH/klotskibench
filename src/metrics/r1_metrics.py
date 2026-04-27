"""Metrics for R1 trajectory-generation tasks.

Benchmark decisions (frozen for this codebase):

1) **R1 strict success (SR@N) — exact goal state, not "any solved"**:
   The final state after the predicted actions must match **case["meta"]["goal_state"]**
   by ``canonicalize_state`` equality. We do **not** use ``is_goal_state`` alone, so
   distinct valid goal encodings that are not exactly the reference goal for that
   case do not count as success. This matches exact-depth / oracle-goal training data.

2) **TCR(t) — reference-chain correctness, not "any optimal path"**:
   At each step t, the predicted action must match **meta["shortest_solution_actions"][t-1]**
   (the packaged shortest path from the generator), not any alternative optimal plan.
   There may be multiple optimal trajectories; this metric scores alignment with the
   benchmark reference, not the full set of optimal moves.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.core import moves_adapter, state_adapter


def compute_r1_metrics(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute SR@N, Alive(t), TCR(t), and Efficiency for R1.

    See module docstring for R1 success and TCR definitions.
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
        goal_state = state_adapter.normalize_state(case["meta"]["goal_state"])
        optimal_depth = int(case["gold"]["optimal_depth"])
        ref_actions = list(case["meta"]["shortest_solution_actions"])
        pred_actions = _extract_predicted_actions(pred)

        eval_out = _evaluate_predicted_trajectory(
            initial_state=initial_state,
            goal_state=goal_state,
            optimal_depth=optimal_depth,
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
            efficiency_values.append(float(optimal_depth) / float(eval_out["successful_length"]))

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
    goal_state: state_adapter.KlotskiState,
    optimal_depth: int,
    predicted_actions: list[dict[str, str]],
    reference_actions: list[dict[str, str]],
) -> dict[str, Any]:
    # Strict success: exact reference goal (canonical), not is_goal_state(final).
    # TCR: stepwise match to reference_actions (shortest_solution_actions), not any optimal path.
    current = state_adapter.normalize_state(initial_state)
    alive_t: dict[int, bool] = {}
    tcr_t: dict[int, bool] = {}

    alive_prefix = True
    tcr_prefix = True

    for t, action in enumerate(predicted_actions, start=1):
        if alive_prefix:
            legal_actions = moves_adapter.get_legal_actions(current)
            if action in legal_actions:
                current = moves_adapter.apply_action(current, action)
                alive_prefix = bool(state_adapter.validate_state(current))
            else:
                alive_prefix = False
        alive_t[t] = alive_prefix

        if tcr_prefix:
            if t > len(reference_actions):
                tcr_prefix = False
            else:
                tcr_prefix = action == reference_actions[t - 1]
        tcr_t[t] = tcr_prefix

    final_is_goal = (
        state_adapter.canonicalize_state(current)
        == state_adapter.canonicalize_state(goal_state)
    )
    strict_success = (
        all(alive_t.values()) if alive_t else True
    ) and final_is_goal and (len(predicted_actions) == optimal_depth)

    return {
        "alive_t": alive_t,
        "tcr_t": tcr_t,
        "strict_success": strict_success,
        "successful_length": len(predicted_actions),
    }


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
