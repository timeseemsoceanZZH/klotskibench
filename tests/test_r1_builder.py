from __future__ import annotations

import importlib
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _install_legacy_state_stub() -> None:
    legacy_state = types.ModuleType("klotski_state")
    legacy_state.CanonicalKey = tuple
    legacy_state.KlotskiState = dict
    legacy_state.normalize_state = lambda s: s
    legacy_state.canonicalize_state = lambda s: tuple(sorted(s.items()))
    legacy_state.validate_state = lambda s: True
    legacy_state.is_goal_state = lambda s: False
    legacy_state.to_flat_state = lambda s: dict(s)
    sys.modules["klotski_state"] = legacy_state


def _restore_real_klotski_state() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    leg = str(root / "legacy")
    if leg not in sys.path:
        sys.path.insert(0, leg)
    for name in ("klotski_state", "src.core.state_adapter", "src.core.moves_adapter"):
        sys.modules.pop(name, None)
    importlib.import_module("klotski_state")
    importlib.import_module("src.core.state_adapter")
    importlib.import_module("src.core.moves_adapter")


def test_build_r1_cases_schema():
    _install_legacy_state_stub()
    try:
        builder = importlib.import_module("src.tasks.r1_builder")

        exact_depth_cases = [
            {
                "case_id": "depth1_0",
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
                "shortest_solution_states": [
                    {
                        "blocks": {"A": {"shape": [1, 1], "pos": [0, 0]}},
                        "grid_size": [5, 4],
                    },
                    {
                        "blocks": {"A": {"shape": [1, 1], "pos": [0, 1]}},
                        "grid_size": [5, 4],
                    },
                ],
                "meta": {"canonical_state_key": (("A", 0),)},
            }
        ]

        cases = builder.build_r1_cases(exact_depth_cases)
        assert len(cases) == 1
        case = cases[0]
        assert case["task"] == "r1"
        assert "initial_state" in case["input"]
        assert case["gold"]["optimal_depth"] == 1
        assert "canonical_state_key" in case["meta"]
        assert "shortest_solution_actions" in case["meta"]
        assert "shortest_solution_states" in case["meta"]
    finally:
        _restore_real_klotski_state()
