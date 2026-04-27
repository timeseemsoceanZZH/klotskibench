"""
Install legacy ``klotski_state`` / ``klotski_moves`` stubs as soon as this
conftest loads, before any test module imports ``src`` (adapters must bind
to the stubbed modules).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_LEGACY = _ROOT / "legacy"
if os.environ.get("KLOTSKIBENCH_LIVE_ENGINE") == "1":
    if str(_LEGACY) not in sys.path:
        sys.path.insert(0, str(_LEGACY))


def _install_legacy_stubs() -> None:
    if os.environ.get("KLOTSKIBENCH_LIVE_ENGINE") == "1":
        return
    legacy_state = types.ModuleType("klotski_state")
    legacy_state.CanonicalKey = tuple
    legacy_state.KlotskiState = dict
    legacy_state.normalize_state = lambda s: s
    legacy_state.canonicalize_state = lambda s: (s["blocks"]["A"]["pos"][0], s["blocks"]["A"]["pos"][1])
    legacy_state.validate_state = (
        lambda s: 0 <= s["blocks"]["A"]["pos"][0] <= 4 and 0 <= s["blocks"]["A"]["pos"][1] <= 3
    )
    legacy_state.is_goal_state = lambda s: False
    legacy_state.to_flat_state = lambda s: dict(s)
    sys.modules["klotski_state"] = legacy_state

    legacy_moves = types.ModuleType("klotski_moves")
    legacy_moves.ActionDict = dict
    legacy_moves.build_occupied_by_id = lambda s: {}

    _AT_ORIGIN: list[dict[str, str]] = [
        {"block_id": "A", "direction": "right"},
    ]

    def get_legal_actions(s: dict) -> list[dict[str, str]]:
        if s.get("blocks", {}).get("A", {}).get("pos") == [0, 0]:
            return list(_AT_ORIGIN)
        return []

    def apply_action(state: dict, action: dict[str, str]) -> dict:
        out = {
            "blocks": {
                "A": {
                    "shape": [1, 1],
                    "pos": list(state["blocks"]["A"]["pos"]),
                }
            },
            "grid_size": [5, 4],
        }
        if action.get("block_id") != "A":
            return out
        d = action.get("direction", "")
        if d == "right":
            out["blocks"]["A"]["pos"][1] += 1
        elif d == "left":
            out["blocks"]["A"]["pos"][1] -= 1
        elif d == "up":
            out["blocks"]["A"]["pos"][0] -= 1
        elif d == "down":
            out["blocks"]["A"]["pos"][0] += 1
        return out

    def enumerate_predecessor_states(s: dict) -> list:  # noqa: ARG001
        return []

    legacy_moves.get_legal_actions = get_legal_actions
    legacy_moves.apply_action = apply_action
    legacy_moves.enumerate_predecessor_states = enumerate_predecessor_states
    sys.modules["klotski_moves"] = legacy_moves

    for name in (
        "src.core.moves_adapter",
        "src.core.state_adapter",
    ):
        mod = sys.modules.get(name)
        if mod is not None:
            importlib.reload(mod)


_install_legacy_stubs()

# Bind adapters while stubs are in `sys.modules` (before any test runs).
import src.core.moves_adapter  # noqa: E402, F401
import src.core.state_adapter  # noqa: E402, F401
