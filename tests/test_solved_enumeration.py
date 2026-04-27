"""Solved layout enumeration and multi-goal goal-seed behavior.

These tests require the real :mod:`legacy` engine. Run (from repo root):

  KLOTSKIBENCH_LIVE_ENGINE=1 python3 -m pytest tests/test_solved_enumeration.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

if os.environ.get("KLOTSKIBENCH_LIVE_ENGINE") != "1":
    pytest.skip(
        "Set KLOTSKIBENCH_LIVE_ENGINE=1 in the *same* process (e.g. "
        "KLOTSKIBENCH_LIVE_ENGINE=1 python3 -m pytest tests/test_solved_enumeration.py) "
        "or tests/conftest will install stubs and the legacy engine is required here.",
        allow_module_level=True,
    )

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

import _bootstrap_paths  # noqa: E402

_bootstrap_paths.ensure_repo_sys_path()

from src.core import state_adapter  # noqa: E402
from src.generators import goal_seed_enumerator  # noqa: E402
from src.generators.solved_layout_enumeration import (  # noqa: E402
    enumerate_solved_huarong_states,
)


def test_is_goal_is_cao_position_not_full_board() -> None:
    ref = state_adapter.normalize_state(
        goal_seed_enumerator.DEFAULT_GOAL_SEED_CANDIDATES[0]
    )
    assert state_adapter.is_goal_state(ref) is True
    layout = list(enumerate_solved_huarong_states())
    assert len(layout) >= 2
    rk = state_adapter.canonicalize_state(ref)
    other = next(
        s for s in layout if state_adapter.canonicalize_state(s) != rk
    )
    assert state_adapter.is_goal_state(other) is True
    assert state_adapter.canonicalize_state(other) != state_adapter.canonicalize_state(
        ref
    )


def test_exhaustive_solved_count() -> None:
    assert len(enumerate_solved_huarong_states()) == 107_712


def test_enumerate_goal_seeds_capped() -> None:
    seeds = goal_seed_enumerator.enumerate_goal_seed_states(
        max_goals=50, require_predecessor_space=True
    )
    assert 1 <= len(seeds) <= 50
    for g in seeds:
        st = state_adapter.normalize_state(g)
        assert state_adapter.is_goal_state(st) is True
