#!/usr/bin/env python3
"""
Export fixed one-shot exemplars for the seven Klotski-Bench tasks.

Each exemplar is a handcrafted case with task input, the expected model output
(exemplar_output), and a short description.  They are intentionally independent
of the depth-stratified evaluation set.

Usage:
  python scripts/export_one_shot_exemplars.py \
      --out-dir output/seven_task_one_shot_exemplars
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Base state shared by S1, S2, T1, T2, T3 exemplars
# Grid 5×4; Caocao at [0,1]; free cells: [3,1],[3,2],[4,1],[4,2]
# ---------------------------------------------------------------------------
_BASE_STATE: dict[str, Any] = {
    "grid_size": [5, 4],
    "blocks": {
        "Caocao": {"shape": [2, 2], "pos": [0, 1]},
        "S1":     {"shape": [1, 1], "pos": [0, 0]},
        "S2":     {"shape": [1, 1], "pos": [0, 3]},
        "S3":     {"shape": [1, 1], "pos": [1, 0]},
        "S4":     {"shape": [1, 1], "pos": [1, 3]},
        "V1":     {"shape": [2, 1], "pos": [2, 0]},
        "H1":     {"shape": [1, 2], "pos": [2, 1]},
        "V2":     {"shape": [2, 1], "pos": [2, 3]},
        "S5":     {"shape": [1, 1], "pos": [4, 0]},
        "S6":     {"shape": [1, 1], "pos": [4, 3]},
    },
}

# After H1 (1×2 at [2,1]) slides down → new pos [3,1]
_T2_NEXT_STATE: dict[str, Any] = {
    "grid_size": [5, 4],
    "blocks": {
        "Caocao": {"shape": [2, 2], "pos": [0, 1]},
        "S1":     {"shape": [1, 1], "pos": [0, 0]},
        "S2":     {"shape": [1, 1], "pos": [0, 3]},
        "S3":     {"shape": [1, 1], "pos": [1, 0]},
        "S4":     {"shape": [1, 1], "pos": [1, 3]},
        "V1":     {"shape": [2, 1], "pos": [2, 0]},
        "H1":     {"shape": [1, 2], "pos": [3, 1]},  # moved
        "V2":     {"shape": [2, 1], "pos": [2, 3]},
        "S5":     {"shape": [1, 1], "pos": [4, 0]},
        "S6":     {"shape": [1, 1], "pos": [4, 3]},
    },
}

# T3 candidate: S5 (1×1 at [4,0]) slides right → new pos [4,1]
_T3_CANDIDATE_NEXT_STATE: dict[str, Any] = {
    "grid_size": [5, 4],
    "blocks": {
        "Caocao": {"shape": [2, 2], "pos": [0, 1]},
        "S1":     {"shape": [1, 1], "pos": [0, 0]},
        "S2":     {"shape": [1, 1], "pos": [0, 3]},
        "S3":     {"shape": [1, 1], "pos": [1, 0]},
        "S4":     {"shape": [1, 1], "pos": [1, 3]},
        "V1":     {"shape": [2, 1], "pos": [2, 0]},
        "H1":     {"shape": [1, 2], "pos": [2, 1]},
        "V2":     {"shape": [2, 1], "pos": [2, 3]},
        "S5":     {"shape": [1, 1], "pos": [4, 1]},  # moved
        "S6":     {"shape": [1, 1], "pos": [4, 3]},
    },
}

# Depth-1 state for R1/R2: Caocao at [2,1]; one move (Caocao→down) solves.
# Free cells: [4,1],[4,2]
_R1_INITIAL_STATE: dict[str, Any] = {
    "grid_size": [5, 4],
    "blocks": {
        "V1":     {"shape": [2, 1], "pos": [0, 0]},
        "H1":     {"shape": [1, 2], "pos": [0, 1]},
        "V2":     {"shape": [2, 1], "pos": [0, 3]},
        "H2":     {"shape": [1, 2], "pos": [1, 1]},
        "V3":     {"shape": [2, 1], "pos": [2, 0]},
        "Caocao": {"shape": [2, 2], "pos": [2, 1]},
        "V4":     {"shape": [2, 1], "pos": [2, 3]},
        "S1":     {"shape": [1, 1], "pos": [4, 0]},
        "S2":     {"shape": [1, 1], "pos": [4, 3]},
    },
}

# ---------------------------------------------------------------------------
# Fixed exemplars
# ---------------------------------------------------------------------------
EXEMPLARS: dict[str, dict[str, Any]] = {
    "s1": {
        "task": "s1",
        "input": {
            "state": _BASE_STATE,
        },
        "gold": {"label": "valid"},
        "exemplar_output": {"label": "valid"},
        "description": (
            "All 10 blocks are within the 5×4 board and non-overlapping: "
            "the state is valid."
        ),
    },
    "s2": {
        "task": "s2",
        "input": {
            "state": {
                "grid_size": [5, 4],
                "blocks": {
                    "Caocao": {"shape": [2, 2], "pos": [0, 1]},
                    "S1":     {"shape": [1, 1], "pos": [0, 0]},
                    "S2":     {"shape": [1, 1], "pos": [0, 3]},
                    "S3":     {"shape": [1, 1], "pos": [1, 0]},
                    "S4":     {"shape": [1, 1], "pos": [1, 3]},
                    "V1":     {"shape": [2, 1], "pos": [2, 0]},
                    "H1":     {"shape": [1, 2], "pos": [2, 1]},
                    "V2":     {"shape": [2, 1], "pos": [2, 3]},
                    "S5":     {"shape": [1, 1], "pos": [4, 0]},
                    "S6":     {"shape": [1, 1], "pos": [4, 4]},  # col 4 out of bounds
                },
            },
        },
        "gold": {
            "errors": [
                {
                    "error_type": "boundary",
                    "block_id": "S6",
                    "cells": [[4, 4]],
                }
            ]
        },
        "exemplar_output": {
            "errors": [
                {
                    "error_type": "boundary",
                    "block_id": "S6",
                    "cells": [[4, 4]],
                }
            ]
        },
        "description": (
            "Block S6 is placed at column 4, outside the board's 4 columns (0–3): "
            "boundary error."
        ),
    },
    "t1": {
        "task": "t1",
        "input": {
            "current_state": _BASE_STATE,
        },
        "gold": {
            "legal_actions": [
                {"block_id": "H1", "direction": "down"},
                {"block_id": "S5", "direction": "right"},
                {"block_id": "S6", "direction": "left"},
            ]
        },
        "exemplar_output": {
            "legal_actions": [
                {"block_id": "H1", "direction": "down"},
                {"block_id": "S5", "direction": "right"},
                {"block_id": "S6", "direction": "left"},
            ]
        },
        "description": (
            "From this state exactly three moves are legal: "
            "H1 slides down, S5 slides right, S6 slides left."
        ),
    },
    "t2": {
        "task": "t2",
        "input": {
            "current_state": _BASE_STATE,
            "action": {"block_id": "H1", "direction": "down"},
        },
        "gold": {"next_state": _T2_NEXT_STATE},
        "exemplar_output": {"next_state": _T2_NEXT_STATE},
        "description": (
            "Applying H1→down shifts H1 from row 2 to row 3; "
            "all other blocks remain unchanged."
        ),
    },
    "t3": {
        "task": "t3",
        "input": {
            "current_state": _BASE_STATE,
            "action": {"block_id": "S5", "direction": "right"},
            "candidate_next_state": _T3_CANDIDATE_NEXT_STATE,
        },
        "gold": {"label": "valid"},
        "exemplar_output": {"label": "valid"},
        "description": (
            "Action S5→right moves S5 from [4,0] to [4,1], "
            "which is in-bounds and unoccupied: valid transition."
        ),
    },
    "r1": {
        "task": "r1",
        "input": {
            "initial_state": _R1_INITIAL_STATE,
        },
        "gold": {"optimal_depth": 1},
        "exemplar_output": {
            "predicted_trajectory": [
                {"block_id": "Caocao", "direction": "down"},
            ]
        },
        "description": (
            "Caocao is one move from the goal: "
            "slide Caocao down to reach position [3,1]."
        ),
    },
    "r2": {
        "task": "r2",
        "input": {
            "initial_state": _R1_INITIAL_STATE,
            "candidate_trajectory": [
                {"block_id": "Caocao", "direction": "down"},
            ],
        },
        "gold": {"label": "valid"},
        "exemplar_output": {"label": "valid"},
        "description": (
            "The trajectory [Caocao→down] is valid: "
            "one legal move that reaches the solved state."
        ),
    },
}


def export_exemplars(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, exemplar in EXEMPLARS.items():
        task_dir = out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / "case.json"
        path.write_text(
            json.dumps(exemplar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  {task} -> {path}")
    print(f"Exported {len(EXEMPLARS)} exemplars to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fixed one-shot exemplars for the seven Klotski-Bench tasks."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root directory (e.g. output/seven_task_one_shot_exemplars).",
    )
    args = parser.parse_args()
    export_exemplars(args.out_dir)


if __name__ == "__main__":
    main()
