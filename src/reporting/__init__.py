"""Lightweight reporting helpers: tables and trajectory-depth breakdowns."""

from __future__ import annotations

from src.reporting.reports import (
    build_conditioned_metrics_table,
    build_depth_wise_trajectory_report,
    build_main_task_table,
    build_oracle_state_table,
)

__all__ = [
    "build_main_task_table",
    "build_conditioned_metrics_table",
    "build_oracle_state_table",
    "build_depth_wise_trajectory_report",
]
