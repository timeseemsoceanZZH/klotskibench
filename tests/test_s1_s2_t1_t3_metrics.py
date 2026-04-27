from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_s1_metrics_accuracy():
    from src.metrics.s1_metrics import compute_s1_metrics

    cases = [
        {"gold": {"label": "valid"}},
        {"gold": {"label": "invalid"}},
    ]
    preds = [{"label": "valid"}, {"label": "invalid"}]
    m = compute_s1_metrics(cases, preds)
    assert m["accuracy"] == 1.0
    assert m["precision_invalid"] == 1.0
    assert m["recall_invalid"] == 1.0


def test_s2_metrics_exact_error():
    from src.metrics.s2_metrics import compute_s2_metrics

    err = [{"block_id": "A", "error_type": "overlap"}]
    cases = [{"gold": {"errors": err}}]
    preds = [{"errors": err}]
    m = compute_s2_metrics(cases, preds)
    assert m["exact_error_match"] == 1.0


def test_t1_metrics_set():
    from src.metrics.t1_metrics import compute_t1_metrics

    gold_acts = [
        {"block_id": "A", "direction": "up"},
        {"block_id": "A", "direction": "down"},
    ]
    cases = [{"gold": {"legal_actions": gold_acts}}]
    preds = [{"legal_actions": [gold_acts[1], gold_acts[0]]}]
    m = compute_t1_metrics(cases, preds)
    assert m["exact_set_match"] == 1.0
    assert m["move_precision"] == 1.0
    assert m["move_recall"] == 1.0


def test_t3_label_and_reason():
    from src.metrics.t3_metrics import compute_t3_metrics

    cases = [
        {"gold": {"label": "valid"}},
        {"gold": {"label": "invalid", "reason": "overlap"}},
    ]
    preds = [
        {"label": "valid"},
        {"label": "invalid", "reason": "overlap"},
    ]
    m = compute_t3_metrics(cases, preds)
    assert m["label_accuracy"] == 1.0
    assert m["invalid_reason_accuracy"] == 1.0
    assert m["joint_verification_reason_accuracy"] == 1.0
