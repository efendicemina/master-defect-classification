from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from defect_classifier.cli import build_parser
from defect_classifier.final_evaluation import (
    EXPECTED_ARTIFACTS,
    EXPECTED_LOCKED_ROWS,
    EXPECTED_LOCKED_SHA256,
    FINAL_EVALUATION_RUNNER_TAG,
    FINAL_MODEL_COMMIT,
    FINAL_MODEL_TAG,
    _prediction_fingerprint,
    _s2_score_metrics,
    run_final_evaluation,
)


def test_final_evaluation_constants_freeze_one_locked_evaluation():
    assert FINAL_MODEL_TAG == "final-model-v1"
    assert FINAL_MODEL_COMMIT == "0f124923ac23234c6e9953f4434f9ad9b90584af"
    assert FINAL_EVALUATION_RUNNER_TAG == "final-evaluation-runner-v1"
    assert EXPECTED_LOCKED_ROWS == 50675
    assert (
        EXPECTED_LOCKED_SHA256 == "c18f2320c896bf2bdcb94d2447ce9a04d17e6923f03b169f29b8174b8a5cf681"
    )
    assert len(EXPECTED_ARTIFACTS) == 11


def test_final_evaluation_never_fits_or_scores_direct_heads():
    source = inspect.getsource(run_final_evaluation)
    assert ".fit(" not in source
    assert "require_locked_test_unlock(protocol)" in source
    assert '"direct_head_metrics_calculated": False' in source
    assert '"direct_head_predictions_read": False' in source


def test_final_evaluation_cli_defaults_to_safe_preflight():
    args = build_parser().parse_args(["run-final-evaluation"])
    assert args.stage == "preflight"
    assert args.locked_dir == Path("data/locked/protocol_v1")
    assert args.artifact_root == Path("data/processed/final_model_v1")
    assert args.final_manifest == Path("reports/final_training_v1/final_training_manifest.json")
    assert args.report_dir == Path("reports/final_evaluation_v1")


def test_prediction_fingerprint_is_deterministic_and_alignment_sensitive():
    stable_ids = ["A:1", "B:2"]
    predictions = ["normal", "major"]
    first = _prediction_fingerprint(stable_ids, predictions)
    second = _prediction_fingerprint(stable_ids, predictions)
    assert first == second
    assert first != _prediction_fingerprint(stable_ids, ["major", "normal"])


def test_s2_score_metrics_are_pr_auc_and_roc_auc():
    labels = ["HIGH_IMPACT", "LOWER_IMPACT", "HIGH_IMPACT", "LOWER_IMPACT"]
    scores = np.asarray([0.9, 0.2, 0.8, 0.1], dtype=np.float64)
    metrics = _s2_score_metrics(labels, scores)
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
