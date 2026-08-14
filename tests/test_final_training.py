from __future__ import annotations

import inspect
from pathlib import Path

from defect_classifier.cli import build_parser
from defect_classifier.final_training import (
    EXPECTED_DEVELOPMENT_ROWS,
    EXPECTED_DEVELOPMENT_SHA256,
    FINAL_TRAINING_TAG,
    SELECTED_FAMILY,
    run_final_training,
)


def test_final_training_is_full_development_and_single_selected_family():
    assert EXPECTED_DEVELOPMENT_ROWS == 207575
    assert (
        EXPECTED_DEVELOPMENT_SHA256
        == "4f62fdf4164594126c421955804b654cd5d5f8f7b46ada345ae0cffa71460d0f"
    )
    assert SELECTED_FAMILY == "B4H_ADAPTED_RTA_LEXICAL_FUSION"
    assert FINAL_TRAINING_TAG == "final-training-runner-v1"


def test_final_training_source_has_no_locked_partition_read_or_unlock():
    source = inspect.getsource(run_final_training)
    signature = inspect.signature(run_final_training)
    assert "locked_dir" not in signature.parameters
    assert "require_locked_test_unlock" not in source
    assert "data/locked" not in source
    assert "FINAL_EVALUATION_ONLY" not in source
    assert "locked_test_model_performance_accessed" in source


def test_final_training_cli_has_no_locked_path_argument():
    args = build_parser().parse_args(["train-final-model"])
    assert args.development_dir == Path("data/processed/protocol_v1/development")
    assert args.freeze_file == Path("reports/model_selection_v1/model_selection_freeze.json")
    assert args.artifact_root == Path("data/processed/final_model_v1")
    assert not hasattr(args, "locked_dir")
