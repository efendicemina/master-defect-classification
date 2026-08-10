import hashlib
import inspect
from pathlib import Path

import pytest

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.transformer_finetuning import (
    MODEL_ID,
    REFERENCE_HASHES,
    balanced_class_weights,
    load_finetuning_config,
    run_id,
    search_space_size,
    validate_finetuning_config,
)


def test_exact_fixed_nine_fit_matrix():
    config, _ = load_finetuning_config()
    assert search_space_size(config) == 9
    assert config["model_id"] == MODEL_ID == "microsoft/deberta-v3-small"
    assert config["max_length"] == 256
    assert config["learning_rate"] == 4.5e-5
    assert config["num_train_epochs"] == config["final_epoch_selection"] == 3
    assert config["per_device_train_batch_size"] == 8
    assert config["per_device_eval_batch_size"] == 16
    assert config["early_stopping"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("model_id", "other/model"),
        ("max_length", 512),
        ("learning_rate", 1e-5),
        ("num_train_epochs", 2),
        ("per_device_train_batch_size", 4),
        ("early_stopping", True),
    ),
)
def test_method_drift_fails_closed(key, value):
    config, _ = load_finetuning_config()
    config[key] = value
    with pytest.raises(BenchmarkError, match="method drift"):
        validate_finetuning_config(config)


def test_balanced_weights_use_only_supplied_train_labels():
    labels = ["A", "A", "A", "B"]
    assert balanced_class_weights(labels, ("A", "B")) == [4 / 6, 2.0]
    with pytest.raises(BenchmarkError, match="TRAIN labels"):
        balanced_class_weights(["A", "A"], ("A", "B"))


def test_deterministic_run_ids_and_checkpoint_resume_path():
    assert run_id("S6", 1, "revision", "config") == run_id("S6", 1, "revision", "config")
    from defect_classifier import transformer_finetuning

    source = inspect.getsource(transformer_finetuning._run_competitive)
    assert "metrics_path.is_file()" in source
    assert "checkpoint provenance drift" in source
    assert "final_epoch" in source


def test_frozen_cv_loader_and_no_locked_artifact_or_fold_reconstruction():
    from defect_classifier import transformer_finetuning

    source = inspect.getsource(transformer_finetuning.run_b2_pipeline)
    assert "_load_fold" in source
    assert "_temporal_folds" not in source
    assert "data/locked" not in source
    assert "locked_test_tokenized" in source


def test_previous_report_fingerprints_unchanged():
    root = Path(__file__).resolve().parents[1] / "reports"
    for _, (relative, expected) in REFERENCE_HASHES.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
