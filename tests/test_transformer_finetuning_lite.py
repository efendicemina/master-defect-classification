import hashlib
import inspect
from pathlib import Path

import pytest

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.transformer_finetuning_lite import (
    MODEL_ID,
    MODEL_REVISION,
    REFERENCE_HASHES,
    lite_run_id,
    load_lite_config,
    search_space_size,
    validate_lite_config,
)


def test_fixed_minilm_method_and_nine_fit_matrix():
    config, _ = load_lite_config()
    assert MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"
    assert config["resolved_revision"] == MODEL_REVISION
    assert search_space_size(config) == 9
    assert config["max_length"] == 256
    assert config["learning_rate"] == 2e-5
    assert config["num_train_epochs"] == config["final_epoch_selection"] == 3
    assert config["per_device_train_batch_size"] == 16
    assert config["per_device_eval_batch_size"] == 32
    assert config["early_stopping"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("model_id", "other"),
        ("resolved_revision", "other"),
        ("max_length", 512),
        ("learning_rate", 3e-5),
        ("num_train_epochs", 2),
        ("per_device_train_batch_size", 8),
        ("early_stopping", True),
    ),
)
def test_lite_method_drift_fails_closed(key, value):
    config, _ = load_lite_config()
    config[key] = value
    with pytest.raises(BenchmarkError, match="method drift"):
        validate_lite_config(config)


def test_train_only_balanced_weights_are_reused():
    from defect_classifier.transformer_finetuning import balanced_class_weights

    assert balanced_class_weights(["A", "A", "A", "B"], ("A", "B")) == [4 / 6, 2.0]


def test_deterministic_lite_ids_and_checkpoint_resume():
    assert lite_run_id("S2", 1, MODEL_REVISION, "x") == lite_run_id("S2", 1, MODEL_REVISION, "x")
    from defect_classifier import transformer_finetuning_lite

    source = inspect.getsource(transformer_finetuning_lite._run_competitive_lite)
    assert "result_path.is_file()" in source
    assert "checkpoint provenance drift" in source
    assert '"final_epoch": 3' in source


def test_frozen_cv_disjoint_loader_no_locked_or_reconstruction():
    from defect_classifier import transformer_finetuning_lite

    source = inspect.getsource(transformer_finetuning_lite.run_lite_pipeline)
    assert "_load_fold" in source
    assert "_temporal_folds" not in source
    assert "data/locked" not in source
    assert "locked_test_tokenized" in source


def test_previous_report_fingerprints_unchanged():
    root = Path(__file__).resolve().parents[1] / "reports"
    for _, (relative, expected) in REFERENCE_HASHES.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected


def test_checkpoint_finalization_cannot_fit_or_load_development_data():
    from defect_classifier import transformer_finetuning_lite

    source = inspect.getsource(transformer_finetuning_lite.finalize_lite_checkpoints)
    assert "_train(" not in source
    assert "_load_model(" not in source
    assert "_load_development_rows(" not in source
    assert "_load_fold(" not in source
    assert "locked" in source
    assert 'stored.get("status") == "SUCCESS"' in source
