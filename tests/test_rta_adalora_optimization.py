import inspect

import pytest

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.rta_adalora_optimization import (
    ACCUMULATION,
    EFFECTIVE_BATCH_SIZE,
    MICRO_BATCH,
    PAD_MULTIPLE,
    deterministic_length_batches,
    dynamic_pad_batch,
    load_optimization_config,
    reject_unstable_bf16,
    run_optimization_feasibility,
    validate_optimization_config,
)
from defect_classifier.rta_fusion import MAX_LENGTH, MODEL_ID, MODEL_REVISION


def test_deterministic_length_bucketing_complete_unique_and_label_free():
    ids = [f"P:{index}" for index in range(23)]
    lengths = [((index * 17) % 50) + 1 for index in range(23)]
    first = deterministic_length_batches(ids, lengths, 4, 20260809, 1)
    second = deterministic_length_batches(ids, lengths, 4, 20260809, 1)
    assert first == second
    flattened = [index for batch in first for index in batch]
    assert sorted(flattened) == list(range(23))
    assert len(flattened) == len(set(flattened))
    source = inspect.getsource(deterministic_length_batches)
    assert "label" not in source.casefold()


def test_dynamic_padding_preserves_content_and_hard_max():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Colorful/RTA",
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    encoded = tokenizer(["a", "a longer bug report"], truncation=True, max_length=MAX_LENGTH)
    padded = dynamic_pad_batch(tokenizer, encoded, [0, 1])
    for index in (0, 1):
        length = len(encoded["input_ids"][index])
        assert padded["input_ids"][index, :length].tolist() == encoded["input_ids"][index]
    assert padded["input_ids"].shape[1] % PAD_MULTIPLE == 0
    assert padded["input_ids"].shape[1] <= MAX_LENGTH


def test_scientific_configuration_is_unchanged():
    config, _ = load_optimization_config()
    assert config["model_id"] == MODEL_ID
    assert config["resolved_revision"] == MODEL_REVISION
    assert config["max_length"] == 512 and config["future_epochs"] == 3
    assert config["effective_batch_size"] == 32
    assert MICRO_BATCH * ACCUMULATION == EFFECTIVE_BATCH_SIZE
    assert config["candidate_dtypes"] == ["float32", "bfloat16"]
    assert config["float16_allowed"] is False


def test_bfloat16_fails_closed_for_any_instability():
    reject_unstable_bf16({"loss_finite": True, "gradients_finite": True})
    with pytest.raises(BenchmarkError, match="failed closed"):
        reject_unstable_bf16({"loss_finite": True, "gradients_finite": False})


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("max_length", 256),
        ("future_epochs", 1),
        ("focal_gamma", 1.0),
        ("candidate_dtypes", ["float16"]),
    ),
)
def test_optimization_or_scientific_drift_fails_closed(key, value):
    config, _ = load_optimization_config()
    config[key] = value
    with pytest.raises(BenchmarkError, match="configuration drift"):
        validate_optimization_config(config)


def test_no_validation_metrics_locked_test_or_competitive_path():
    source = inspect.getsource(run_optimization_feasibility)
    assert 'stage != "feasibility"' in source
    assert '"competitive_metrics": None' in source
    assert "classification_metrics" not in source
    assert "data/locked" not in source
    assert "del validation_ids" in source
    assert "competitive_models_fitted" in source


def test_candidate_measurements_are_checkpointed_before_final_reporting():
    source = inspect.getsource(run_optimization_feasibility)
    assert "_candidate_checkpoint(work, o1)" in source
    assert "_candidate_checkpoint(work, o2)" in source
    assert 'work / "bfloat16_preflight.json"' in source
    assert 'work / "token_padding_audit.json"' in source
