import inspect
import json
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.rta_adalora import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    FOCAL_GAMMA,
    INIT_R,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    MODULES_TO_SAVE,
    PEFT_METHOD,
    PEFT_VERSION,
    TARGET_MODULES,
    TARGET_R,
    adalora_schedule,
    balanced_mean_one_weights,
    deterministic_stratified_subset,
    future_adapter_matrix,
    json_safe,
    load_adalora_config,
    run_adalora_feasibility,
    validate_adalora_config,
    weighted_focal_cross_entropy,
)
from defect_classifier.rta_fusion import MAX_LENGTH, MODEL_ID, MODEL_REVISION


def test_exact_rta_adalora_method_and_training_policy():
    config, _ = load_adalora_config()
    assert MODEL_ID == "Colorful/RTA"
    assert MODEL_REVISION == "56cc614ee7cf17b8c1875e6848037f9e5bafc41a"
    assert PEFT_METHOD == "ADALORA" and PEFT_VERSION == "0.19.1"
    assert (INIT_R, TARGET_R, LORA_ALPHA, LORA_DROPOUT) == (12, 8, 16, 0.05)
    assert TARGET_MODULES == ("query", "value")
    assert MODULES_TO_SAVE == ("classifier",)
    assert EPOCHS == 3 and LEARNING_RATE == 1e-4 and FOCAL_GAMMA == 2.0
    assert MAX_LENGTH == 512 and EFFECTIVE_BATCH_SIZE == 32
    assert config["early_stopping"] is False


def test_schedule_uses_frozen_fractions_and_about_100_updates():
    schedule = adalora_schedule(939)
    assert schedule["tinit"] == round(0.10 * 939)
    assert schedule["tfinal"] == round(0.20 * 939)
    assert schedule["deltaT"] == round((939 - 94 - 188) / 100)
    assert adalora_schedule(3)["deltaT"] == 1


def test_train_only_weights_are_balanced_and_mean_one():
    values = balanced_mean_one_weights(["LOW", "LOW", "LOW", "HIGH"], ("HIGH", "LOW"))
    assert np.mean(values) == pytest.approx(1.0)
    assert values[0] > values[1]
    with pytest.raises(BenchmarkError, match="every task class"):
        balanced_mean_one_weights(["LOW"], ("HIGH", "LOW"))


def test_weighted_focal_loss_is_finite_and_gamma_fixed():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.5, 0.5])
    loss = weighted_focal_cross_entropy(logits, labels, weights)
    assert torch.isfinite(loss) and loss.item() > 0


def test_engineering_subset_is_deterministic_and_distribution_preserving():
    ids = [f"P:{index}" for index in range(100)]
    labels = {stable_id: ("HIGH" if index < 20 else "LOW") for index, stable_id in enumerate(ids)}
    first = deterministic_stratified_subset(ids, labels, 50, 20260809)
    assert first == deterministic_stratified_subset(ids, labels, 50, 20260809)
    assert sum(labels[value] == "HIGH" for value in first) == 10
    assert sum(labels[value] == "LOW" for value in first) == 40


def test_future_matrix_and_outputs_are_pre_specified():
    config, _ = load_adalora_config()
    assert len(future_adapter_matrix()) == 9
    assert len(set(future_adapter_matrix())) == 9
    assert config["future_outputs"] == ["DIRECT_CLASSIFIER", "LEXICAL_FUSION"]
    assert config["semantic_weight"] == 1.0
    assert config["lexical"]["S6"] == {
        "representation_id": "S6-R3",
        "representation": "CHAR",
        "classifier": "LINEARSVC",
        "class_weight": "BALANCED",
        "c": 0.25,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    (("peft_method", "LORA"), ("epochs", 4), ("focal_gamma", 1.0), ("semantic_weight", 0.5)),
)
def test_method_drift_fails_closed(key, value):
    config, _ = load_adalora_config()
    config[key] = value
    with pytest.raises(BenchmarkError, match="configuration drift"):
        validate_adalora_config(config)


def test_real_training_path_updates_adalora_and_never_evaluates_performance():
    from defect_classifier import rta_adalora

    training = inspect.getsource(rta_adalora._train_engineering)
    assert "update_and_allocate(optimizer_step)" in training
    assert "clip_grad_norm_" in training
    assert "get_linear_schedule_with_warmup" in training
    source = inspect.getsource(run_adalora_feasibility)
    assert 'stage != "feasibility"' in source
    assert "competitive_metrics_calculated" in source
    assert '"competitive_metrics": None' in source
    assert "_load_fold(" in source and "del _validation_ids" in source
    assert "data/locked" not in source
    assert "classification_metrics" not in source


def test_peft_injection_freezes_base_and_saves_classifier_by_construction():
    from defect_classifier import rta_adalora

    source = inspect.getsource(rta_adalora._load_adalora_model)
    assert "AdaLoraConfig(" in source
    assert "modules_to_save=list(MODULES_TO_SAVE)" in source
    assert "parameter.requires_grad_(False)" in source
    assert "attention.self" in source
    assert "trainable-parameter boundary drift" in source


def test_json_safe_recursively_serializes_report_value_types():
    torch = pytest.importorskip("torch")

    class Example(Enum):
        VALUE = "value"

    value = {
        "scalar_tensor": torch.tensor(True),
        "vector_tensor": torch.tensor([1.0, 2.0]),
        "numpy_scalar": np.int64(3),
        "numpy_array": np.array([4, 5]),
        "path": Path("report.json"),
        "enum": Example.VALUE,
    }
    converted = json_safe(value)
    assert converted == {
        "scalar_tensor": True,
        "vector_tensor": [1.0, 2.0],
        "numpy_scalar": 3,
        "numpy_array": [4, 5],
        "path": "report.json",
        "enum": "value",
    }
    json.dumps(converted)


def test_json_safe_rejects_large_tensor_reports():
    torch = pytest.importorskip("torch")
    with pytest.raises(BenchmarkError, match="large tensor"):
        json_safe({"weights": torch.zeros(100_001)})
