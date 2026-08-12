import inspect

import numpy as np
import pytest

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.protocol import load_protocol
from defect_classifier.rta_adalora import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    FOCAL_GAMMA,
    INIT_R,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    PEFT_METHOD,
    PEFT_VERSION,
    TARGET_MODULES,
    balanced_mean_one_weights,
)
from defect_classifier.rta_adalora_multitask import (
    B4_SUBSET_SHA256,
    HEAD_DIMENSIONS,
    TASK_LOSS_WEIGHT,
    equal_joint_loss,
    future_shared_adapter_folds,
    hierarchical_targets,
    load_multitask_config,
    run_multitask_feasibility,
    validate_multitask_config,
)
from defect_classifier.rta_fusion import MAX_LENGTH, MODEL_ID, MODEL_REVISION


def test_exact_multitask_adalora_method():
    config, _ = load_multitask_config()
    assert MODEL_ID == "Colorful/RTA"
    assert MODEL_REVISION == "56cc614ee7cf17b8c1875e6848037f9e5bafc41a"
    assert PEFT_METHOD == "ADALORA" and PEFT_VERSION == "0.19.1"
    assert TARGET_MODULES == ("query", "value")
    assert (INIT_R, config["target_r"], LORA_ALPHA, LORA_DROPOUT) == (12, 8, 16, 0.05)
    assert MAX_LENGTH == 512 and EPOCHS == 3 and LEARNING_RATE == 1e-4
    assert FOCAL_GAMMA == 2.0 and EFFECTIVE_BATCH_SIZE == 32
    assert config["early_stopping"] is False


def test_three_heads_and_equal_joint_loss():
    torch = pytest.importorskip("torch")
    assert HEAD_DIMENSIONS == {"S6": 6, "S3": 3, "S2": 2}
    assert TASK_LOSS_WEIGHT == 1 / 3
    losses = [torch.tensor(3.0), torch.tensor(6.0), torch.tensor(9.0)]
    assert equal_joint_loss(*losses).item() == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("s6", "s3", "s2"),
    (
        ("blocker", "HIGH", "HIGH_IMPACT"),
        ("critical", "HIGH", "HIGH_IMPACT"),
        ("major", "HIGH", "HIGH_IMPACT"),
        ("normal", "MEDIUM", "LOWER_IMPACT"),
        ("minor", "LOW", "LOWER_IMPACT"),
        ("trivial", "LOW", "LOWER_IMPACT"),
    ),
)
def test_frozen_hierarchical_mapping(s6, s3, s2):
    assert hierarchical_targets(s6, load_protocol()) == (s6, s3, s2)


def test_train_only_class_weights_for_all_heads_have_mean_one():
    samples = {
        "S6": ["a", "a", "b", "c", "d", "e", "f"],
        "S3": ["high", "high", "medium", "low"],
        "S2": ["high", "low", "low"],
    }
    for values in samples.values():
        weights = balanced_mean_one_weights(values, tuple(dict.fromkeys(values)))
        assert np.mean(weights) == pytest.approx(1.0)


def test_future_design_is_three_shared_adapters_and_prespecified_outputs():
    config, _ = load_multitask_config()
    assert future_shared_adapter_folds() == (1, 2, 3)
    assert config["future_matrix"]["shared_adapter_count"] == 3
    assert config["future_direct_outputs"] == ["S6", "S3", "S2"]
    assert config["future_fusion_outputs"] == ["S6", "S3", "S2"]
    assert config["shared_embedding_extractions_per_fold"] == 1
    assert config["semantic_weight"] == 1.0
    assert config["feasibility"]["b4_subset_sha256"] == B4_SUBSET_SHA256


def test_b15_lexical_definitions_unchanged():
    config, _ = load_multitask_config()
    assert config["lexical"] == {
        "S6": {
            "representation_id": "S6-R3",
            "representation": "CHAR",
            "classifier": "LINEARSVC",
            "class_weight": "BALANCED",
            "c": 0.25,
        },
        "S3": {
            "representation_id": "S3-R5",
            "representation": "WORD",
            "classifier": "LOGREG",
            "class_weight": "BALANCED",
            "c": 2.0,
        },
        "S2": {
            "representation_id": "S2-R3",
            "representation": "WORD_CHAR",
            "classifier": "LOGREG",
            "class_weight": "BALANCED",
            "c": 2.0,
        },
    }


@pytest.mark.parametrize(
    ("key", "value"),
    (("peft_method", "LORA"), ("task_loss_weights", [1, 0, 0]), ("epochs", 2)),
)
def test_multitask_method_drift_fails_closed(key, value):
    config, _ = load_multitask_config()
    config[key] = value
    with pytest.raises(BenchmarkError, match="configuration drift"):
        validate_multitask_config(config)


def test_model_injection_saves_three_heads_and_freezes_base():
    from defect_classifier import rta_adalora_multitask as module

    source = inspect.getsource(module._build_multitask_model)
    assert "RobertaClassificationHead" in source
    assert 'modules_to_save=["classifier"]' in source
    assert "parameter.requires_grad_(False)" in source
    assert "attention.self" in source
    assert "for task in TASKS" in source


def test_engineering_run_has_no_competitive_or_locked_test_path():
    from defect_classifier import rta_adalora_multitask as module

    source = inspect.getsource(run_multitask_feasibility)
    assert 'stage != "feasibility"' in source
    assert '"competitive_metrics": None' in source
    assert "classification_metrics" not in source
    assert "data/locked" not in source
    assert "del validation_ids" in source
    training = inspect.getsource(module._train)
    assert "update_and_allocate(optimizer_step)" in training
    inference = inspect.getsource(module._inference)
    assert "output.hidden_states[-1][:, 0, :]" in inference
    assert "output.logits" in inference
