import inspect

import numpy as np
import pytest

from defect_classifier.embedding_cache import (
    EmbeddingCacheError,
    provenance_fingerprint,
    stable_id,
    validate_embeddings,
    validate_membership,
    validate_provenance,
)
from defect_classifier.semantic_benchmark import (
    aggregate_semantic_results,
    semantic_experiment_id,
)
from defect_classifier.semantic_embeddings import (
    load_semantic_config,
    search_space_size,
    validate_semantic_config,
)


def test_exact_search_space_and_fixed_method():
    config, _ = load_semantic_config()
    assert search_space_size(config) == 72
    assert config["c"] == 1.0
    assert [row["id"] for row in config["encoders"]] == ["E1", "E2"]
    config["c"] = 2.0
    with pytest.raises(EmbeddingCacheError, match="method drift"):
        validate_semantic_config(config)


def test_cache_membership_rejects_nondevelopment_and_locked_rows():
    development = {"P:1", "P:2"}
    validate_membership(set(development), development, {"P:3"})
    with pytest.raises(EmbeddingCacheError, match="exact frozen DEVELOPMENT"):
        validate_membership({"P:1", "P:3"}, development)
    with pytest.raises(EmbeddingCacheError, match="locked"):
        validate_membership(development, development, {"P:2"})
    assert stable_id("P", "1") == "P:1"


def test_cache_provenance_invalidates_protocol_revision_and_text_policy():
    base = {"protocol": "one", "revision": "abc", "text": {"separator": "\n\n"}}
    assert provenance_fingerprint(base) == provenance_fingerprint(dict(base))
    for changed in (
        {**base, "protocol": "two"},
        {**base, "revision": "def"},
        {**base, "text": {"separator": " "}},
    ):
        with pytest.raises(EmbeddingCacheError, match="provenance"):
            validate_provenance(base, changed)


def test_embedding_validation_normalization_dimension_and_finiteness():
    values = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    validate_embeddings(values, 2)
    with pytest.raises(EmbeddingCacheError, match="dimension"):
        validate_embeddings(values, 3)
    values[0, 0] = np.nan
    with pytest.raises(EmbeddingCacheError, match="non-finite"):
        validate_embeddings(values, 2)


def _fold(configuration, fold, score):
    task, encoder, classifier, weight = configuration
    labels = ("HIGH_IMPACT", "LOWER_IMPACT") if task == "S2" else ("A", "B")
    return {
        "status": "SUCCESS",
        "task": task,
        "encoder_id": encoder,
        "classifier": classifier,
        "class_weight": weight,
        "fold": fold,
        "metrics": {
            "macro_f1": score,
            "balanced_accuracy": score,
            "accuracy": score,
            "weighted_f1": score,
            "per_class": {
                label: {"precision": score, "recall": score, "f1": score, "support": 1}
                for label in labels
            },
        },
    }


def test_deterministic_id_and_leaderboard_tie_break():
    args = ("S6", "E1", "LOGREG", "NONE", 1, "fingerprint")
    assert semantic_experiment_id(*args) == semantic_experiment_id(*args)
    rows = []
    for configuration in (
        ("S6", "E2", "LOGREG", "NONE"),
        ("S6", "E1", "LOGREG", "NONE"),
    ):
        rows.extend(_fold(configuration, fold, 0.5) for fold in (1, 2, 3))
    board = aggregate_semantic_results(rows)
    assert [row["encoder_id"] for row in board] == ["E1", "E2"]


def test_encoder_path_is_frozen_inference_and_modelling_needs_no_locked_artifact():
    from defect_classifier import semantic_benchmark, semantic_embeddings

    source = inspect.getsource(semantic_embeddings.encoder_smoke)
    assert "inference_mode" in source
    assert "requires_grad_(False)" in inspect.getsource(semantic_embeddings._model_provenance)
    benchmark = inspect.getsource(semantic_benchmark.run_semantic_benchmark)
    assert "_load_fold" in benchmark
    assert "_temporal_folds" not in benchmark
    assert "data/locked" not in benchmark
