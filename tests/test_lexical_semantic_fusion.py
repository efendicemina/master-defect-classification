import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from defect_classifier.classical_features import FeatureResult
from defect_classifier.embedding_cache import EmbeddingCacheError, write_shard
from defect_classifier.lexical_semantic_fusion import (
    REFERENCE_HASHES,
    TASKS,
    fuse_sparse_features,
    fusion_experiment_id,
    load_fusion_config,
    load_mpnet_cache,
    search_space_size,
    selected_variant,
    validate_fusion_config,
)
from defect_classifier.protocol import load_protocol


def test_exact_nine_fit_matrix_and_fixed_methods():
    config, _ = load_fusion_config()
    assert search_space_size(config) == 9
    assert config["semantic_encoder_id"] == "E2"
    assert config["semantic_weight"] == 1.0
    assert {
        task: (
            values["representation_id"],
            values["representation"],
            values["classifier"],
            values["c"],
        )
        for task, values in config["tasks"].items()
    } == {
        "S6": ("S6-R3", "CHAR", "LINEARSVC", 0.25),
        "S3": ("S3-R5", "WORD", "LOGREG", 2.0),
        "S2": ("S2-R3", "WORD_CHAR", "LOGREG", 2.0),
    }
    config["semantic_weight"] = 0.5
    with pytest.raises(Exception, match="method drift"):
        validate_fusion_config(config)


def test_exact_a2_variants_are_selected():
    fusion, _ = load_fusion_config()
    from defect_classifier.classical_optimization import load_optimization_config

    a2, _ = load_optimization_config()
    assert [selected_variant(a2, fusion, task)["id"] for task in TASKS] == [
        "S6-R3",
        "S3-R5",
        "S2-R3",
    ]


def test_sparse_fusion_alignment_and_fixed_weight():
    lexical = FeatureResult(
        None,
        sparse.csr_matrix(np.eye(2, dtype=np.float32)),
        sparse.csr_matrix(np.ones((1, 2), dtype=np.float32)),
        2,
    )
    result = fuse_sparse_features(
        lexical,
        np.ones((2, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        1.0,
    )
    assert sparse.isspmatrix_csr(result.training)
    assert result.training.shape == (2, 5)
    with pytest.raises(Exception, match="row alignment"):
        fuse_sparse_features(lexical, np.ones((1, 3), dtype=np.float32), np.ones((1, 3)), 1.0)
    with pytest.raises(Exception, match="weight"):
        fuse_sparse_features(lexical, np.ones((2, 3)), np.ones((1, 3)), 0.5)


def _cache(tmp_path, projects):
    protocol = load_protocol()
    config, _ = load_fusion_config()
    root = tmp_path / "E2"
    metadata = {
        "encoder_id": config["semantic_encoder_id"],
        "model_id": config["semantic_model_id"],
        "resolved_revision": config["semantic_revision"],
        "embedding_dimension": 768,
        "normalize_embeddings": True,
        "protocol_sha256": protocol.fingerprint,
        "development_membership_sha256": "development",
    }
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(metadata))
    for project, issues in projects:
        values = np.zeros((len(issues), 768), dtype=np.float32)
        values[:, 0] = 1.0
        write_shard(
            root / "shards" / f"{project}-{len(list((root / 'shards').glob('*')))}.parquet",
            project,
            issues,
            values,
        )
    return protocol, config


def test_missing_and_duplicate_embedding_identity_fail_closed(tmp_path):
    protocol, config = _cache(tmp_path / "missing", [("P", ["1"])])
    with pytest.raises(EmbeddingCacheError, match="exact frozen DEVELOPMENT"):
        load_mpnet_cache(tmp_path / "missing", {"P:1", "P:2"}, protocol, "development", config)
    protocol, config = _cache(tmp_path / "duplicate", [("P", ["1"]), ("P", ["1"])])
    with pytest.raises(EmbeddingCacheError, match="duplicate"):
        load_mpnet_cache(tmp_path / "duplicate", {"P:1"}, protocol, "development", config)


def test_train_only_tfidf_deterministic_ids_and_no_locked_dependency():
    from defect_classifier import classical_features, lexical_semantic_fusion

    source = inspect.getsource(classical_features.build_sparse_features)
    assert "fit_transform(training_text)" in source
    assert "transform(validation_text)" in source
    assert fusion_experiment_id("S6", 1, "x") == fusion_experiment_id("S6", 1, "x")
    benchmark = inspect.getsource(lexical_semantic_fusion.run_fusion_benchmark)
    assert "data/locked" not in benchmark
    assert "_load_fold" in benchmark


def test_reference_report_fingerprints_unchanged():
    root = Path(__file__).resolve().parents[1] / "reports"
    for _, (relative, expected) in REFERENCE_HASHES.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
