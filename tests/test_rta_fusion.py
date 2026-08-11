import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.rta_fusion import (
    ARCHITECTURE,
    DIMENSION,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    REFERENCE_HASHES,
    REPRESENTATION,
    extract_first_token,
    finalize_rta_checkpoints,
    future_fit_count,
    load_rta_config,
    normalize_rta_embeddings,
    run_rta_feasibility,
    validate_rta_config,
)


def test_exact_rta_identity_and_fixed_future_matrix():
    config, _ = load_rta_config()
    assert MODEL_ID == "Colorful/RTA"
    assert len(MODEL_REVISION) == 40
    assert ARCHITECTURE == "RobertaForSequenceClassification"
    assert REPRESENTATION == "base_encoder_final_layer_first_token"
    assert DIMENSION == 768 and MAX_LENGTH == 512
    assert config["semantic_weight"] == 1.0
    assert future_fit_count(config) == 9


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("model_id", "roberta-base"),
        ("resolved_revision", "main"),
        ("representation", "mean"),
        ("embedding_dimension", 1024),
        ("semantic_weight", 0.5),
    ),
)
def test_rta_provenance_or_method_drift_fails_closed(key, value):
    config, _ = load_rta_config()
    config[key] = value
    with pytest.raises(BenchmarkError, match="method drift"):
        validate_rta_config(config)


def test_first_token_extraction_is_deterministic_and_not_logits():
    hidden = np.arange(2 * 4 * 768, dtype=np.float32).reshape(2, 4, 768)
    first = extract_first_token(hidden)
    assert first.shape == (2, 768)
    assert np.array_equal(first, hidden[:, 0, :])
    assert np.array_equal(first, extract_first_token(hidden))


def test_l2_normalization_is_deterministic_and_768_dimensional():
    values = np.ones((3, 768), dtype=np.float32)
    output = normalize_rta_embeddings(values)
    assert output.dtype == np.float32 and output.shape == (3, 768)
    assert np.allclose(np.linalg.norm(output, axis=1), 1.0)
    assert np.array_equal(output, normalize_rta_embeddings(values))


def test_feasibility_cannot_launch_full_or_fit():
    source = inspect.getsource(run_rta_feasibility)
    assert 'stage == "full"' in source and "not authorized" in source
    assert "_fit_one(" not in source
    assert "run_fusion_benchmark(" not in source
    assert "data/locked" not in source
    assert "competitive_fits_started" in source


def test_full_path_uses_authenticated_resume_and_frozen_cv():
    from defect_classifier import rta_fusion

    source = inspect.getsource(rta_fusion.run_rta_full)
    assert "sidecar.is_file()" in source
    assert "RTA shard provenance drift" in source
    assert "read_shard" in source and "validate_membership" in source
    assert "resume=resume" in source
    assert "_load_fold(" in source and "_temporal_folds" not in source
    assert "data/locked" not in source
    assert "locked_test_accessed=False" in source
    assert "for task in TASKS" in source


def test_finalization_can_only_reuse_persisted_artifacts():
    source = inspect.getsource(finalize_rta_checkpoints)
    assert "read_shard" in source and "existing_checkpoints_reused" in source
    assert "competitive_models_refitted" in source
    assert "_load_model(" not in source
    assert "_encode(" not in source
    assert "_run_fit(" not in source
    assert "build_sparse_features(" not in source
    assert "data/locked" not in source


def test_encoder_inputs_use_text_only_without_targets():
    from defect_classifier import rta_fusion

    source = inspect.getsource(rta_fusion.audit_tokenization)
    assert "text_combined" in source
    assert "target_" not in source and "severity" not in source
    assert "_membership_fingerprint" in source
    loader = inspect.getsource(rta_fusion._load_model)
    assert "eval()" in loader and "requires_grad_(False)" in loader
    assert "revision=MODEL_REVISION" in loader


def test_full_stage_fails_before_io():
    with pytest.raises(BenchmarkError, match="not authorized"):
        run_rta_feasibility(
            development_dir=Path("missing"),
            protocol_report_dir=Path("missing"),
            reports_root=Path("missing"),
            report_dir=Path("missing"),
            protocol=None,  # type: ignore[arg-type]
            stage="full",
        )


def test_reference_reports_unchanged():
    root = Path(__file__).resolve().parents[1] / "reports"
    for _, (relative, expected) in REFERENCE_HASHES.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
