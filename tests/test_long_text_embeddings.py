import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from defect_classifier.classical_benchmark import BenchmarkError
from defect_classifier.long_text_embeddings import (
    CHUNK_LENGTH,
    CHUNK_OVERLAP,
    CONTENT_LENGTH,
    EMBEDDING_DIMENSION,
    MODEL_REVISION,
    REFERENCE_HASHES,
    audit_development_chunks,
    chunk_count,
    chunk_token_ids,
    future_fit_count,
    load_long_text_config,
    mean_pool_documents,
    run_long_text_feasibility,
)


def test_frozen_long_text_method_and_future_matrix():
    config, _ = load_long_text_config()
    assert MODEL_REVISION == "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    assert CHUNK_LENGTH == 384
    assert CHUNK_OVERLAP == 64
    assert config["semantic_weight"] == 1.0
    assert future_fit_count(config) == 9


def test_one_chunk_and_complete_deterministic_multi_chunk_coverage():
    short = list(range(20))
    assert chunk_token_ids(short) == [short]
    tokens = list(range(1000))
    first = chunk_token_ids(tokens)
    second = chunk_token_ids(tokens)
    assert first == second
    assert len(first) == chunk_count(len(tokens)) == 3
    assert first[0][-CHUNK_OVERLAP:] == first[1][:CHUNK_OVERLAP]
    assert first[1][-CHUNK_OVERLAP:] == first[2][:CHUNK_OVERLAP]
    assert first[-1] == tokens[2 * (CONTENT_LENGTH - CHUNK_OVERLAP) :]
    covered = {token for chunk in first for token in chunk}
    assert covered == set(tokens)


def test_exact_boundary_and_final_short_chunk():
    assert chunk_count(CONTENT_LENGTH) == 1
    assert chunk_count(CONTENT_LENGTH + 1) == 2
    chunks = chunk_token_ids(list(range(CONTENT_LENGTH + 1)))
    assert len(chunks[0]) == CONTENT_LENGTH
    assert len(chunks[1]) == CHUNK_OVERLAP + 1


def test_mean_pool_then_l2_is_deterministic_and_768_dimensional():
    values = np.zeros((3, EMBEDDING_DIMENSION), dtype=np.float32)
    values[0, 0] = values[1, 1] = values[2, 2] = 1.0
    pooled = mean_pool_documents(values, [0, 0, 1], 2)
    repeated = mean_pool_documents(values, [0, 0, 1], 2)
    assert pooled.shape == (2, 768)
    assert np.array_equal(pooled, repeated)
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)
    assert np.allclose(pooled[0, :2], [2**-0.5, 2**-0.5])


def test_wrong_embedding_dimension_fails_closed():
    values = np.ones((1, 767), dtype=np.float32)
    with pytest.raises(Exception, match="dimension"):
        mean_pool_documents(values, [0], 1)


def test_estimate_uses_development_membership_without_labels_or_locked_text():
    audit_source = inspect.getsource(audit_development_chunks)
    assert "text_combined" in audit_source
    assert "target_" not in audit_source
    assert "severity" not in audit_source
    assert "_membership_fingerprint" in audit_source
    run_source = inspect.getsource(run_long_text_feasibility)
    assert "locked" not in audit_source.casefold()
    assert 'stage == "full"' in run_source
    assert "not authorized" in run_source


def test_encoder_is_frozen_and_estimate_cannot_fit_classifier():
    from defect_classifier import long_text_embeddings

    loader = inspect.getsource(long_text_embeddings._load_frozen_model)
    runner = inspect.getsource(run_long_text_feasibility)
    assert "requires_grad_(False)" in loader
    assert "eval()" in loader
    assert "_fit_one" not in runner
    assert "run_fusion_benchmark" not in runner


def test_reference_reports_unchanged():
    root = Path(__file__).resolve().parents[1] / "reports"
    for _, (relative, expected) in REFERENCE_HASHES.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected


def test_full_stage_is_explicitly_prohibited_before_any_io():
    with pytest.raises(BenchmarkError, match="not authorized"):
        run_long_text_feasibility(
            development_dir=Path("missing"),
            protocol_report_dir=Path("missing"),
            reports_root=Path("missing"),
            report_dir=Path("missing"),
            protocol=None,  # type: ignore[arg-type]
            stage="full",
        )
