import inspect

import pytest

from defect_classifier import classical_benchmark
from defect_classifier.classical_benchmark import (
    BenchmarkError,
    _checkpoint,
    _load_checkpoint,
    aggregate_leaderboard,
    experiment_id,
)


def _result(task, representation, score, fold, stage="COMPETITIVE"):
    labels = ("HIGH_IMPACT", "LOWER_IMPACT") if task == "S2" else ("A", "B")
    per_class = {
        label: {"precision": score, "recall": score, "f1": score, "support": 1} for label in labels
    }
    return {
        "stage": stage,
        "status": "SUCCESS",
        "task": task,
        "representation": representation,
        "classifier": "LINEARSVC",
        "class_weight": "NONE",
        "fold": fold,
        "metrics": {
            "macro_f1": score,
            "balanced_accuracy": score,
            "accuracy": score,
            "weighted_f1": score,
            "per_class": per_class,
        },
    }


def test_deterministic_ids_and_ranking_excludes_stage0():
    args = ("COMPETITIVE", "S3", "WORD", "LINEARSVC", "NONE", 1, "abc")
    assert experiment_id(*args) == experiment_id(*args)
    results = [_result("S3", "WORD", 0.5, fold) for fold in (1, 2, 3)]
    results += [_result("S3", "CHAR", 0.5, fold) for fold in (1, 2, 3)]
    results.append(_result("S3", "WORD", 1.0, 1, "ENGINEERING_ONLY"))
    board = aggregate_leaderboard(results)
    assert [row["representation"] for row in board] == ["CHAR", "WORD"]
    assert all(row["mean_macro_f1"] == 0.5 for row in board)


def test_s2_guard_is_informational():
    board = aggregate_leaderboard([_result("S2", "WORD", 0.29, fold) for fold in (1, 2, 3)])
    assert board[0]["legacy_reproduction_guard"] == "FAIL"


def test_checkpoint_resume_and_protocol_drift_fail_closed(tmp_path):
    path = tmp_path / "checkpoint.json"
    provenance = {"protocol_sha256": "one"}
    _checkpoint(path, {**provenance, "status": "SUCCESS"})
    assert _load_checkpoint(path, provenance)["status"] == "SUCCESS"
    with pytest.raises(BenchmarkError, match="provenance drift"):
        _load_checkpoint(path, {"protocol_sha256": "two"})


def test_engine_uses_manifest_loader_and_has_no_locked_artifact_dependency():
    source = inspect.getsource(classical_benchmark.run_classical_benchmark)
    assert "_load_fold" in source
    assert "load_cv_membership" in inspect.getsource(classical_benchmark._load_fold)
    assert "_temporal_folds" not in source
    assert "data/locked" not in source
