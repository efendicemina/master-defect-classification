import inspect

import pytest

from defect_classifier import classical_optimization
from defect_classifier.classical_benchmark import BenchmarkError, DevelopmentRow
from defect_classifier.classical_optimization import (
    ALLOWED_C_GRID,
    aggregate_results,
    comparison_delta,
    load_optimization_config,
    search_space_size,
    select_best,
    text_for_view,
    validate_search_space,
)


def _fold(configuration_id, fold, macro, balanced=0.5, features=100):
    return {
        "stage": "A2.1",
        "status": "SUCCESS",
        "task": "S6",
        "configuration_id": configuration_id,
        "c": 1.0,
        "representation_id": "S6-R1",
        "text_view": "SUMMARY_DESCRIPTION",
        "classifier": "LINEARSVC",
        "fold": fold,
        "feature_count": features,
        "metrics": {
            "macro_f1": macro,
            "balanced_accuracy": balanced,
            "accuracy": 0.8,
            "weighted_f1": 0.7,
            "per_class": {},
        },
    }


def test_exact_frozen_search_space_and_model_families():
    config, _ = load_optimization_config()
    assert tuple(config["c_grid"]) == ALLOWED_C_GRID
    assert search_space_size(config) == {"A2.1": 45, "A2.2": 54, "A2.3": 18}
    assert {
        task: (value["classifier"], value["representation"])
        for task, value in config["winners"].items()
    } == {
        "S6": ("LINEARSVC", "CHAR"),
        "S3": ("LOGREG", "WORD"),
        "S2": ("LOGREG", "WORD_CHAR"),
    }
    assert [len(config["representations"][task]) for task in ("S6", "S3", "S2")] == [6, 6, 6]


def test_unapproved_search_axis_fails_closed():
    config, _ = load_optimization_config()
    config["thresholds"] = [0.5]
    with pytest.raises(BenchmarkError, match="unapproved"):
        validate_search_space(config)


def test_selection_uses_minimum_fold_then_features_then_id():
    rows = []
    rows += [
        _fold("wide", fold, score, features=200) for fold, score in enumerate((0.4, 0.5, 0.6), 1)
    ]
    rows += [_fold("stable", fold, 0.5, features=200) for fold in (1, 2, 3)]
    board = aggregate_results(rows)
    assert select_best(board, "A2.1", "S6")["configuration_id"] == "stable"
    rows = [
        _fold(name, fold, 0.5, features=count)
        for name, count in (("z", 100), ("a", 50))
        for fold in (1, 2, 3)
    ]
    assert select_best(aggregate_results(rows), "A2.1", "S6")["configuration_id"] == "a"


def test_text_views_are_separate_and_canonical_combined_is_reused():
    row = DevelopmentRow("p:1", "sum\n\ndesc", {}, summary="sum", description="desc")
    assert text_for_view(row, "SUMMARY_DESCRIPTION") == "sum\n\ndesc"
    assert text_for_view(row, "SUMMARY_ONLY") == "sum"
    assert text_for_view(row, "DESCRIPTION_ONLY") == "desc"
    assert (
        len({text_for_view(row, view) for view in classical_optimization.ALLOWED_TEXT_VIEWS}) == 3
    )


def test_a1_delta_and_locked_independence():
    delta = comparison_delta([0.4, 0.5, 0.6], [0.5, 0.6, 0.7])
    assert delta["absolute_delta"] == pytest.approx(0.1)
    assert delta["fold_deltas"] == pytest.approx([0.1, 0.1, 0.1])
    source = inspect.getsource(classical_optimization.run_classical_optimization)
    assert "_load_fold" in source
    assert "_temporal_folds" not in source
    assert "data/locked" not in source
