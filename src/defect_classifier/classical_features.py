"""Sparse classical text representations for the Phase A1 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureResult:
    transformer: Any
    training: Any
    validation: Any
    feature_count: int


def build_sparse_features(
    representation: str,
    training_text: list[str],
    validation_text: list[str],
    config: dict[str, Any],
    *,
    smoke_max_features: int | None = None,
) -> FeatureResult:
    """Fit TF-IDF on training text only, then transform validation text."""
    import numpy as np
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    def vectorizer(section: str, max_features: int) -> TfidfVectorizer:
        values = config[section]
        return TfidfVectorizer(
            analyzer=values["analyzer"],
            ngram_range=(values["ngram_min"], values["ngram_max"]),
            min_df=values["min_df"],
            sublinear_tf=values["sublinear_tf"],
            lowercase=values["lowercase"],
            max_features=max_features,
            dtype=np.float32,
        )

    if representation == "WORD":
        transformer: Any = vectorizer("word", smoke_max_features or config["word"]["max_features"])
    elif representation == "CHAR":
        transformer = vectorizer("char", smoke_max_features or config["char"]["max_features"])
    elif representation == "WORD_CHAR":
        word_cap = smoke_max_features or config["word_char"]["word_max_features"]
        char_cap = smoke_max_features or config["word_char"]["char_max_features"]
        transformer = FeatureUnion(
            [("word", vectorizer("word", word_cap)), ("char", vectorizer("char", char_cap))],
            n_jobs=1,
        )
    else:
        raise ValueError(f"unknown representation: {representation}")
    training_matrix = transformer.fit_transform(training_text)
    validation_matrix = transformer.transform(validation_text)
    if not sparse.issparse(training_matrix) or not sparse.issparse(validation_matrix):
        raise TypeError("TF-IDF representation unexpectedly became dense")
    return FeatureResult(
        transformer=transformer,
        training=training_matrix.tocsr(),
        validation=validation_matrix.tocsr(),
        feature_count=training_matrix.shape[1],
    )
