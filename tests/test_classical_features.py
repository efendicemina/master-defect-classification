from defect_classifier.classical_features import build_sparse_features

CONFIG = {
    "word": {
        "analyzer": "word",
        "ngram_min": 1,
        "ngram_max": 1,
        "min_df": 1,
        "sublinear_tf": True,
        "lowercase": True,
        "max_features": 100,
    },
    "char": {
        "analyzer": "char_wb",
        "ngram_min": 3,
        "ngram_max": 3,
        "min_df": 1,
        "sublinear_tf": True,
        "lowercase": True,
        "max_features": 100,
    },
    "word_char": {"word_max_features": 50, "char_max_features": 50},
}


def test_vectorizers_fit_train_only_and_remain_sparse():
    result = build_sparse_features("WORD", ["training token"], ["validationonly"], CONFIG)
    from scipy import sparse

    assert "validationonly" not in result.transformer.vocabulary_
    assert sparse.issparse(result.training)
    assert sparse.issparse(result.validation)
    assert result.training.shape[0] == 1
    assert result.validation.shape[0] == 1


def test_combined_vectorizers_fit_train_only():
    result = build_sparse_features("WORD_CHAR", ["alpha train"], ["zzzzvalidation"], CONFIG)
    from scipy import sparse

    word = result.transformer.transformer_list[0][1]
    assert "zzzzvalidation" not in word.vocabulary_
    assert sparse.issparse(result.training)
    assert sparse.issparse(result.validation)
