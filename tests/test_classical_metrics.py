import pytest

from defect_classifier.classical_metrics import classification_metrics


def test_fixed_complete_labels_and_confusion_order():
    labels = ("A", "B", "C")
    metrics = classification_metrics(["A", "B"], ["A", "A"], labels)
    assert list(metrics["per_class"]) == list(labels)
    assert metrics["per_class"]["C"]["support"] == 0
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0 + 0) / 3)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 0]]
