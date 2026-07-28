from __future__ import annotations

import numpy as np

from mnist_digit_recognition.evaluate import (
    evaluate_model,
    most_common_misclassifications,
    write_results_report,
)


def test_evaluate_model_computes_accuracy_and_confusion_matrix():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    result = evaluate_model("test_model", y_true, y_pred)

    assert result.accuracy == 4 / 5
    assert result.confusion_matrix.shape == (2, 2)
    assert "precision" in result.classification_report


def test_most_common_misclassifications_orders_by_frequency():
    y_true = np.array([1, 1, 1, 2, 2])
    y_pred = np.array([7, 7, 1, 3, 2])

    top = most_common_misclassifications(y_true, y_pred, top_n=2)

    assert top[0] == ((1, 7), 2)
    assert len(top) == 2


def test_write_results_report_creates_markdown_file(tmp_path):
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    result = evaluate_model("demo", y_true, y_pred)

    output_path = write_results_report([result], tmp_path / "RESULTS.md")

    content = output_path.read_text(encoding="utf-8")
    assert "demo" in content
    assert "0.75" in content or "0.7500" in content
