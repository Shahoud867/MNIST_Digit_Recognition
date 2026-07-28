"""Model evaluation: metrics, confusion matrices, and misclassification analysis.

Extracted from notebook sections 3 and 4. Plotting is optional
(figures are saved to disk rather than only shown inline), so this
module is usable headlessly in CI/scripts, not just in Jupyter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mnist_digit_recognition.logging_config import get_logger

logger = get_logger("evaluate")


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    accuracy: float
    classification_report: str
    confusion_matrix: np.ndarray


def evaluate_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult:
    """Compute accuracy, a classification report, and a confusion matrix."""
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    logger.info("%s accuracy: %.4f", model_name, accuracy)
    return EvaluationResult(
        model_name=model_name,
        accuracy=accuracy,
        classification_report=report,
        confusion_matrix=cm,
    )


def plot_confusion_matrix(result: EvaluationResult, output_path: Path) -> Path:
    """Render and save a confusion-matrix heatmap. Requires matplotlib/seaborn."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe backend for CI/scripts
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(result.confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{result.model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", output_path)
    return output_path


def most_common_misclassifications(
    y_true: np.ndarray, y_pred: np.ndarray, top_n: int = 5
) -> list[tuple[tuple[int, int], int]]:
    """Return the most frequent (actual, predicted) error pairs, most common first."""
    misclassified_idx = np.where(y_pred != y_true)[0]
    error_pairs = list(
        zip(y_true[misclassified_idx].tolist(), y_pred[misclassified_idx].tolist(), strict=False)
    )
    return Counter(error_pairs).most_common(top_n)


def write_results_report(results: list[EvaluationResult], output_path: Path) -> Path:
    """Write a Markdown report summarizing all model results (for README/reports)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Model Evaluation Results", ""]
    lines.append("| Model | Accuracy |")
    lines.append("|---|---|")
    for result in results:
        lines.append(f"| {result.model_name} | {result.accuracy:.4f} |")
    lines.append("")
    for result in results:
        lines.append(f"## {result.model_name} — Classification Report")
        lines.append("```")
        lines.append(result.classification_report)
        lines.append("```")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote results report to %s", output_path)
    return output_path
