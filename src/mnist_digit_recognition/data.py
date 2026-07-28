"""Dataset loading, splitting, and feature scaling.

Extracted from the original notebook's "Data Preparation" section
(cells 1.1–1.4). Behavior is preserved exactly (same OpenML dataset,
same stratified split, same StandardScaler) but each step is now a
tested, importable function instead of inline cell code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sk_shuffle

from mnist_digit_recognition.config import DataConfig
from mnist_digit_recognition.logging_config import get_logger

logger = get_logger("data")


class DatasetLoadError(RuntimeError):
    """Raised when the MNIST dataset cannot be fetched or is malformed."""


@dataclass(frozen=True)
class Dataset:
    """A train/test split of raw (unscaled) MNIST pixel data."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_mnist(config: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Fetch MNIST from OpenML and return shuffled ``(X, y)`` arrays.

    Raises:
        DatasetLoadError: if the dataset cannot be retrieved (e.g. no
            network access, or an unexpected OpenML response shape).
    """
    from sklearn.datasets import fetch_openml  # local import: heavy + network I/O

    logger.info(
        "Fetching dataset %r (version=%s) from OpenML",
        config.openml_name,
        config.openml_version,
    )
    try:
        mnist = fetch_openml(config.openml_name, version=config.openml_version, as_frame=False)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain-specific error
        raise DatasetLoadError(
            f"Failed to fetch dataset {config.openml_name!r} from OpenML. "
            "Check network connectivity or try again later."
        ) from exc

    X, y = mnist["data"], mnist["target"]
    if X is None or y is None or len(X) == 0:
        raise DatasetLoadError("OpenML returned an empty or malformed dataset.")

    X, y = sk_shuffle(X, y, random_state=config.random_state)
    y = y.astype(np.uint8)
    logger.info("Loaded dataset: X=%s y=%s", X.shape, y.shape)
    return X, y


def split_dataset(X: np.ndarray, y: np.ndarray, config: DataConfig) -> Dataset:
    """Stratified train/test split, matching the original notebook's ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    logger.info(
        "Split dataset: train=%s test=%s (test_size=%s)",
        X_train.shape,
        X_test.shape,
        config.test_size,
    )
    return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a :class:`StandardScaler` on training data only (no leakage)."""
    scaler = StandardScaler()
    scaler.fit(X_train.astype(np.float64))
    return scaler


def scale_features(
    scaler: StandardScaler, X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fitted scaler to train and test splits."""
    X_train_scaled = scaler.transform(X_train.astype(np.float64))
    X_test_scaled = scaler.transform(X_test.astype(np.float64))
    return X_train_scaled, X_test_scaled
