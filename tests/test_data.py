from __future__ import annotations

import numpy as np
import pytest

from mnist_digit_recognition.config import DataConfig
from mnist_digit_recognition.data import (
    DatasetLoadError,
    fit_scaler,
    load_mnist,
    scale_features,
    split_dataset,
)


def test_split_dataset_respects_test_size(synthetic_dataset):
    X, y = synthetic_dataset
    config = DataConfig(test_size=20, random_state=0)

    dataset = split_dataset(X, y, config)

    assert len(dataset.X_test) == 20
    assert len(dataset.X_train) == len(X) - 20
    assert len(dataset.y_train) == len(dataset.X_train)
    assert len(dataset.y_test) == len(dataset.X_test)


def test_split_dataset_is_stratified(synthetic_dataset):
    X, y = synthetic_dataset
    config = DataConfig(test_size=20, random_state=0)

    dataset = split_dataset(X, y, config)

    # 20 test samples across 10 balanced classes -> exactly 2 per class.
    _, counts = np.unique(dataset.y_test, return_counts=True)
    assert set(counts.tolist()) == {2}


def test_fit_scaler_normalizes_training_data(synthetic_dataset):
    X, _ = synthetic_dataset

    scaler = fit_scaler(X)
    X_scaled, _ = scale_features(scaler, X, X)

    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-6)


def test_scale_features_does_not_refit_on_test_data(synthetic_dataset):
    """The scaler must be fit on train only — transforming test data
    should not change its learned mean/scale (guards against data leakage).
    """
    X, _ = synthetic_dataset
    X_train, X_test = X[:100], X[100:]

    scaler = fit_scaler(X_train)
    mean_before = scaler.mean_.copy()
    scale_features(scaler, X_train, X_test)

    assert np.array_equal(scaler.mean_, mean_before)


def test_load_mnist_wraps_fetch_failures(monkeypatch):
    def _boom(*args, **kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr("sklearn.datasets.fetch_openml", _boom)

    with pytest.raises(DatasetLoadError):
        load_mnist(DataConfig())


def test_load_mnist_rejects_empty_response(monkeypatch):
    def _empty(*args, **kwargs):
        return {"data": np.empty((0, 784)), "target": np.empty((0,))}

    monkeypatch.setattr("sklearn.datasets.fetch_openml", _empty)

    with pytest.raises(DatasetLoadError):
        load_mnist(DataConfig())
