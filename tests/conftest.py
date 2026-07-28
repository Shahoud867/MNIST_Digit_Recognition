"""Shared pytest fixtures.

Tests never hit the network or download real MNIST data — everything
uses small synthetic arrays so the suite runs in seconds, both locally
and in CI.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


@pytest.fixture
def synthetic_dataset(rng: np.random.Generator):
    """A tiny, balanced, MNIST-shaped (784 features, digits 0-9) dataset."""
    n_samples_per_class = 12
    n_features = 784
    classes = np.arange(10)

    X_parts = []
    y_parts = []
    for cls in classes:
        # Each class clusters around a distinct mean so a classifier can
        # actually learn something non-trivial from it.
        cluster_mean = rng.uniform(0, 255, size=n_features)
        samples = rng.normal(loc=cluster_mean, scale=10.0, size=(n_samples_per_class, n_features))
        X_parts.append(np.clip(samples, 0, 255))
        y_parts.append(np.full(n_samples_per_class, cls))

    X = np.vstack(X_parts).astype(np.float64)
    y = np.concatenate(y_parts).astype(np.uint8)
    return X, y


@pytest.fixture
def synthetic_image(rng: np.random.Generator) -> np.ndarray:
    """A random 28x28 uint8 grayscale image, MNIST's native shape."""
    return rng.integers(0, 256, size=(28, 28), dtype=np.uint8)
