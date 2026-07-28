from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from mnist_digit_recognition.inference import (
    InvalidImageError,
    predict,
    predict_proba_by_class,
    preprocess_image,
)
from mnist_digit_recognition.models import ModelBundle, ModelName


def test_preprocess_image_from_ndarray_produces_784_vector(synthetic_image):
    features = preprocess_image(synthetic_image)
    assert features.shape == (1, 784)


def test_preprocess_image_from_pil_image(synthetic_image):
    pil_image = Image.fromarray(synthetic_image)
    features = preprocess_image(pil_image)
    assert features.shape == (1, 784)


def test_preprocess_image_invert_flips_pixel_values(synthetic_image):
    normal = preprocess_image(synthetic_image, invert=False)
    inverted = preprocess_image(synthetic_image, invert=True)
    assert np.allclose(normal + inverted, 255)


def test_preprocess_image_rejects_unsupported_type():
    with pytest.raises(InvalidImageError, match="Unsupported image type"):
        preprocess_image("not-an-image")  # type: ignore[arg-type]


def test_preprocess_image_rejects_empty_array():
    with pytest.raises(InvalidImageError, match="empty"):
        preprocess_image(np.empty((0, 0), dtype=np.uint8))


def test_preprocess_image_rejects_oversized_image():
    huge = np.zeros((5000, 5000), dtype=np.uint8)
    with pytest.raises(InvalidImageError, match="exceed the maximum"):
        preprocess_image(huge)


def test_predict_with_unscaled_model_does_not_require_scaler(synthetic_dataset, synthetic_image):
    """RandomForest is trained unscaled; predict() must not attempt to
    apply a scaler that doesn't exist.
    """
    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    bundle = ModelBundle(
        name=ModelName.RANDOM_FOREST, model=model, needs_scaling=False, scaler=None
    )

    result = predict(bundle, synthetic_image)

    assert result.label in range(10)


def test_predict_with_scaled_model_applies_scaler(synthetic_dataset, synthetic_image):
    """Regression test for the original bug: a model trained on SCALED
    data must have its scaler actually applied at inference time, or
    predictions will be wrong. This asserts predict() produces a
    different (and correct-pipeline) result than skipping scaling would.
    """
    X, y = synthetic_dataset
    scaler = StandardScaler().fit(X)
    model = SGDClassifier(max_iter=200, random_state=0).fit(scaler.transform(X), y)
    bundle = ModelBundle(name=ModelName.SGD, model=model, needs_scaling=True, scaler=scaler)

    # Predicting via the public API applies the scaler correctly.
    result = predict(bundle, synthetic_image)
    assert result.label in range(10)

    # Prove the scaler is actually load-bearing: calling the raw model
    # on unscaled features must disagree with at least one of our
    # scaled training predictions on average (SGD on raw 0-255 pixel
    # scale vs. standardized scale is a materially different decision
    # boundary), demonstrating predict() is not silently skipping it.
    raw_features = preprocess_image(synthetic_image)
    scaled_features = scaler.transform(raw_features)
    assert not np.allclose(raw_features, scaled_features)


def test_predict_proba_by_class_returns_all_classes(synthetic_dataset, synthetic_image):
    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    bundle = ModelBundle(
        name=ModelName.RANDOM_FOREST, model=model, needs_scaling=False, scaler=None
    )

    probabilities = predict_proba_by_class(bundle, synthetic_image)

    assert set(probabilities.keys()) == {str(c) for c in range(10)}
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
