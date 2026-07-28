from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from mnist_digit_recognition.config import RandomForestConfig, SGDConfig
from mnist_digit_recognition.models import (
    ModelBundle,
    ModelName,
    ModelPersistenceError,
    build_model,
    load_bundle,
    needs_scaling,
    save_bundle,
    train_model,
)


def test_needs_scaling_only_true_for_sgd():
    assert needs_scaling(ModelName.SGD) is True
    assert needs_scaling(ModelName.RANDOM_FOREST) is False


def test_build_model_returns_correct_estimator_types():
    rf_config, sgd_config = RandomForestConfig(n_estimators=5), SGDConfig()

    assert isinstance(build_model(ModelName.SGD, rf_config, sgd_config), SGDClassifier)
    assert isinstance(
        build_model(ModelName.RANDOM_FOREST, rf_config, sgd_config), RandomForestClassifier
    )


def test_model_bundle_requires_scaler_when_needs_scaling_true():
    model = RandomForestClassifier(n_estimators=2)
    with pytest.raises(ValueError, match="needing scaling"):
        ModelBundle(name=ModelName.SGD, model=model, needs_scaling=True, scaler=None)


def test_model_bundle_allows_no_scaler_when_not_needed():
    model = RandomForestClassifier(n_estimators=2)
    bundle = ModelBundle(
        name=ModelName.RANDOM_FOREST, model=model, needs_scaling=False, scaler=None
    )
    assert bundle.scaler is None


def test_save_and_load_bundle_roundtrip(tmp_path, synthetic_dataset):
    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    train_model(model, X, y)

    bundle = ModelBundle(
        name=ModelName.RANDOM_FOREST, model=model, needs_scaling=False, scaler=None
    )
    save_bundle(bundle, tmp_path)

    loaded = load_bundle(tmp_path, ModelName.RANDOM_FOREST)

    assert loaded.name == ModelName.RANDOM_FOREST
    assert loaded.needs_scaling is False
    assert loaded.scaler is None
    assert list(loaded.model.predict(X[:1])) == list(bundle.model.predict(X[:1]))


def test_save_and_load_bundle_preserves_scaler(tmp_path, synthetic_dataset):
    X, y = synthetic_dataset
    scaler = StandardScaler().fit(X)
    model = SGDClassifier(max_iter=100, random_state=0)
    train_model(model, scaler.transform(X), y)

    bundle = ModelBundle(name=ModelName.SGD, model=model, needs_scaling=True, scaler=scaler)
    save_bundle(bundle, tmp_path)

    loaded = load_bundle(tmp_path, ModelName.SGD)

    assert loaded.needs_scaling is True
    assert loaded.scaler is not None


def test_load_bundle_raises_clear_error_when_missing(tmp_path):
    with pytest.raises(ModelPersistenceError, match="No trained model found"):
        load_bundle(tmp_path, ModelName.RANDOM_FOREST)
