"""Model construction, training, and persistence.

Replaces the duplicated per-model cells (2.1/2.2 and 5.1 in the original
notebook) with a single, parameterized code path plus a
:class:`ModelBundle` that records whether a model needs scaled input —
this is the fix for the original scaler bug, where the Gradio app saved
a scaler it never actually applied. Now "does this model need scaling"
is an explicit, checked property instead of an implicit assumption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from mnist_digit_recognition.config import RandomForestConfig, SGDConfig
from mnist_digit_recognition.logging_config import get_logger

logger = get_logger("models")


class ModelName(str, Enum):
    SGD = "sgd"
    RANDOM_FOREST = "random_forest"


class ModelPersistenceError(RuntimeError):
    """Raised when a model/scaler bundle cannot be saved or loaded."""


@dataclass(frozen=True)
class ModelBundle:
    """A trained model plus everything needed to correctly run inference.

    ``needs_scaling`` is the explicit fix for the original bug: the RF
    model was trained on raw pixels while a scaler was saved alongside
    it but never applied at inference time. Callers must now check this
    flag (see :func:`mnist_digit_recognition.inference.predict`) rather
    than assuming a scaler should always be used.
    """

    name: ModelName
    model: ClassifierMixin
    needs_scaling: bool
    scaler: StandardScaler | None = None

    def __post_init__(self) -> None:
        if self.needs_scaling and self.scaler is None:
            raise ValueError(
                f"Model {self.name!r} is marked as needing scaling but no " "scaler was provided."
            )


def build_model(
    name: ModelName, rf_config: RandomForestConfig, sgd_config: SGDConfig
) -> ClassifierMixin:
    """Construct an untrained estimator for the given model name."""
    if name is ModelName.SGD:
        return SGDClassifier(
            loss=sgd_config.loss,
            random_state=sgd_config.random_state,
            max_iter=sgd_config.max_iter,
            tol=sgd_config.tol,
        )
    if name is ModelName.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=rf_config.n_estimators,
            random_state=rf_config.random_state,
            n_jobs=rf_config.n_jobs,
        )
    raise ValueError(f"Unknown model name: {name!r}")  # pragma: no cover - exhaustive enum


def needs_scaling(name: ModelName) -> bool:
    """Whether a given model expects standardized input features.

    RandomForest is scale-invariant (tree splits are threshold-based),
    so — matching the original notebook's actual training code — it is
    trained and served on raw pixel values. SGD/linear models are
    scale-sensitive and require standardization.
    """
    return name is ModelName.SGD


def train_model(
    model: ClassifierMixin, X_train: np.ndarray, y_train: np.ndarray
) -> ClassifierMixin:
    """Fit ``model`` in place, logging wall-clock training time."""
    started = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    logger.info("Trained %s in %.2fs", type(model).__name__, elapsed)
    return model


def save_bundle(bundle: ModelBundle, model_dir: Path) -> Path:
    """Persist a model bundle to ``model_dir`` as ``<name>.joblib``.

    Returns the path written. Raises :class:`ModelPersistenceError` on
    any I/O failure so callers get a clear, typed error instead of a
    bare ``OSError``/``PermissionError`` traceback.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{bundle.name.value}.joblib"
    payload: dict[str, Any] = {
        "name": bundle.name.value,
        "model": bundle.model,
        "needs_scaling": bundle.needs_scaling,
        "scaler": bundle.scaler,
    }
    try:
        joblib.dump(payload, path)
    except OSError as exc:
        raise ModelPersistenceError(f"Failed to save model bundle to {path}") from exc
    logger.info("Saved model bundle to %s", path)
    return path


def load_bundle(model_dir: Path, name: ModelName) -> ModelBundle:
    """Load a previously saved model bundle.

    Raises:
        ModelPersistenceError: if the artifact is missing or corrupt.
    """
    path = model_dir / f"{name.value}.joblib"
    if not path.exists():
        raise ModelPersistenceError(
            f"No trained model found at {path}. Run the training script first: "
            "`python scripts/train.py`."
        )
    try:
        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain-specific error
        raise ModelPersistenceError(f"Failed to load model bundle from {path}") from exc

    return ModelBundle(
        name=ModelName(payload["name"]),
        model=payload["model"],
        needs_scaling=payload["needs_scaling"],
        scaler=payload.get("scaler"),
    )
