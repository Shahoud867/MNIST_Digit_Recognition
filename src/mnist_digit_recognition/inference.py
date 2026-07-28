"""Image preprocessing and prediction for the interactive demo.

This is where the original notebook's bug lived: a scaler was fit,
saved, and reloaded, but `predict_digit()` called `rf_clf.predict()`
directly on unscaled pixels, silently ignoring it. It happened to
"work" only because RandomForest doesn't need scaling — but the same
code path would have produced wrong predictions for the SGD model.

`DigitPredictor` fixes this by branching on `ModelBundle.needs_scaling`
explicitly, and this module's tests assert that both the scaled and
unscaled paths are exercised correctly (see tests/test_inference.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from mnist_digit_recognition.logging_config import get_logger
from mnist_digit_recognition.models import ModelBundle

logger = get_logger("inference")

MNIST_IMAGE_SIZE = (28, 28)
MAX_INPUT_DIMENSION = 4096  # guards against pathological/oversized uploads


class InvalidImageError(ValueError):
    """Raised when an uploaded image can't be safely preprocessed."""


@dataclass(frozen=True)
class Prediction:
    label: int
    confidence: float | None  # None when the model has no predict_proba


def preprocess_image(image: Image.Image | np.ndarray, invert: bool = False) -> np.ndarray:
    """Convert a user-supplied image into a flattened 28x28 grayscale vector.

    Raises:
        InvalidImageError: for unusable input (wrong type, zero-sized,
            or implausibly large images that would be expensive/unsafe
            to process).
    """
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise InvalidImageError("Received an empty image array.")
        pil_image = Image.fromarray(image.astype("uint8"))
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise InvalidImageError(
            f"Unsupported image type: {type(image).__name__}. Expected a PIL "
            "Image or numpy array."
        )

    width, height = pil_image.size
    if width == 0 or height == 0:
        raise InvalidImageError("Image has zero width or height.")
    if width > MAX_INPUT_DIMENSION or height > MAX_INPUT_DIMENSION:
        raise InvalidImageError(
            f"Image dimensions {pil_image.size} exceed the maximum allowed "
            f"({MAX_INPUT_DIMENSION}px per side)."
        )

    grayscale = pil_image.convert("L").resize(MNIST_IMAGE_SIZE)
    pixels = np.array(grayscale)
    if invert:
        pixels = 255 - pixels
    return pixels.flatten().reshape(1, -1).astype(np.float64)


def predict(
    bundle: ModelBundle, image: Image.Image | np.ndarray, invert: bool = False
) -> Prediction:
    """Run the full preprocess -> (optional scale) -> predict pipeline.

    This is the corrected version of the original notebook's
    `predict_digit`: it only applies `bundle.scaler` when
    `bundle.needs_scaling` is true, so the scaler is never silently
    unused (the original bug) nor silently skipped for models that
    require it.
    """
    features = preprocess_image(image, invert=invert)

    if bundle.needs_scaling:
        if bundle.scaler is None:  # pragma: no cover - guarded by ModelBundle.__post_init__
            raise RuntimeError(f"Model {bundle.name!r} needs scaling but has no scaler.")
        features = bundle.scaler.transform(features)

    label = int(bundle.model.predict(features)[0])

    confidence: float | None = None
    if hasattr(bundle.model, "predict_proba"):
        probabilities = bundle.model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

    logger.debug("Predicted label=%s confidence=%s (model=%s)", label, confidence, bundle.name)
    return Prediction(label=label, confidence=confidence)


def predict_proba_by_class(
    bundle: ModelBundle, image: Image.Image | np.ndarray, invert: bool = False
) -> dict[str, float]:
    """Return a full class -> probability mapping, for UIs that show top-k."""
    features = preprocess_image(image, invert=invert)
    if bundle.needs_scaling and bundle.scaler is not None:
        features = bundle.scaler.transform(features)

    if hasattr(bundle.model, "predict_proba"):
        probabilities = bundle.model.predict_proba(features)[0]
        classes = bundle.model.classes_
        return {str(cls): float(prob) for cls, prob in zip(classes, probabilities, strict=False)}

    label = int(bundle.model.predict(features)[0])
    return {str(label): 1.0}
