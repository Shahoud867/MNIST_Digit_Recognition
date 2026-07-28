"""Gradio web demo for interactive digit recognition.

Replaces notebook section 5.2. User-facing errors are now surfaced as
`gr.Error` messages instead of raw tracebacks (see UX/error-handling
findings from the repo audit).
"""

from __future__ import annotations

import gradio as gr

from mnist_digit_recognition.config import AppConfig
from mnist_digit_recognition.inference import InvalidImageError, predict_proba_by_class
from mnist_digit_recognition.logging_config import get_logger
from mnist_digit_recognition.models import (
    ModelBundle,
    ModelName,
    ModelPersistenceError,
    load_bundle,
)

logger = get_logger("app")


def build_predict_fn(bundle: ModelBundle):
    """Return a Gradio-callback closure bound to a loaded model bundle."""

    def _predict(image, invert: bool) -> dict[str, float]:
        if image is None:
            raise gr.Error("Please draw or upload a digit image before predicting.")
        try:
            return predict_proba_by_class(bundle, image, invert=invert)
        except InvalidImageError as exc:
            logger.warning("Rejected invalid input image: %s", exc)
            raise gr.Error(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - convert to a user-safe message
            logger.exception("Unexpected error during prediction")
            raise gr.Error(
                "Something went wrong while predicting. Please try a different image."
            ) from exc

    return _predict


def build_interface(bundle: ModelBundle) -> gr.Interface:
    """Construct the Gradio Interface for a given loaded model bundle."""
    return gr.Interface(
        fn=build_predict_fn(bundle),
        inputs=[
            gr.Image(image_mode="L", label="Draw or upload a digit (0-9)"),
            gr.Checkbox(label="Invert colors (black digit on white background)"),
        ],
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="MNIST Digit Recognizer",
        description=(
            f"Model: **{bundle.name.value}**. Draw a digit or upload a 28x28-ish "
            "grayscale image and the model will predict it."
        ),
        examples=None,
    )


def launch(config: AppConfig, model_name: ModelName = ModelName.RANDOM_FOREST) -> None:
    """Load a trained model and launch the Gradio server."""
    try:
        bundle = load_bundle(config.model_dir, model_name)
    except ModelPersistenceError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    interface = build_interface(bundle)
    interface.launch(
        share=config.gradio_share,
        server_port=config.gradio_server_port,
        server_name=config.gradio_server_name,
    )
