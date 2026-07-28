"""Command-line entrypoints: `mnist-train` and `mnist-serve`.

Also invoked directly by scripts/train.py and scripts/serve.py for
users who haven't installed the package via pip.
"""

from __future__ import annotations

import argparse
import sys

from mnist_digit_recognition.config import load_config
from mnist_digit_recognition.data import (
    DatasetLoadError,
    fit_scaler,
    load_mnist,
    scale_features,
    split_dataset,
)
from mnist_digit_recognition.evaluate import (
    evaluate_model,
    most_common_misclassifications,
    plot_confusion_matrix,
    write_results_report,
)
from mnist_digit_recognition.logging_config import get_logger, setup_logging
from mnist_digit_recognition.models import (
    ModelBundle,
    ModelName,
    build_model,
    needs_scaling,
    save_bundle,
    train_model,
)

logger = get_logger("cli")


def _parse_train_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST digit classifiers.")
    parser.add_argument(
        "--model",
        choices=["sgd", "random_forest", "all"],
        default="all",
        help="Which model(s) to train (default: all).",
    )
    parser.add_argument("--log-level", default=None, help="Override MNIST_LOG_LEVEL.")
    return parser.parse_args(argv)


def train_cli(argv: list[str] | None = None) -> int:
    """Entrypoint for `mnist-train` / `python scripts/train.py`."""
    args = _parse_train_args(argv)
    config = load_config()
    setup_logging(args.log_level or config.log_level)
    config.ensure_directories()

    model_names = (
        [ModelName.SGD, ModelName.RANDOM_FOREST] if args.model == "all" else [ModelName(args.model)]
    )

    try:
        X, y = load_mnist(config.data)
    except DatasetLoadError:
        logger.exception("Could not load the MNIST dataset")
        return 1

    dataset = split_dataset(X, y, config.data)
    scaler = fit_scaler(dataset.X_train)
    X_train_scaled, X_test_scaled = scale_features(scaler, dataset.X_train, dataset.X_test)

    results = []
    for name in model_names:
        model = build_model(name, config.random_forest, config.sgd)
        use_scaled = needs_scaling(name)
        X_train = X_train_scaled if use_scaled else dataset.X_train
        X_test = X_test_scaled if use_scaled else dataset.X_test

        train_model(model, X_train, dataset.y_train)
        y_pred = model.predict(X_test)
        result = evaluate_model(name.value, dataset.y_test, y_pred)
        results.append(result)

        bundle = ModelBundle(
            name=name,
            model=model,
            needs_scaling=use_scaled,
            scaler=scaler if use_scaled else None,
        )
        save_bundle(bundle, config.model_dir)
        plot_confusion_matrix(result, config.reports_dir / f"{name.value}_confusion_matrix.png")

        top_errors = most_common_misclassifications(dataset.y_test, y_pred)
        logger.info("%s top misclassifications: %s", name.value, top_errors)

    write_results_report(results, config.reports_dir / "RESULTS.md")
    logger.info(
        "Training complete. Artifacts in %s, reports in %s", config.model_dir, config.reports_dir
    )
    return 0


def _parse_serve_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the MNIST digit recognizer as a Gradio app."
    )
    parser.add_argument(
        "--model",
        choices=["sgd", "random_forest"],
        default="random_forest",
        help="Which trained model to serve (default: random_forest).",
    )
    parser.add_argument("--share", action="store_true", help="Request a public Gradio share link.")
    parser.add_argument(
        "--port", type=int, default=None, help="Server port (default: 7860 or GRADIO_SERVER_PORT)."
    )
    parser.add_argument("--log-level", default=None, help="Override MNIST_LOG_LEVEL.")
    return parser.parse_args(argv)


def serve_cli(argv: list[str] | None = None) -> int:
    """Entrypoint for `mnist-serve` / `python scripts/serve.py`."""
    from mnist_digit_recognition.app import (
        launch,  # local import: gradio is a heavy/optional dep for serving only
    )

    args = _parse_serve_args(argv)
    config = load_config()
    setup_logging(args.log_level or config.log_level)

    if args.share:
        config = _with_overrides(config, gradio_share=True)
    if args.port:
        config = _with_overrides(config, gradio_server_port=args.port)

    launch(config, ModelName(args.model))
    return 0


def _with_overrides(config, **overrides):
    from dataclasses import replace

    return replace(config, **overrides)


def main_train() -> None:  # console_script entrypoint
    sys.exit(train_cli())


def main_serve() -> None:  # console_script entrypoint
    sys.exit(serve_cli())


if __name__ == "__main__":
    sys.exit(train_cli())
