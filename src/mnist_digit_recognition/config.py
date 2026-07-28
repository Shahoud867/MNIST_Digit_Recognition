"""Centralized, environment-overridable configuration.

Every tunable value that was previously hardcoded inline in the notebook
(dataset version, split ratios, model hyperparameters, artifact paths)
lives here so experiments are reproducible and scriptable without editing
source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


@dataclass(frozen=True)
class DataConfig:
    """Controls how the MNIST dataset is fetched and split."""

    openml_name: str = "mnist_784"
    openml_version: int = 1
    test_size: int = 10_000
    random_state: int = 42


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class SGDConfig:
    loss: str = "hinge"
    max_iter: int = 1_000
    tol: float = 1e-3
    random_state: int = 42


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration, overridable via environment variables.

    Environment variables (all optional):
        MNIST_MODEL_DIR   — directory for trained model/scaler artifacts.
        MNIST_REPORTS_DIR — directory for generated reports/figures.
        MNIST_LOG_LEVEL   — Python logging level name (e.g. "DEBUG").
        MNIST_RANDOM_SEED — overrides the random seed for data + models.
        GRADIO_SHARE      — "1" to request a public Gradio share link.
        GRADIO_SERVER_PORT — port for the Gradio server (default 7860).
        GRADIO_SERVER_NAME — bind address for the Gradio server (default
            "127.0.0.1", loopback-only; the Docker image sets this to
            "0.0.0.0" so the server is reachable from outside the container).
    """

    model_dir: Path = field(
        default_factory=lambda: _env_path("MNIST_MODEL_DIR", PROJECT_ROOT / "models")
    )
    reports_dir: Path = field(
        default_factory=lambda: _env_path("MNIST_REPORTS_DIR", PROJECT_ROOT / "reports")
    )
    log_level: str = field(default_factory=lambda: _env_str("MNIST_LOG_LEVEL", "INFO"))
    random_seed: int = field(default_factory=lambda: _env_int("MNIST_RANDOM_SEED", 42))
    gradio_share: bool = field(default_factory=lambda: _env_str("GRADIO_SHARE", "0") == "1")
    gradio_server_port: int = field(default_factory=lambda: _env_int("GRADIO_SERVER_PORT", 7860))
    gradio_server_name: str = field(
        default_factory=lambda: _env_str("GRADIO_SERVER_NAME", "127.0.0.1")
    )

    data: DataConfig = field(default_factory=DataConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    sgd: SGDConfig = field(default_factory=SGDConfig)

    def ensure_directories(self) -> None:
        """Create model/report output directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    """Build an :class:`AppConfig` from environment variables and defaults."""
    return AppConfig()
