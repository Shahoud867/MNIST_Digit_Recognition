from __future__ import annotations

from pathlib import Path

from mnist_digit_recognition.config import load_config


def test_load_config_defaults(monkeypatch):
    monkeypatch.delenv("MNIST_MODEL_DIR", raising=False)
    monkeypatch.delenv("MNIST_LOG_LEVEL", raising=False)

    config = load_config()

    assert config.log_level == "INFO"
    assert config.data.test_size == 10_000
    assert config.gradio_share is False
    assert config.gradio_server_name == "127.0.0.1"


def test_load_config_reads_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MNIST_MODEL_DIR", str(tmp_path / "custom_models"))
    monkeypatch.setenv("MNIST_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("GRADIO_SHARE", "1")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")

    config = load_config()

    assert config.model_dir == Path(tmp_path / "custom_models")
    assert config.log_level == "DEBUG"
    assert config.gradio_share is True
    assert config.gradio_server_name == "0.0.0.0"


def test_ensure_directories_creates_missing_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("MNIST_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("MNIST_REPORTS_DIR", str(tmp_path / "reports"))

    config = load_config()
    config.ensure_directories()

    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "reports").is_dir()
