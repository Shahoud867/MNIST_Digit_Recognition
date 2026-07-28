"""MNIST Digit Recognition — a small, production-shaped ML package.

Provides data loading, model training/evaluation, and inference utilities
for classifying handwritten digits from the MNIST dataset, plus a Gradio
web demo. See ``README.md`` for usage.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mnist-digit-recognition")
except PackageNotFoundError:  # pragma: no cover - local/dev install without build
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
