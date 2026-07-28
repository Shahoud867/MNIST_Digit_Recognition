#!/usr/bin/env python
"""Standalone serving entrypoint: ``python scripts/serve.py [--model ...]``.

Launches the Gradio demo using artifacts produced by ``scripts/train.py``.
See ``mnist_digit_recognition.cli.serve_cli`` for the actual implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mnist_digit_recognition.cli import serve_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(serve_cli())
