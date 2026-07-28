#!/usr/bin/env python
"""Standalone training entrypoint: ``python scripts/train.py [--model ...]``.

Thin wrapper so the project is runnable without a `pip install -e .`
step. See ``mnist_digit_recognition.cli.train_cli`` for the actual
implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mnist_digit_recognition.cli import train_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(train_cli())
