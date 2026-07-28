"""Application-wide logging setup.

The original notebook used bare ``print()`` statements for all
diagnostics. This module gives every part of the package structured,
leveled logging instead, so failures and progress are debuggable in both
local runs and CI/production logs.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger exactly once per process.

    Safe to call multiple times (e.g. from both a CLI entrypoint and a
    test fixture) — subsequent calls only adjust the log level.
    """
    global _CONFIGURED
    root = logging.getLogger("mnist_digit_recognition")

    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True

    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the package's logging tree."""
    return logging.getLogger(f"mnist_digit_recognition.{name}")
