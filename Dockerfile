# syntax=docker/dockerfile:1
FROM python:3.14-slim AS base

# Prevent Python from writing .pyc files / buffering stdout, keep pip lean.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps required by scikit-image/Pillow for image decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir --no-deps -e .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

ENV MNIST_MODEL_DIR=/app/models \
    MNIST_REPORTS_DIR=/app/reports \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SHARE=0

EXPOSE 7860

# The image serves a pre-trained model. Mount/copy trained artifacts into
# /app/models (see README "Docker" section) before running, or run
# `docker run ... python scripts/train.py` first to populate them.
CMD ["python", "scripts/serve.py", "--model", "random_forest"]
