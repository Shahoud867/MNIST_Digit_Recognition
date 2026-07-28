# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-07-28

### Added
- Proper Python package (`src/mnist_digit_recognition/`) with typed,
  documented, and tested modules: `config`, `data`, `models`, `evaluate`,
  `inference`, `app`, `cli`, `logging_config`.
- `pytest` test suite covering data splitting, model persistence, and
  inference — including a regression test for the scaler bug (see Fixed).
- GitHub Actions CI (lint + test on every push/PR), Dependabot config.
- `Dockerfile` for containerized serving of the Gradio demo.
- `.pre-commit-config.yaml` (black, ruff, nbstripout, basic hygiene hooks).
- Full GitHub community health files: LICENSE (MIT), CONTRIBUTING.md,
  CODE_OF_CONDUCT.md, SECURITY.md, issue templates, PR template.
- CLI entrypoints `mnist-train` and `mnist-serve` (also runnable directly
  via `scripts/train.py` / `scripts/serve.py`).
- Environment-based configuration (`.env.example`) replacing hardcoded
  hyperparameters and paths.
- `reports/RESULTS.md` generation with per-model accuracy and confusion
  matrices.

### Changed
- Restructured the project from a single notebook into
  `src/` (library code) + `notebooks/` (exploratory analysis, now
  importing from `src/` instead of duplicating logic) + `scripts/` (CLI).
- Rewrote `README.md` with installation, usage, architecture, results,
  and badges.
- Pinned all runtime and dev dependencies to exact versions
  (`requirements.txt`, `requirements-dev.txt`).

### Security
- Upgraded `gradio` (4.29.0 → 6.20.0), `pillow` (10.3.0 → 12.3.0), and
  `scikit-learn` (1.4.2 → 1.7.2) to versions with no known CVEs per
  `pip-audit` (the original pins carried 50+ known advisories, mostly in
  the Gradio 4.x line). CI now runs `pip-audit` on every push.

### Fixed
- **Scaler bug**: the original notebook fit, saved, and reloaded a
  `StandardScaler` but never applied it inside `predict_digit()` before
  calling the model — silently correct only by coincidence because
  RandomForest was trained on unscaled data. `ModelBundle.needs_scaling`
  now makes this explicit and is covered by a regression test
  (`tests/test_inference.py::test_predict_with_scaled_model_applies_scaler`).
- Missing input validation on uploaded images (unsupported types,
  zero-sized, or oversized images) now raises a clear `InvalidImageError`
  instead of crashing the Gradio app.

## [0.1.0] - 2025 (original notebook)

- Initial notebook: MNIST data loading, SGD/RandomForest training,
  evaluation, misclassification analysis, and a Gradio demo.
