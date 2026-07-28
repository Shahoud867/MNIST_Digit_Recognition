# Contributing

Thanks for considering a contribution to MNIST Digit Recognition.

## Getting started

```bash
git clone https://github.com/Shahoud867/MNIST_Digit_Recognition.git
cd MNIST_Digit_Recognition
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Development workflow

1. Create a branch: `git checkout -b feature/short-description`.
2. Make your change. Keep it focused — one logical change per PR.
3. Run the checks locally before pushing:
   ```bash
   ruff check .
   black --check .
   pytest
   ```
4. Commit with a clear, imperative message (e.g. `Fix scaler not applied for SGD inference`).
5. Open a pull request using the PR template. Link any related issue.

## Code style

- Formatting: [black](https://github.com/psf/black) (line length 100).
- Linting: [ruff](https://github.com/astral-sh/ruff).
- Type hints are expected on new public functions.
- Prefer small, composable functions over large ones; see `src/mnist_digit_recognition/` for the existing module boundaries (data / models / evaluate / inference / app).

## Tests

- New behavior needs a test in `tests/`.
- Tests must not require network access or download real MNIST data — use the `synthetic_dataset` / `synthetic_image` fixtures in `tests/conftest.py`.
- Run the full suite with `pytest` (coverage report prints automatically).

## Reporting bugs / requesting features

Please use the [issue templates](.github/ISSUE_TEMPLATE/) — they ask for the information needed to reproduce or evaluate a request.

## Code of Conduct

Participation in this project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
