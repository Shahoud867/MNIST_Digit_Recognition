from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier

from mnist_digit_recognition.app import build_interface, build_predict_fn
from mnist_digit_recognition.models import ModelBundle, ModelName


def test_build_interface_does_not_raise(synthetic_dataset):
    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    bundle = ModelBundle(
        name=ModelName.RANDOM_FOREST, model=model, needs_scaling=False, scaler=None
    )

    interface = build_interface(bundle)

    assert interface is not None


def test_predict_fn_raises_gradio_error_on_missing_image(synthetic_dataset):
    import gradio as gr

    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    bundle = ModelBundle(
        name=ModelName.RANDOM_FOREST, model=model, needs_scaling=False, scaler=None
    )
    predict_fn = build_predict_fn(bundle)

    with pytest.raises(gr.Error):
        predict_fn(None, False)


def test_predict_fn_raises_gradio_error_on_invalid_image(synthetic_dataset):
    import gradio as gr
    import numpy as np

    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    bundle = ModelBundle(
        name=ModelName.RANDOM_FOREST, model=model, needs_scaling=False, scaler=None
    )
    predict_fn = build_predict_fn(bundle)

    with pytest.raises(gr.Error):
        predict_fn(np.zeros((5000, 5000), dtype=np.uint8), False)
