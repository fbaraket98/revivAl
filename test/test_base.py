# Copyright MewsLabs 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import tempfile

import numpy as np
import pytest
from catboost import CatBoostRegressor
import pandas as pd
from revival import LiteModel, load_model


@pytest.fixture
def lite_model():
    model = CatBoostRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        allow_writing_files=False,
        silent=True,
    )

    X = pd.DataFrame(
        {
            "tension": [1.1324, 1.345, 1.2431, 1.6452],
            "amplitude": [4.1324, 4.345, 5.2431, 4.6452],
        }
    )

    y = pd.DataFrame(
        {
            "min": [0, 0, 1, 0],

        }
    )
    lite_model = LiteModel(X, y, model)
    lite_model.train()
    X_new = pd.DataFrame(
        {
            "tension": [1.3564, 1.7845, 1.6934, 1.12345],
            "amplitude": [4.325, 5.1234, 4.134, 4.7542],
        }
    )
    lite_model.predict(X_new)
    lite_model.get_model_info()

    return lite_model


def test_X(lite_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        lite_model.dump(tmpdir, "cat_boost")
        loaded_model = load_model(tmpdir, "cat_boost")

    assert np.allclose(
        loaded_model.X_train, lite_model.X_train
    ), "X_train data are not matching."


def test_y(lite_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        lite_model.dump(tmpdir, "cat_boost")
        loaded_model = load_model(tmpdir, "cat_boost")

    assert np.allclose(
        loaded_model.y_train, lite_model.y_train
    ), "y_train data are not matching."


def test_prediction(lite_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        lite_model.dump(tmpdir, "cat_boost")
        loaded_model = load_model(tmpdir, "cat_boost")
    assert np.allclose(
        loaded_model.prediction, lite_model.prediction
    ), "Prediction data are not matching."


def test_save_multi_output_model():
    from sklearn import multioutput
    import pandas as pd

    model = multioutput.MultiOutputRegressor(
        CatBoostRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            allow_writing_files=False,
            silent=True,
        )
    )
    X = pd.DataFrame(
        {
            "tension": [1.1324, 1.345, 1.2431, 1.6452],
            "amplitude": [4.1324, 4.345, 5.2431, 4.6452],
        }
    )

    y = pd.DataFrame(
        {
            "min": [0.1344, 0.4325, 0.1465, 0.2344],
            "40%": [0.2453, 0.3456, 0.3654, 0.1234],
        }
    )
    surrogate_model = LiteModel(X, y, model)

    surrogate_model.train()
    sr_prediction = surrogate_model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        surrogate_model.dump(tmpdir, "file_test")
        loaded_model = load_model(tmpdir, "file_test")
    load_prediction = loaded_model.prediction
    assert np.allclose(
        sr_prediction, load_prediction
    ), "Predictions are not matching for multioutput Regressor"
