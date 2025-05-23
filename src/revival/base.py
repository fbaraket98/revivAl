# Copyright MewsLabs 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import io
import json
import os
import tempfile
from datetime import date

import h5py
import joblib
import numpy as np
import pandas as pd
import tensorflow.keras.models as krs_models
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor


class LiteModel:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model):
        self._X_train = X_train
        self._y_train = y_train
        self.prediction = pd.DataFrame()
        self._model = model
        self._fitted = False
        self.score = None
        self._is_multi = None
        self.hyperparams = None
        self.lib_name = None
        self.cls_name = None
        self.lib_name_classe = None
        self.est_params = None
        self.wrapp_params = None

    @property
    def X_train(self):
        return self._X_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        """Function that aims to train the model"""
        try:
            self._model.set_training_values(self._X_train, self._y_train)
            self._model.train()
        except AttributeError:
            try:
                self._model.train(self._X_train, self._y_train)
            except AttributeError:
                self._model.fit(self._X_train, self._y_train)
        self._fitted = True

    def get_if_classifier(self):
        model = (
            self._model.estimator
            if isinstance(self._model, (MultiOutputRegressor, MultiOutputClassifier))
            else self._model
        )
        is_classifier = (
            hasattr(model, "predict_proba")
            or hasattr(model, "_estimator_type")
            and model._estimator_type == "classifier"
        )
        return is_classifier

    def re_train(self):
        import importlib

        module = importlib.import_module(self.lib_name_classe)
        ModelClass = getattr(module, self.cls_name)
        if self._is_multi and self.get_if_classifier():
            self._model = MultiOutputClassifier(ModelClass(**self.est_params))

        elif self._is_multi and not self.get_if_classifier():
            self._model = MultiOutputRegressor(ModelClass(**self.est_params))
        else:
            self._model = ModelClass(**self.est_params)
        self.train()

    def get_model_info(self, X=None, y=None):
        """Function that prints the model used and its score"""

        # Multioutput wrapper: get inner model
        model = (
            self._model.estimator
            if isinstance(self._model, (MultiOutputRegressor, MultiOutputClassifier))
            else self._model
        )
        if self.score is None:
            X_eval = X if X is not None else self._X_train
            y_eval = y if y is not None else self._y_train

            y_pred = self.predict(X_eval).values
            y_true = y_eval

            # Detect classification (y must be categorical or discrete)
            is_classifier = self.get_if_classifier()
            if isinstance(y_true, pd.DataFrame):
                y_true = y_true.values

            if is_classifier:
                self.score = accuracy_score(y_true, y_pred)
            else:
                self.score = r2_score(y_true, y_pred)

        print("=" * 40)
        print(f"✨ Model used: {self.get_model_name()}")
        print(f"🔹 Features ({len(self.X_train.columns)}):")
        for feat in self.X_train.columns:
            print(f"   - {feat}")
        print(f"🎯 Targets ({len(self.y_train.columns)}):")
        for target in self.y_train.columns:
            print(f"   - {target}")
        print(f"✅ Model score: {self.score}")
        print("=" * 40)

    def get_model_name(self):
        """Function that returns the name of the model"""
        if self._is_multi:
            return type(self._model.estimator).__name__
        else:
            return type(self._model).__name__

    def predict(self, X_test: pd.DataFrame) -> None:

        if not self._model:
            raise ValueError("The model was not set or loaded.")
        try:
            self.prediction = self._model.predict(X_test)
        except:
            try:
                self.prediction = self._model.predict_values(X_test)
            except:
                self.re_train()
                try:
                    self.prediction = self._model.predict(X_test)
                except:
                    self.prediction = self._model.predict_values(X_test)
        if self.prediction.ndim == 3 and self.prediction.shape[0] == 1:
            self.prediction = self.prediction[0]

        self.prediction = pd.DataFrame(self.prediction)
        self.prediction.columns = pd.DataFrame(self.y_train).columns
        return self.prediction

    @staticmethod
    def _safe_serialize_params(params):
        def convert(v):
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            elif isinstance(v, (list, tuple)):
                return [convert(i) for i in v]
            elif isinstance(v, dict):
                return {k: convert(i) for k, i in v.items()}
            else:
                return str(v)

        return json.dumps(convert(params))

    def get_lib_name(self):
        self.is_multi_output()
        if self._is_multi:
            library = type(self._model.estimator).__module__.split(".")[0]
        else:
            library = type(self._model).__module__.split(".")[0]

        version = __import__(library).__version__
        return {library: version}

    def _get_model_library(self) -> dict:

        library = type(self._model).__module__.split(".")[0]
        version = __import__(library).__version__
        return {library: version}

    def is_multi_output(self):
        if isinstance(self._model, (MultiOutputRegressor, MultiOutputClassifier)):
            self._is_multi = True
        else:
            self._is_multi = False

    def _serialize_model(self) -> bytes:
        """Serialize any model as bytes"""

        buffer = io.BytesIO()
        if isinstance(self._model, krs_models.Model):
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                self._model.save(tmp.name)
                tmp.seek(0)
                buffer.write(tmp.read())
            os.remove(tmp.name)
        elif isinstance(self._model, (CatBoostClassifier, CatBoostRegressor)):
            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
                self._model.save_model(tmp.name)
                tmp.seek(0)
                buffer.write(tmp.read())
            os.remove(tmp.name)
        else:
            joblib.dump(self._model, buffer)
        return buffer.getvalue()

    def dump(self, output_dir: str, file_name: str = None) -> None:
        """Function that aims to save the trained model and its data
        Parameters
        ----------
        output_dir : str
            path where to save the file
        file_name : str
            name of the file where to save the model and its data
        """
        os.makedirs(output_dir, exist_ok=True)
        d = date.today().strftime("%Y%m%d")
        file_name = file_name if file_name else f"{type(self._model).__name__}_{d}"
        file_path = os.path.join(output_dir, f"{file_name}.h5")

        with h5py.File(file_path, "w") as f:
            if self._X_train is not None:
                if isinstance(self._X_train, pd.DataFrame):
                    f.create_dataset("X_train", data=self._X_train.values)
                    f.create_dataset(
                        "X_train_columns",
                        data=np.array(self._X_train.columns.astype(str), dtype="S"),
                    )
                    f.create_dataset(
                        "X_train_index",
                        data=np.array(self._X_train.index.astype(str), dtype="S"),
                    )
                else:
                    f.create_dataset("X_train", data=self._X_train)
            if self._y_train is not None:
                if isinstance(self._y_train, pd.DataFrame):
                    f.create_dataset("y_train", data=self._y_train.values)
                    f.create_dataset(
                        "y_train_columns",
                        data=np.array(self._y_train.columns.astype(str), dtype="S"),
                    )
                    f.create_dataset(
                        "y_train_index",
                        data=np.array(self._y_train.index.astype(str), dtype="S"),
                    )
                else:
                    f.create_dataset("y_train", data=self._y_train)

            if self.prediction is not None:
                f.create_dataset("y_predict", data=self.prediction)
            if self.score is not None:
                f.attrs["score"] = self.score

            f.attrs["fitted"] = self._fitted

            # Serialize and save the model as raw bytes
            model_data = self._serialize_model()
            f.create_dataset("model_data", data=np.void(model_data))

            model_group = f.create_group("model_meta")
            if isinstance(self._model, (MultiOutputRegressor, MultiOutputClassifier)):
                model_group.attrs["is_multi"] = True
                base_model = self._model.estimator
                model_group.attrs["wrapper_class"] = (
                    self._model.__class__.__module__
                    + "."
                    + self._model.__class__.__name__
                )
                model_group.attrs["wrapper_params"] = self._safe_serialize_params(
                    self._model.get_params(deep=False)
                )
                model_group.attrs["estimator_class"] = (
                    base_model.__class__.__module__
                    + "."
                    + base_model.__class__.__name__
                )
                model_group.attrs["estimator_params"] = self._safe_serialize_params(
                    base_model.get_params()
                )
            else:
                model_group.attrs["is_multi"] = False
                model_group.attrs["model_class"] = (
                    self._model.__class__.__module__
                    + "."
                    + self._model.__class__.__name__
                )
                try:
                    model_group.attrs["params"] = self._safe_serialize_params(
                        self._model.get_params()
                    )
                except AttributeError:
                    model_group.attrs["params"] = json.dumps({})

            model_group.attrs["library"] = json.dumps(self._get_model_library())
            model_group.attrs["lib_name_class"] = json.dumps(self.get_lib_name())
        print(f"Model and metadata saved to {file_path}")
