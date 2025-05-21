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

from utils.utils import deserialize_model


class LiteModel:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model):
        self._X_train = X_train
        self._y_train = y_train
        self.prediction = pd.DataFrame()
        self._model = model
        self._fitted = False
        self.score = None

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

    def get_model_info(self, X=None, y=None):
        """Function that prints the model used and its score"""

        # Multioutput wrapper: get inner model
        model = (
            self._model.estimator.model
            if isinstance(self._model, (MultiOutputRegressor, MultiOutputClassifier))
            else self._model
        )
        if self.score is None:
            X_eval = X if X is not None else self._X_train
            y_eval = y if y is not None else self._y_train

            y_pred = self.predict(X_eval).values
            y_true = y_eval

            # Detect classification (y must be categorical or discrete)
            is_classifier = (
                hasattr(model, "predict_proba")
                or hasattr(model, "_estimator_type")
                and model._estimator_type == "classifier"
            )
            if isinstance(y_true, pd.DataFrame):
                y_true = y_true.values

            if is_classifier:
                self.score = accuracy_score(y_true, y_pred)
            else:
                self.score = r2_score(y_true, y_pred)

        print(f"The model used is : {self.get_model_name(model)}")
        print(f"The score of the model is {self.score}")

    @staticmethod
    def get_model_name(model):
        """Function that returns the name of the model"""
        if hasattr(model, "name"):
            return model.name
        elif hasattr(model, "__class__"):
            return model.__class__.__name__
        else:
            return str(type(model))

    def predict(self, X_test: pd.DataFrame) -> None:
        if not self._model:
            raise ValueError("The model was not set or loaded.")
        try:
            self.prediction = self._model.predict(X_test)
        except:
            try:
                self.prediction = self._model.predict_values(X_test)
            except:
                raise ValueError("Model was not trained!")
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

    def _get_model_library(self) -> dict:
        library = type(self._model).__module__.split(".")[0]
        version = __import__(library).__version__
        return {library: version}

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

        print(f"Model and metadata saved to {file_path}")

    @staticmethod
    def load(path: str, file_name: str) -> None:
        """Function that loads a trained model and its data from a hdf5 file.
        Parameters:
        ----------
        path: str
            the path where the file is saved
        file_name: str
            the name of the file where the model is saved
        """
        file_path = os.path.join(path, f"{file_name}.h5")
        with h5py.File(file_path, "r") as f:
            if "X_train" in f:
                X_data = f["X_train"][()]
                if "X_train_columns" in f and "X_train_index" in f:
                    columns = [col.decode("utf-8") for col in f["X_train_columns"][()]]
                    index = [idx.decode("utf-8") for idx in f["X_train_index"][()]]
                    _X_train = pd.DataFrame(X_data, columns=columns, index=index)
                else:
                    _X_train = X_data
            if "y_train" in f:
                y_data = f["y_train"][()]
                if "y_train_columns" in f and "y_train_index" in f:
                    columns = [col.decode("utf-8") for col in f["y_train_columns"][()]]
                    index = [idx.decode("utf-8") for idx in f["y_train_index"][()]]
                    _y_train = pd.DataFrame(y_data, columns=columns, index=index)
                else:
                    _y_train = y_data

            prediction = f["y_predict"][()] if "y_predict" in f else None
            score = f.attrs.get("score")
            # Load model bytes and metadata
            model_data = bytes(f["model_data"][()])
            model_meta = f["model_meta"]
            library = json.loads(model_meta.attrs["library"])
            lib_name = next(iter(library.keys()))
            model_class = (
                model_meta.attrs["estimator_class"]
                if model_meta.attrs.get("is_multi", False)
                else model_meta.attrs["model_class"]
            )
            lib_name = next(iter(json.loads(model_meta.attrs["library"]).keys()))
            cls_name = model_class.split(".")[-1]

            _model = deserialize_model(model_data, lib_name, cls_name)

        print(f"Full model loaded from {file_path}")
        litemodel = LiteModel(X_train=_X_train, y_train=_y_train, model=_model)
        litemodel.score = score
        litemodel.prediction = prediction
        return litemodel
