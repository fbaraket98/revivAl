import importlib
import json
import os
import h5py
import joblib
import io
import tensorflow.keras.models as krs_models
from datetime import date
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
import tempfile
import numpy as np


class LiteModel:
    def __init__(self):
        self._X_train = None
        self._y_train = None
        self.predict = None
        self._model = None
        self._fitted = False

    @property
    def X_train(self):
        return self._X_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def model(self):
        return self._model

    def set(self, X, y, model) -> None:
        self._X_train = X
        self._y_train = y
        self._model = model

    def train(self) -> None:
        try:
            self._model.set_training_values(self._X_train, self._y_train)
            self._model.train()
        except AttributeError:
            try:
                self._model.train(self._X_train, self._y_train)
            except AttributeError:
                self._model.fit(self._X_train, self._y_train)
        self._fitted = True

    def prediction(self, X) -> None:
        if not self._fitted:
            self.train()
        if not self._model:
            raise ValueError("The model was not set or loaded.")
        try:
            self.predict = self._model.predict(X)
        except:
            try:
                self.predict = self._model.predict_values(X)
            except:
                raise ValueError("Model was not trained!")

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
            krs_models.save_model(self._model, buffer, save_format='h5')
        elif isinstance(self._model, (CatBoostClassifier, CatBoostRegressor)):
            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
                self._model.save_model(tmp.name)
                tmp.seek(0)
                buffer.write(tmp.read())
            os.remove(tmp.name)
        else:
            joblib.dump(self._model, buffer)
        return buffer.getvalue()

    def _deserialize_model(self, buffer: bytes, lib: str, class_name: str):
        buffer_io = io.BytesIO(buffer)

        if lib in ['keras', 'tensorflow']:
            return krs_models.load_model(buffer_io)

        elif lib == 'catboost':
            cls = getattr(importlib.import_module("catboost"), class_name)
            model = cls()

            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
                tmp.write(buffer_io.read())
                tmp.flush()
                model.load_model(tmp.name)
            os.remove(tmp.name)
            return model

        else:
            return joblib.load(buffer_io)

    def dump(self, output_dir, file_name=None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        d = date.today().strftime("%Y%m%d")
        file_name = file_name if file_name else f"{type(self._model).__name__}_{d}"
        file_path = os.path.join(output_dir, f"{file_name}.h5")

        with h5py.File(file_path, "w") as f:
            if self._X_train is not None:
                f.create_dataset("X_train", data=self._X_train)
            if self._y_train is not None:
                f.create_dataset("y_train", data=self._y_train)
            if self.predict is not None:
                f.create_dataset("y_predict", data=self.predict)
            f.attrs["fitted"] = self._fitted

            # Serialize and save the model as raw bytes
            model_data = self._serialize_model()
            f.create_dataset("model_data", data=np.void(model_data))

            model_group = f.create_group("model_meta")
            if isinstance(self._model, (MultiOutputRegressor, MultiOutputClassifier)):
                model_group.attrs["is_multi"] = True
                base_model = self._model.estimator
                model_group.attrs["wrapper_class"] = (
                        self._model.__class__.__module__ + "." + self._model.__class__.__name__
                )
                model_group.attrs["wrapper_params"] = self._safe_serialize_params(
                    self._model.get_params(deep=False)
                )
                model_group.attrs["estimator_class"] = (
                        base_model.__class__.__module__ + "." + base_model.__class__.__name__
                )
                model_group.attrs["estimator_params"] = self._safe_serialize_params(
                    base_model.get_params()
                )
            else:
                model_group.attrs["is_multi"] = False
                model_group.attrs["model_class"] = (
                        self._model.__class__.__module__ + "." + self._model.__class__.__name__
                )
                try:
                    model_group.attrs["params"] = self._safe_serialize_params(self._model.get_params())
                except AttributeError:
                    model_group.attrs["params"] = json.dumps({})

            model_group.attrs["library"] = json.dumps(self._get_model_library())

        print(f"Model and metadata saved to {file_path}")

    def load(self, path, file_name: str) -> None:
        file_path = os.path.join(path, f"{file_name}.h5")
        with h5py.File(file_path, "r") as f:
            self._X_train = f["X_train"][()] if "X_train" in f else None
            self._y_train = f["y_train"][()] if "y_train" in f else None
            self.predict = f["y_predict"][()] if "y_predict" in f else None
            self._fitted = f.attrs.get("fitted", False)

            # Load model bytes and metadata
            model_data = bytes(f["model_data"][()])
            model_meta = f["model_meta"]
            library = json.loads(model_meta.attrs["library"])
            lib_name = next(iter(library.keys()))
            model_class = model_meta.attrs["estimator_class"] if model_meta.attrs.get("is_multi", False) \
                else model_meta.attrs["model_class"]
            lib_name = next(iter(json.loads(model_meta.attrs["library"]).keys()))
            cls_name = model_class.split(".")[-1]

            self._model = self._deserialize_model(model_data, lib_name, cls_name)

        print(f"Full model loaded from {file_path}")
