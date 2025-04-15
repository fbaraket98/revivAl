import h5py
import json
import os
from datetime import date


class LiteModel:
    def __init__(self):

        self._X_train = None
        self._y_train = None
        self.predict = None
        self._model = None

    def set(self, X, y, model) -> None:
        self._X_train = X
        self._y_train = y
        self._model = model

    def train(self) -> None:
        """
        Train the model with data
        """
        try:
            self._model.set_training_values(self._X_train, self._y_train)
            self._model.train()
        except AttributeError:
            try:
                self._model.train(self._X_train, self._y_train)
            except AttributeError:

                self._model.fit(self._X_train, self._y_train)

    @property
    def X_train(self) -> any:
        """Return X.train"""
        return self._X_train

    @property
    def y_train(self) -> any:
        """Return y.train"""
        return self._y_train

    @property
    def model(self) -> any:
        """Return model"""
        return self._model

    def prediction(self, X) -> None:
        """
        Predict value for data X.
        """

        if not self._model:
            raise ValueError("The model was not set or load.")
        try:
            self.predict = self._model.predict(X)
        except:
            try:
                self.predict = self._model.predict_values(X)
            except:
                raise ValueError('Model was not trained !')

    def _get_model_library(self) -> dict:
        """
        Detect the library used for the model
        """
        library = type(self._model).__module__.split(".")[0]
        version = __import__(library).__version__
        return {library: version}

    def dump(self, output_dir, file_name=None) -> None:
        """
        save data, the instance of the model and the dependencies in a hdf5 file.
        :param output_dir: folder path where to save the hdf5 file
        :param file_name: HDF5 file name where to save the data of the model
        """
        os.makedirs(output_dir, exist_ok=True)
        d = date.today()
        if not file_name:
            h5_path = os.path.join(output_dir, f"{type(self._model).__name__}_{d}.h5")
        else:
            h5_path = os.path.join(output_dir, f"{file_name}.h5")
        model_type = type(self._model).__name__
        module_imported = self._model.__class__.__module__
        with h5py.File(h5_path, "w") as f:
            if self.X_train is not None:
                f.create_dataset("X_train", data=self.X_train)
            if self.y_train is not None:
                f.create_dataset("y_train", data=self.y_train)
            if self.predict is not None:
                f.create_dataset("y_predict", data=self.predict)

            model_group = f.create_group("model")
            model_group.attrs['type'] = model_type
            model_group.attrs['Imported'] = module_imported
            try:
                model_group.attrs['params'] = self._safe_serialize_params(self._model.get_params())
            except AttributeError:
                model_group.attrs['params'] = json.dumps({})
            # Save library and version information
            library_info = self._get_model_library()
            model_group.attrs["library"] = json.dumps(library_info)
        print(f"Data of the model are saved in {h5_path}.")

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
                return str(v)  # fallback: stringify unsupported types

        return json.dumps(convert(params))

    def load(self, path, file_name: str) -> None:
        """
        load data, instance and dependencies from a folder.
        :param path: folder path with files to load.
        :param file_name: HDF5 file name to load
        """

        file_h5 = os.path.join(path, file_name)
        with h5py.File(file_h5, "r") as f:
            self._X_train = f["X_train"][()] if "X_train" in f else None
            self._y_train = f["y_train"][()] if "y_train" in f else None
            self.predict = f['y_predict'][()] if "y_predict" in f else None
            model_group = f["model"]
            model_type = model_group.attrs["type"]
            model_params = json.loads(model_group.attrs["params"])
            module_name = model_group.attrs["Imported"]
            library = model_group.attrs["library"]
            try:
                model_class = getattr(__import__(module_name, fromlist=[model_type]), model_type)
                self._model = model_class(**model_params)
            except:
                raise ValueError(f'You must install the package of {module_name}')
        print(f"Data is loaded from {file_h5}.")
