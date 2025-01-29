
import h5py
import json
import numpy as np
import os
from datetime import date
import inspect



class SurrogateModel:
    def __init__(self):
        self._X_train = None
        self._y_train = None
        self.predict = None
        self._model = None

    def set(self, X, y, model)->None:
        self._X_train = X
        self._y_train = y
        self._model = model


    def train(self)->None:
        """
        Train the model with data
        """
        self._model.fit(self._X_train, self._y_train)

    @property
    def X_train(self)->np.ndarray :
        """Return X.train"""
        return self._X_train

    @property
    def y_train(self)->np.ndarray:
        """Return y.train"""
        return self._y_train

    @property
    def model(self)-> any:
        """Return model"""
        return self._model

    def prediction(self, X)->None:
        """
        Predict value for data X.
        """

        if not self._model:
            raise ValueError("The model was not set or load.")
        try:
            self.predict = self._model.predict(X)
        except:
            raise ValueError('Model was not trained !')


    def _get_model_library(self)-> dict:
        """
        Detect the library used for the model
        """
        library = type(self._model).__module__.split(".")[0]
        version = __import__(library).__version__
        return {library: version}

    def dump(self, output_dir)->None:
        """
        save data, the instance of the model and the dependencies in a hdf5 file.
        :param output_dir: folder path where to save the hdf5 file
        """
        os.makedirs(output_dir, exist_ok=True)

        d = date.today()

        h5_path = os.path.join(output_dir, f"{type(self.model).__name__}_{d}.h5")
        model_type = type(self.model).__name__
        obj = globals().get(model_type, None)
        module = inspect.getmodule(obj)
        module_imported = module.__name__
        with h5py.File(h5_path, "w") as f:
            if self.X_train is not None:
                f.create_dataset("X_train", data=self.X_train)
            if self.y_train is not None:
                f.create_dataset("y_train", data=self.y_train)
            if self.predict is not None:
                f.create_dataset("y_predict", data = self.predict)

            model_group = f.create_group("model")
            model_group.attrs['type'] = model_type
            model_group.attrs['Imported'] = module_imported
            model_group.attrs['params'] = json.dumps(self._model.get_params())

            # Save library and version information
            library_info = self._get_model_library()
            model_group.attrs["library"] = json.dumps(library_info)
        print(f"Data of the model are saved in {h5_path}.")

    def load(self, path)->None:
        """
        load data, instance and dependencies from a folder.
        :param output_dir: folder path with files to load.
        """

        for file in os.listdir(path):
            if file.endswith('.h5'):
                file_h5 = os.path.join(path, file)
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

if __name__ == '__main__':
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
    # Initialisation du modèle et des données
    surrogate_model = SurrogateModel()

    # Données d'entraînement cohérentes
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])

    surrogate_model.set(X, y, model)
    surrogate_model.train()

    # Prédiction sur de nouvelles données
    X_new = np.array([[2, 3, 4], [5, 6, 7]])
    surrogate_model.prediction(X_new)

    # Sauvegarde des données
    surrogate_model.dump("./model_data")
    #from sklearn.ensemble import RandomForestRegressor
    from catboost import CatBoostRegressor
    #from sklearn.tree import DecisionTreeClassifier
    # Créer une instance de modèle (par exemple, RandomForestRegressor)


    #model = DecisionTreeClassifier()
    surrogate_model = SurrogateModel()
    X =[[0.76431152 ,0.66447784 ,0.40252745],
        [0.77784719 ,0.2680739 , 0.51301999],
        [0.37428748, 0.36702566 ,0.8352832 ],
        [0.84944394, 0.52850647 ,0.28959054],
        [0.04679972 ,0.24170513, 0.75792477]]
    y = [0.3678914 , 0.3134304,  0.53511834 ,0.95278193, 0.04955847]

    surrogate_model.set(X, y, model)
    surrogate_model.train()
    X_new =[[0.81176762 ,0.60318144, 0.94496069],
             [0.64052046, 0.42480091 ,0.59984412],
             [0.18545746, 0.6309216,  0.48833392],
             [0.32811995 ,0.72646501, 0.41143818],
             [0.51717996 ,0.85007028, 0.16759747]]

    surrogate_model.prediction(X_new)

    # Sauvegarder dans un dossier
    surrogate_model.dump("./model_data")
    test =[0.45523601 ,0.39392024 ,0.37198506 ,0.40304345 ,0.48881252]
    # Charger le modèle et les données depuis le dossier

    loaded_model = SurrogateModel()
    loaded_model.load("./model_data")
    # Prédire avec le modèle chargé
    X_new = np.random.rand(5, 10)
    loaded_model.train()
    loaded_model.prediction(X_new)
    print("Prédictions :", loaded_model.predict)







