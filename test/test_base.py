import os
from revivAI import base
import numpy as np
import inspect


def test_dump():
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
    # Initialise the model and the data
    surrogate_model = base.SurrogateModel()

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])

    surrogate_model.set(X, y, model)
    surrogate_model.train()

    # Prediction
    X_new = np.array([[2, 3, 4], [5, 6, 7]])
    surrogate_model.prediction(X_new)

    # Assert
    assert np.allclose(surrogate_model.X_train, X), "Les données X_train ne correspondent pas."
    assert np.allclose(surrogate_model.y_train, y), "Les données y_train ne correspondent pas."
    assert surrogate_model.model == model, "Le modèle n'a pas été correctement défini."

    # save the model
    surrogate_model.dump("./model_data")

    # Assert
    assert os.path.exists('./model_data'), "Le répertoire model_data n'existe pas."
    assert os.path.exists('./model_data/CatBoostRegressor_2025-01-22.h5'), "Le fichier HDF5 n'a pas été généré."


def test_load():
    # Data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])

    #load the model
    loaded_model = base.SurrogateModel()
    loaded_model.load("./model_data")

    # Assert
    assert np.allclose(loaded_model.X_train, X), "Les données X_train chargées ne correspondent pas."
    assert np.allclose(loaded_model.y_train, y), "Les données y_train chargées ne correspondent pas."


if __name__ == "__main__":
    test_dump()
    test_load()

