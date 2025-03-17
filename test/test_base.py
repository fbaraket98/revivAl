
from revivai.base import SurrogateModel
import numpy as np


def test_X():
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
    surrogate_model = SurrogateModel()

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])
    surrogate_model.set(X, y, model)
    surrogate_model.train()

    X_new = np.array([[2, 3, 4], [5, 6, 7]])

    surrogate_model.prediction(X_new)
    path = './metamodel_data'
    surrogate_model.dump(path,"cat_boost")

    load_model = SurrogateModel()
    load_model.load(path, "cat_boost.h5")

    assert np.allclose(load_model.X_train,surrogate_model.X_train), "X_train data are not matching."

def test_y():
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
    surrogate_model = SurrogateModel()

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])
    surrogate_model.set(X, y, model)
    surrogate_model.train()

    X_new = np.array([[2, 3, 4], [5, 6, 7]])

    surrogate_model.prediction(X_new)
    path = './metamodel_data'
    surrogate_model.dump(path,"cat_boost")

    load_model = SurrogateModel()
    load_model.load(path, "cat_boost.h5")

    assert np.allclose(load_model.y_train,surrogate_model.y_train), "y_train data are not matching."

def test_prediction():
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
    surrogate_model = SurrogateModel()

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])
    surrogate_model.set(X, y, model)
    surrogate_model.train()

    X_new = np.array([[2, 3, 4], [5, 6, 7]])

    surrogate_model.prediction(X_new)
    path = './metamodel_data'
    surrogate_model.dump(path, "cat_boost")

    load_model = SurrogateModel()
    load_model.load(path, "cat_boost.h5")

    assert np.allclose(load_model.predict, surrogate_model.predict), "Prediction data are not matching."




