
from revivai.base import LiteModel
import numpy as np


def test_X():
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
    lite_model = LiteModel()

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])
    lite_model.set(X, y, model)
    lite_model.train()

    X_new = np.array([[2, 3, 4], [5, 6, 7]])

    lite_model.prediction(X_new)
    path = './metamodel_data'
    lite_model.dump(path,"cat_boost")

    load_model = LiteModel()
    load_model.load(path, "cat_boost.h5")

    assert np.allclose(load_model.X_train,lite_model.X_train), "X_train data are not matching."

def test_y():
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
    lite_model = LiteModel()

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])
    lite_model.set(X, y, model)
    lite_model.train()

    X_new = np.array([[2, 3, 4], [5, 6, 7]])

    lite_model.prediction(X_new)
    path = './metamodel_data'
    lite_model.dump(path,"cat_boost")

    load_model = LiteModel()
    load_model.load(path, "cat_boost.h5")

    assert np.allclose(load_model.y_train,lite_model.y_train), "y_train data are not matching."

def test_prediction():
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
    lite_model = LiteModel()

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([3, 6, 9, 12, 15])
    lite_model.set(X, y, model)
    lite_model.train()

    X_new = np.array([[2, 3, 4], [5, 6, 7]])

    lite_model.prediction(X_new)
    path = './metamodel_data'
    lite_model.dump(path, "cat_boost")

    load_model = LiteModel()
    load_model.load(path, "cat_boost.h5")

    assert np.allclose(load_model.predict, lite_model.predict), "Prediction data are not matching."

def test_save_multi_output_model():
    from catboost import CatBoostRegressor
    from sklearn import multioutput
    import pandas as pd
    model = multioutput.MultiOutputRegressor(CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42))
    surrogate_model = LiteModel()
    X = pd.DataFrame({"tension":[1.1324,1.345,1.2431,1.6452],"amplitude":[4.1324,4.345,5.2431,4.6452]})

    y = pd.DataFrame({'min':[0.1344,0.4325,0.1465,0.2344], '40%':[0.2453,0.3456,0.3654,0.1234]})
    surrogate_model.set(X, y, model)
    surrogate_model.train()
    surrogate_model.prediction(X)
    path = './metamodel_data'
    surrogate_model.dump(path,'file_test')
    load_model = LiteModel()
    load_model.load(path, 'file_test.h5')

   