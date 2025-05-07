# revivAl

![code coverage](https://raw.githubusercontent.com/eurobios-mews-labs/revivAl/coverage-badge/coverage.svg?raw=true)

A python package to save and reuse AI models.

## Install

`pip install revival@git+https://github.com/eurobios-mews-labs/revival.git`

## Simple usage

* **Save AI model**: train the model,predict and save the results

```python
import numpy as np
from catboost import CatBoostRegressor
from revival import LiteModel
# Initialise model and data
model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
y = np.array([3, 6, 9, 12, 15])

#Instanciate LiteModel
surrogate_model = LiteModel()
#set X, y and model
surrogate_model.set(X, y, model)
#fit the model
surrogate_model.train()

# Prediction
X_new = np.array([[2, 3, 4], [5, 6, 7]])
y_pred = surrogate_model.predict(X_new)

#Save the trained model
surrogate_model.dump("./model_data",'model_file')
```

The result of this example is a HDF5 file where the information about the model used are stored

* **Reused the stored AI model** Load the stored model and reuse it.

```python
from revival import LiteModel
import numpy as np

loaded_model = LiteModel()
loaded_model.load("./model_data", 'model_file')
X_new = np.array([[1, 4, 2], [7, 2, 1], [3, 3, 9], [4, 10, 11], [12, 3, 11]])
y_pred = loaded_model.predict(X_new)
```

The stored model is loaded.
