# revival
![code coverage](https://raw.githubusercontent.com/eurobios-mews-labs/revivAl/coverage-badge/coverage.svg?raw=true)

A python package to save and reuse AI models.

## Install

`pip install revial@git+https://github.com/eurobios-mews-labs/revival.git`

## Simple usage
* **Save AI model**: train the model,predict and save the results

```python
import numpy as np
from catboost import CatBoostRegressor

from revivai.base import SurrogateModel

model = CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42)
# Initialise the model and the data


X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
y = np.array([3, 6, 9, 12, 15])
surrogate_model = SurrogateModel()
surrogate_model.set(X, y, model)
surrogate_model.train()
# Prediction
X_new = np.array([[2, 3, 4], [5, 6, 7]])
surrogate_model.prediction(X_new)
surrogate_model.dump("./model_data")
```
The result of this example is, a HDF5 file where the information about the model used are stored
* **Reused the stored AI model** Load the stored model and reuse it.

```python
from revivai.base import SurrogateModel

loaded_model = SurrogateModel()
loaded_model.load("./model_data")
```
The stored model is loaded.

