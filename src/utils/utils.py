# Copyright MewsLabs 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import importlib
import io
import os
import tempfile

import joblib


def deserialize_model(buffer: bytes, lib: str, class_name: str):
    buffer_io = io.BytesIO(buffer)
    if lib == "catboost":
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

def train_or_fit(model, X, y):
    """
    Entraîne le modèle en utilisant train() si disponible, sinon fit().
    """
    try:
        model.set_training_values(X, y)
        model.train()
    except AttributeError:
        try:
            model.train(X, y)
        except AttributeError:
            try:
                model.fit(X, y)
            except AttributeError:
                raise ValueError("Le modèle n'a pas de méthode train ou fit")
    return model