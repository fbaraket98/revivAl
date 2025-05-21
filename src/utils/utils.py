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
import tensorflow.keras.models as krs_models

def deserialize_model(buffer: bytes, lib: str, class_name: str):
    buffer_io = io.BytesIO(buffer)

    if lib in ["keras", "tensorflow"]:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(buffer_io.read())
            tmp.flush()
            model = krs_models.load_model(tmp.name)
        os.remove(tmp.name)
        return model

    elif lib == "catboost":
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

