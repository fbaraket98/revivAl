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
