from typing import Union

import mlflow
import pandas as pd

from disease_prediction.utils import prepare_features


def load_model(experiment_id, run_id):
    """Get the model."""

    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # client = mlflow.tracking.MlflowClient()

    source = f"mlflow-artifacts:/{experiment_id}/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(source)
    dv_uri = f"mlflow-artifacts:/{experiment_id}/{run_id}/artifacts/dict_vectorizer"
    # dv = f"./mlruns/{experiment_id}/{run_id}/artifacts/dict_vectorizer/dv.pkl"
    dv = mlflow.pyfunc.load_model(dv_uri)

    return model, dv


def make_prediction(model, dv, input_data: Union[list[dict], pd.DataFrame]):
    """Make prediction from features dict or DataFrame."""

    X = prepare_features(input_data, dv)
    preds = model.predict(X)

    return preds
