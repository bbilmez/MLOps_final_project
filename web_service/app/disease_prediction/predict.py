import pickle
from typing import Union
import mlflow
import pandas as pd
from disease_prediction.utils import prepare_features


def load_model(experiment_id, run_id):
    """Get the model."""

    client = mlflow.tracking.MlflowClient()

    source = f"mlflow-artifacts:/{experiment_id}/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(source)

    # model_local_path = client.download_artifacts(run_id, "model/model.pkl", "./")
    dv_local_path = client.download_artifacts(run_id, "dict_vectorizer/dv.pkl", "./")

    with open(dv_local_path, "rb") as f_in:
        dict_vect = pickle.load(f_in)

    return model, dict_vect


def make_prediction(model, dict_vect, input_data: Union[list[dict], pd.DataFrame]):
    """Make prediction from features dict or DataFrame."""

    processed_data = prepare_features(input_data, dict_vect)
    preds = model.predict(processed_data)

    return preds
