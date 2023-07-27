
import os
import pickle
import argparse
import mlflow
import ast
import xgboost as xgb
from datetime import datetime
from prefect import flow, task, get_run_logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score


@task(name="Register the best model")
def run_register_model(tracking_uri: str, experiment_name: str, top_n: int):

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()
    
    logger = get_run_logger()
    
    # Select the model with the highest test accuracy
    logger.info("Getting best model from current experiment")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy ASC"]
    )[0]

    # Register the best model
    logger.info("Registering and staging best model")
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=f"heart_disease_xgb_model-{run_id}"
    )

    client.transition_model_version_stage(
        name=f"heart_disease_xgb_model-{run_id}",
        version=registered_model.version,
        stage="Staging",
    )

    # Update description of staged model
    logger.info("Updating description of staged model")
    client.update_model_version(
        name=f"heart_disease_xgb_model-{run_id}",
        version=registered_model.version,
        description=f"[{datetime.now()}] The model version {registered_model.version} from experiment '{experiment_name}' was transitioned to Staging.",
    )


@flow(name="Register the best model")
def main(tracking_uri, experiment_name, top_n):
    run_register_model(tracking_uri=tracking_uri, experiment_name=experiment_name, top_n=top_n)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", default="http://127.0.0.1:5000", help="Mlflow tracking uri.")
    parser.add_argument("--experiment_name", default="heart-disease-hyperopt", help="mlflow tracking experiment name.")
    parser.add_argument("--top_n", default=1, help="Number of top models that need to be evaluated to decide which one to promote")
    args = parser.parse_args()

    parameters = {
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "top_n": args.top_n
    }
    main(**parameters)
