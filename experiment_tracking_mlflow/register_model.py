
import os
import pickle
import click
import mlflow
import ast
import xgboost as xgb
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

HPO_EXPERIMENT_NAME = "heart-disease-hyperopt"
EXPERIMENT_NAME = "heart-disease-xgb-best-models"
XGB_PARAMS = ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 
                'reg_alpha', 'reg_lambda']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        inner_params = ast.literal_eval(params['parameters'])  # Convert string to dictionary
        for param in XGB_PARAMS:
            inner_params[param] = int(inner_params[param])

        xgb_model = xgb.XGBClassifier(**inner_params)
        xgb_model.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_accuracy= accuracy_score(y_val, xgb_model.predict(X_val))
        mlflow.log_metric("val_accuracy", val_accuracy)
        test_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
        mlflow.log_metric("test_rmse", test_accuracy)


@click.command()
@click.option(
    "--data_path",
    default="../Output",
    help="Location where the heart disease data was saved"
)
@click.option(
    "--top_n",
    default=1,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)


def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy ASC"]
    )

    
    for run in runs:
        print(run.data.params)
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the highest test accuracy
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy ASC"]
    )[0]

    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name='heart_disease_xgb_model'
    )


if __name__ == '__main__':
    run_register_model()