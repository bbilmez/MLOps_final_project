
import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("heart_disease_experiment")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./Output",
    help="Location where the heart disease data was saved"
)

def run_train(data_path: str):
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        dv_path = os.path.join(data_path, "dv.pkl")
        dv = load_pickle(dv_path)

        max_depth = 10
        rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        with open(dv_path, 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(dv_path, artifact_path="dict_vectorizer")


        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(rf, "model")


if __name__ == '__main__':
    run_train()
