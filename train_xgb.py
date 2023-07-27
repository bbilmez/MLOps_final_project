import os
import pickle
import argparse
import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def run_train(tracking_uri:str, experiment_name:str, data_path: str):

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        dv_path = os.path.join(data_path, "dv.pkl")
        dv = load_pickle(dv_path)

        params = {
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 77,
            'min_child_weight': 7,
            'gamma': 0.5,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'reg_alpha': 0.05,
            'reg_lambda': 1e-7,
            'eval_metric': 'mlogloss',
            'tree_method': 'approx', 
            'eval_metric' : 'logloss',
        }
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        
        with open(dv_path, 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(dv_path, artifact_path="dict_vectorizer")

        mlflow.log_param("parameters", params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(xgb_model, "model")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", default="http://127.0.0.1:5000", help="Mlflow tracking uri.")
    parser.add_argument("--experiment_name", default="heart-disease-experiment", help="mlflow tracking experiment name.")
    parser.add_argument("--data_path", default="./Output", help="Location where the heart disease data was saved")
    args = parser.parse_args()

    parameters = {
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "data_path": args.data_path,
    }
    run_train(**parameters)
