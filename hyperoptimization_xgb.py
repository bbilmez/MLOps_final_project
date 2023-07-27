import os
import pickle
import argparse
import mlflow
import optuna
import numpy as np
from datetime import date
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@flow
def run_optimization(tracking_uri:str, experiment_name:str, num_trials: int, data_path: str):

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        dv_path = os.path.join(data_path, "dv.pkl")
        dv = load_pickle(dv_path)

        def objective(trial):
            
            params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
            'eval_metric': 'mlogloss',
            'enable_categorical':True,
            'tree_method': 'approx', # 'gpu_hist',
            'missing' : np.nan,
            'eval_metric' : 'logloss',
        }

            xgb_model = xgb.XGBClassifier(**params)
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            return accuracy

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=num_trials)

        trial = study.best_trial
        params = trial.params
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        mlflow.log_param("parameters", params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(model, "model")
        
        with open(dv_path, 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(dv_path, artifact_path="dict_vectorizer")

        markdown_accuracy_report = f"""# Accuracy Report

        ## Summary

        Heart Disease Prediction 

        ## Accuracy XGBoost Model

        | Metric    |        |
        |:----------|-------:|
        | accuracy  | {accuracy:.2f} |
        | f1 score  | {f1:.2f} |
        | precision | {precision:.2f} |
        | recall    | {recall:.2f} |
        """

        create_markdown_artifact(
            key="heart-disease-model-report", markdown=markdown_accuracy_report
        )

        return None



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", default="http://127.0.0.1:5000", help="Mlflow tracking uri.")
    parser.add_argument("--experiment_name", default="heart-disease-hyperopt", help="mlflow tracking experiment name.")
    parser.add_argument("--num_trials", default=10, help="The number of parameter evaluations for the optimizer to explore")
    parser.add_argument("--data_path", default="./Output", help="Location where the heart disease data was saved")
    args = parser.parse_args()

    parameters = {
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "num_trials": args.num_trials,
        "data_path": args.data_path,
    }
    run_optimization(**parameters)

