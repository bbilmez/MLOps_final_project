from datetime import datetime, timedelta
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule, IntervalSchedule
from hyperoptimization_xgb import run_optimization
from register_model import main

date_str = datetime.today().strftime("%Y-%m-%d")

mlflow_training_deployment = Deployment.build_from_flow(
    name="deploy-mlflow-training",
    schedule=IntervalSchedule(interval=timedelta(minutes=10080)),
    flow=run_optimization,
    parameters={
        "tracking_uri": "http://127.0.0.1:5000",
        "experiment_name": "heart_disease_experiment",
        "num_trials":10, 
        "data_path": "./Output"
    },
    tags=["ml-training"],
)

mlflow_staging_deployment = Deployment.build_from_flow(
    name="deploy-mlflow-staging",
    schedule=CronSchedule(
        cron="0 9 1 * *",
    ),
    flow=main,
    parameters={
        "tracking_uri": "http://127.0.0.1:5000",
        "experiment_name": f"heart-disease-experiment_{date_str}",
        "top_n": 1
    },
    tags=["ml-staging"],
)

if __name__ == "__main__":
    mlflow_training_deployment.apply()
    mlflow_staging_deployment.apply()