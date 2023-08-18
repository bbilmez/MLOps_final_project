import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
from dotenv import load_dotenv
import mlflow
import pickle
import os
from typing import Union
# from datetime import datetime

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists metrics;
create table metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	max_value_of_cholesterol float
)
"""

def load_model(experiment_id, run_id):
    """Get the model."""

    client = mlflow.tracking.MlflowClient()

    source = f"mlflow-artifacts:/{experiment_id}/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(source)
    dv_local_path = client.download_artifacts(run_id, "dict_vectorizer/dv.pkl", "./")

    with open(dv_local_path, "rb") as f_in:
        dv = pickle.load(f_in)

    return model, dv

def prepare_features(input_data: Union[list[dict], pd.DataFrame], dv):

    X = dv.transform(input_data)
    return X

def make_prediction(model, dv, input_data: Union[list[dict], pd.DataFrame]):
    """Make prediction from features dict or DataFrame."""

    X = prepare_features(input_data, dv)
    preds = model.predict(X)

    return preds



load_dotenv()

current_data_path = os.getenv("current_data_path")
current_data = pd.read_csv(current_data_path)

reference_data = pd.read_parquet('../Data/reference.parquet')

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = os.getenv("RUN_ID")
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID")

model, dv = load_model(run_id=RUN_ID, experiment_id=EXPERIMENT_ID)


begin = datetime.datetime(2023, 8, 19, 0, 0)

categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

column_mapping = ColumnMapping(
    prediction='Prediction',
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    target=None
)

report = Report(
    metrics = [
        ColumnDriftMetric(column_name='Prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name="Cholesterol", quantile=0.5),
    ]
)

@task(retries=2, retry_delay_seconds=5, name="prepare database")
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='heart_disease'")
		if len(res.fetchall()) == 0:
			conn.execute("create database heart_disease;")
		with psycopg.connect("host=localhost port=5432 dbname=heart_disease user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task(retries=2, retry_delay_seconds=5, name="calculate metrics")
def calculate_metrics_postgresql(curr, i):

    current_data_dict = current_data[numerical_features + categorical_features].to_dict(orient='records')

    current_data_preds = make_prediction(
                model, dv, current_data_dict)
				
    current_data['Prediction'] = current_data_preds

    report.run(reference_data = reference_data, current_data = current_data, column_mapping=column_mapping)

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M")
    report.save_html(f"reports/data_report-{date_time}.html")
    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    max_value_of_cholesterol = result['metrics'][3]['result']['current']['value']


    curr.execute(
        "insert into metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, max_value_of_cholesterol) values (%s, %s, %s, %s, %s)",
        (begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values, max_value_of_cholesterol)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=heart_disease user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 27):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()
