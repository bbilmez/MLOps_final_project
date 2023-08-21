# Final project for MLOps zoomcamp by DataTalks 

## Problem Statement  ##
Cardiovascular diseases stand as a significant global contributor to mortality, claiming approximately 17.9 million lives annually and representing 31% of all global deaths. Among the outcomes of these diseases is heart failure, a frequent occurrence. Those with cardiovascular disease or at heightened risk (due to factors like hypertension, diabetes, hyperlipidemia, or existing conditions) necessitate timely identification and intervention. This is where a machine learning model can offer substantial assistance. By automating this aspect, we address yet another natural challenge, allowing us to redirect our attention toward the next issue through the application of AI techniques.

Objective:
The aim of this study is to create a classification model that predicts whether a patient is at risk of experiencing heart failure based on a variety of attributes. The task involves binary classification using both numerical and categorical features.

Attributes in the Dataset:

Age: The age of the patient in years.
Sex: The sex of the patient (M: Male, F: Female).
ChestPainType: The type of chest pain experienced (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic).
RestingBP: Resting blood pressure in mm Hg.
Cholesterol: Serum cholesterol level in mm/dl.
FastingBS: Fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise).
RestingECG: Results of the resting electrocardiogram (Normal: Normal, ST: ST-T wave abnormality, LVH: Probable or definite left ventricular hypertrophy by Estes' criteria).
MaxHR: Maximum heart rate achieved (numeric value between 60 and 202).
ExerciseAngina: Presence of exercise-induced angina (Y: Yes, N: No).
Oldpeak: ST depression measured in numeric units.
ST_Slope: The slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping).
HeartDisease: Output class indicating the presence of heart disease (1: heart disease, 0: Normal).

In this analysis, we aim to develop a model that can effectively classify patients' heart health based on these attributes, contributing to early detection and intervention for potential heart-related issues.

## Project Setup ##

Clone the project from the repository
```
git clone https://github.com/bbilmez/MLOpssoomcamp_capstone.git
```

Change to MLOPS_FINAL_PROJECT directory
```
cd MLOPS_FINAL_PROJECT
```

Setup and install project dependencies
```
make setup
```
Add your current directory to python path
```
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

# Start Local Prefect Server #

In a new terminal window or tab run the command below to start prefect server

```
prefect server start
```

# Start Local Mlflow Server #

In a new terminal window or tab run the following commands below

```
mlflow server
```

# Preprocess data for modeling #
```
python preprocess_data.py --raw_data_path="Data" --dest_path="Output"
```
This python file reads the train/test/valiaton data and extracts predictor and target columns from these dataset.
Then, use DictVectorizer() to convert a dataframes into a numeric feature matrix suitable for machine learning algorithms.

![Process data flow on prefect ui](images/prefect_process_flow.png)

# Running model training and model registery pipelines locally #

```
python train_xgb.py --tracking_uri="http://127.0.0.1:5000" --experiment_name="heart-disease-experiment" --data_path="./Output"
```

This python file starts an MLflow experiment (heart_disease_experiment), train and test a xgboost model using train and validation data and defined parameters. Log this model, its parameters, metrics and DictVectorizer to MLflow. 

![heart_disease_experiment in MLFlow dashboard](./images/heart_disease_experiment.png)

![trained model without parameter optimization](./images/trained_xgb_wout_opt.png)

```
python hyperoptimization_xgb.py --tracking_uri="http://127.0.0.1:5000" --experiment_name="heart-disease-hyperopt" --num_trials=10 --data_path="./Output"
```

This python file starts another MLflow experiment (heart-disease-hyperopt), search best parameters for xgboost model to obtain highest accuracy. Train and test a xgboost model using these best parameters. Log this model, its parameters, metrics and DictVectorizer to MLflow. 

![heart-disease-hyperopt in MLFlow dashboard](./images/heart_disease_hyperopt.png)

![trained model with parameter optimization](./images/trained_xgb_optimized.png)

![Model optimization flow on Prefect](images/prefect_model_optimization.png)


```
python register_model.py --tracking_uri="http://127.0.0.1:5000" --top_n=1
```

This python file logs the best model giving highest validation and test accuracy. Register this model to MLflow.

![Registered model on MLflow](./images/registered_model.png)

![Model registration flow on Prefect](images/prefect_register_model.png)

# Create scheduled deployments #

```
python deployment.py
```

![Prefect deployments](./images/prefect_deployments.png)

Run deployments locally

```
prefect deployment run run-optimization/deploy-mlflow-training
prefect deployment run register-best-model/deploy-mlflow-staging
```
![Prefect deployment run](images/prefect_deployment_run.png)


# Deploying a model locally #

Add .env file containing environemt variables: EXPERIMENT_ID and RUN_ID.

```
cd web_service/app
python main.py
```
main.py will reach spesified experiment/run and get the artifacts (model and dictionary vectorizer) from this experiment. On your browser, open <http://127.0.0.1:9696/predict>. Fill in the form and after submitting it, the app will direct you to the result page (<http://127.0.0.1:9696/result>) and you will see the prediction.

![Input data form on http://127.0.0.1:9696/predict](./images/data_form_on_server.png)
![Prediction on http://127.0.0.1:9696/result](./images/prediction_on_server.png)

Without opening prediction server on your browser, prediction could also be done with following lines:

```
curl -X POST -d "Age=57&Sex=M&ChestPainType=ATA&RestingBP=130&Cholesterol=210&FastingBS=0&RestingECG=ST&MaxHR=122&ExerciseAngina=N&Oldpeak=1.5&ST_Slope=Up" http://127.0.0.1:9696/result
```

# Deploying a model as web-service with docker

## Packaging the app to docker

```
docker build -t heart-disease-app:v1 .
```

## Running the docker container service with logs

```
docker run --rm -v $(pwd):/app \
    -p 9696:9696  heart-disease-app:v1
```

## Test the web-service

```
python test.py
```

## Run all the above steps in one command

```
make build_webservice
```

### Model monitoring ###

There are 3 services for monitoring the model predictions is realtime:

Evidently AI for calculating metrics including data drift, missing value and quartile metrics. 
Adminer for collecting monitoring data. Prometheus UI: http://localhost:8080
Grafana for Dashboards UI. Grafana UI: http://localhost:3000 (default user/pass: admin, admin)

To run the script for calculate and report these metrics, add .env file containing environemt variables: EXPERIMENT_ID, RUN_ID and current data path. Then, follow below steps from project directory.
```
cd model_monitoring
docker-compose up --build
```
and in another terminal tab:
```
python evidently_metric_calculation.py
```

 This script calculates some evidently metrics and save them in as a Postgresql database. This database can be viewed on http://localhost:8080 and dashboards can be made on http://localhost:3000 using this database. The script also saves Evidently report in model_monitoring/reports folder. 

![Dashboard on Grafana for our data monitoring](./images/dashboard_grafana.png)

Sample report file: (model_monitoring/reports/data_report-2023-08-18_12-35.html)

