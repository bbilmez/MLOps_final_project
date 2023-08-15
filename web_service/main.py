import json
import os
from urllib.parse import parse_qs

import mlflow
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

from disease_prediction.predict import load_model, make_prediction

# from get_profile import get_profile


load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model with run ID and experiment ID defined in the env.
RUN_ID = os.getenv("RUN_ID")
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID")

model, dv = load_model(run_id=RUN_ID, experiment_id=EXPERIMENT_ID)

app = Flask("heart-disease-prediction")


@app.route("/predict", methods=["GET", "POST"])
def predict_endpoint():
    """Get the data from users"""
    return render_template("predict.html")


@app.route("/result", methods=["POST", "GET"])
def result_endpoint():
    """Predict the presence of heart disease heart_disease_xgb_model."""
    if request.method == "POST":

        user_data = request.form.to_dict()
        prediction_result = make_prediction(
            model, dv, user_data
        )  # Call your prediction function again
        result_label = "No Heart Disease" if prediction_result == 0 else "Heart Disease"

        return f"{result_label}"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
