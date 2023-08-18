
import sys
import os
from datetime import datetime
import pandas as pd
import pickle

module_path = os.path.abspath('../web_service/app')
sys.path.append(module_path)

from disease_prediction.utils import prepare_features


def test_prepare_features():
    
    user_data = {
        "Age": 57,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 130,
        "Cholesterol": 210,
        "FastingBS": 0,
        "RestingECG": "ST",
        "MaxHR": 122,
        "ExerciseAngina": "N",
        "Oldpeak": 1.5,
        "ST_Slope": "Up",
    }
    
    with open("./web_service/app/dict_vectorizer/dv.pkl", "rb") as f_in:
        dv = pickle.load(f_in)

    input_data = [user_data]
    X = prepare_features(input_data, dv)

    assert X.shape == (1,20)  

