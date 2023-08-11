import json

import requests

user_data = {
    "Age" : 57,
    "Sex" : "M",
    "ChestPainType" : "ATA",
    "RestingBP" : 130,	
    "Cholesterol" : 210,
    "FastingBS" : 0,
    "RestingECG" : "ST",
    "MaxHR" : 122,
    "ExerciseAngina" : "N",
    "Oldpeak" : 1.5,
    "ST_Slope" : "Up",
}


if __name__ == "__main__":

    # host = "http://127.0.0.1:9696"
    # url = f"{host}/predict"
    # response = requests.post(url, json=user_data)
    # result = response.json()

    # print(result)

    url = "http://0.0.0.0:9696/predict"
    response = requests.post(url, json=user_data)
    
    if response.status_code == 200:
        result = response.json()
        print(result)
    else:
        print("Request failed:", response.status_code)
        # app.run(debug=True, host="0.0.0.0", port=9696)