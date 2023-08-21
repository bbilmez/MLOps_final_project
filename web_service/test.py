import requests

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

if __name__ == "__main__":

    host = "http://127.0.0.1:9696"
    url = f"{host}/result"
    response = requests.post(url, data=user_data)
    result = response.text

    print(result)
