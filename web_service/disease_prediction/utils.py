from typing import Union

import pandas as pd

# from haversine import Unit, haversine


def prepare_features(input_data: Union[list[dict], pd.DataFrame], dv):

    df = pd.DataFrame(input_data)
    dicts = df.to_dict(orient="records")
    X = dv.transform(dicts)

    return X


if __name__ == "__main__":
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

    input_data = [user_data]
    X = prepare_features(input_data)
    print(X)
