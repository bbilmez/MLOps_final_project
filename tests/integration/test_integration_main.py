
# pylint: disable=duplicate-code

import json

import requests
from deepdiff import DeepDiff

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

host = "http://127.0.0.1:9696"
url = f"{host}/result"
response = requests.post(url, data=user_data)
actual_result = response.text

expected_result = "No Heart Disease"


diff = DeepDiff(actual_result, expected_result, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff

# import unittest
# import requests

# class TestWebServiceIntegration(unittest.TestCase):
#     def test_prediction(self):
#         user_data = {
#             "Age": 57,
#             "Sex": "M",
#             "ChestPainType": "ATA",
#             "RestingBP": 130,
#             "Cholesterol": 210,
#             "FastingBS": 0,
#             "RestingECG": "ST",
#             "MaxHR": 122,
#             "ExerciseAngina": "N",
#             "Oldpeak": 1.5,
#             "ST_Slope": "Up",
#         }

#         host = "http://127.0.0.1:9696"
#         url = f"{host}/result"
#         response = requests.post(url, json=user_data)  # Send JSON data
#         result = response.text

#         self.assertEqual(response.status_code, 200)  # Ensure successful response
#         self.assertIn("prediction", result)  # Ensure the response contains the prediction

# if __name__ == "__main__":
#     unittest.main()
