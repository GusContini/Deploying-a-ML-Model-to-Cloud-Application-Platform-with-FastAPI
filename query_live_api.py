''' Docstring
'''

import json
import requests

# Define the URL of your deployed API
URL = 'https://deploying-a-ml-model-to-cloud-whnf.onrender.com/predict'

# Define the data to be sent in the POST request
payload = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 250000,
    "education": "Bachelors",
    "education-num": 20,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States"
}

# Convert the data to JSON format
json_data = json.dumps(payload)

# Set the headers for the request
headers = {'Content-Type': 'application/json'}

# Send the POST request
response = requests.post(URL, data=json_data, headers=headers)

# Extract the result and status code from the response
result = response.json()
status_code = response.status_code

# Print the result and status code
print("Result:", result)
print("Status Code:", status_code)
