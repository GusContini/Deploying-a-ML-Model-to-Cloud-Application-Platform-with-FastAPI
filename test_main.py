''' Docstring
'''
# test_main.py
#import pytest
import requests

# Base URL of the FastAPI app
BASE_URL = "http://127.0.0.1:8000"

# Test GET request
def test_get_root():
    ''' Docstring
    '''
    response = requests.get(BASE_URL, timeout=30)
    assert response.status_code == 200
    assert response.json() == {'message': 'Welcome to the ML Model Inference API!'}

# Test POST request for prediction where prediction is 0
def test_predict_income_0():
    ''' Docstring
    '''
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = requests.post(BASE_URL + "/predict", json=payload, timeout=30)
    assert response.status_code == 200
    assert response.json()["predictions"] == "[0]<= 50K"

# Test POST request for prediction where prediction is 1
def test_predict_income_1():
    ''' Docstring
    '''
    payload = {
        "age": 29,
        "workclass": "Private",
        "fnlgt": 185908,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States"
    }
    response = requests.post(BASE_URL + "/predict", json=payload, timeout=30)
    assert response.status_code == 200
    assert response.json()["predictions"] == "[1]> 50K"

# Run tests
if __name__ == '__main__':
    test_get_root()
    test_predict_income_0()
    test_predict_income_1()
