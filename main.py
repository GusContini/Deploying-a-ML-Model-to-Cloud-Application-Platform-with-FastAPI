from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'starter', 'ml'))
import data
import model

app = FastAPI()

class InputData(BaseModel):
    age: conlist(int, min_items=1, max_items=1)
    workclass: conlist(str, min_items=1, max_items=1)
    fnlwgt: conlist(int, min_items=1, max_items=1)
    education: conlist(int, min_items=1, max_items=1)
    education-num: conlist(int, min_items=1, max_items=1)
    marital-status: conlist(str, min_items=1, max_items=1)
    occupation: conlist(str, min_items=1, max_items=1)
    relationship: conlist(str, min_items=1, max_items=1)
    race: conlist(str, min_items=1, max_items=1)
    sex: conlist(str, min_items=1, max_items=1)
    capital-gain: conlist(int, min_items=1, max_items=1)
    capital-loss: conlist(int, min_items=1, max_items=1)
    hours-per-week: conlist(str, min_items=1, max_items=1)
    native-country: conlist(str, min_items=1, max_items=1)

# Load trained model
with open(os.path.join(os.path.dirname(__file__), 'starter', 'trained_model.pkl'), 'rb') as file:
    trained_model, encoder, lb = pickle.load(file)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def perform_inference(input_data):
    X_processed, _, _, _ = data.process_data(
        input_data, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )
    predictions = model.inference(trained_model, X_processed)
    return predictions

@app.get('/')
async def read_root():
    return {'message': 'Welcome to the ML Model Inference API!'}

@app.post('/predict')
async def predict_income(input_data: InputData):
    try:
        predictions = perform_inference(input_data.dict())
        return {'predictions': predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))