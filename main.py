''' Docstring
'''

import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starter.ml import data, model

app = FastAPI()


class InputData(BaseModel):
    ''' Docstring
    '''
    age: int = Field(examples=[39])
    workclass: str = Field(examples=["State-gov"])
    fnlgt: int = Field(examples=[77516])
    education: str = Field(examples=["Bachelors"])
    education_num: int = Field(..., alias="education-num",
                               examples=[13])
    marital_status: str = Field(..., alias="marital-status",
                                examples=["Never-married"])
    occupation: str = Field(examples=["Adm-clerical"])
    relationship: str = Field(examples=["Not-in-family"])
    race: str = Field(examples=["White"])
    sex: str = Field(examples=["Male"])
    capital_gain: int = Field(..., alias="capital-gain",
                              examples=[2174])
    capital_loss: int = Field(..., alias="capital-loss",
                              examples=[0])
    hours_per_week: int = Field(..., alias="hours-per-week",
                                examples=[40])
    native_country: str = Field(..., alias="native-country",
                                examples=["United-States"])


# Load trained model
with open(os.path.join(
    os.path.dirname(__file__), 'starter', 'trained_model.pkl'),
        'rb'
    ) as file:
    trained_model, encoder, lb = pickle.load(file)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def perform_inference(input_data):
    ''' Docstring
    '''
    input_dict = input_data.dict()
    input_dict['salary'] = '0'

    X_processed, _, _, _ = data.process_data(
        input_dict,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    predictions = model.inference(trained_model, X_processed)
    return predictions


@app.get('/')
async def read_root():
    ''' Docstring
    '''
    return {'message': 'Welcome to the ML Model Inference API!'}


@app.post('/predict')
async def predict_income(input_data: InputData):
    ''' Docstring
    '''
    try:
        predictions = perform_inference(input_data)
        if predictions == 0:
            pred_label = '<= 50K'
        else:
            pred_label = '> 50K'
        return {'predictions': str(predictions) + pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
