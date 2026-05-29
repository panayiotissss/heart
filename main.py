from contextlib import asynccontextmanager
from typing import Literal
import json
import os

from dotenv import load_dotenv
load_dotenv()

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel


TRAIN_API_KEY = os.getenv('TRAIN_API_KEY', 'dev-key')
_api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)

def verify_train_key(key: str = Depends(_api_key_header)):
    if key != TRAIN_API_KEY:
        raise HTTPException(status_code=401, detail='Unauthorized')


#Startup/Shutdown
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists('model.pkl'):
        from model import train_and_save
        train_and_save()
    model = joblib.load('model.pkl')
    yield  #Before yield = on startup after = on shutdown
    model = None  


app = FastAPI(title='Disease Prediction API', lifespan=lifespan)


#Input of api request
# Pydantic validates every incoming request against this class.
# If a field is missing or the wrong type, FastAPI returns a 422 automatically.
# Literal['X', 'Y'] restricts the value to only those strings.

class PatientFeatures(BaseModel):
    age:                  float
    bmi:                  float
    glucose_mg_dl:        float
    cholesterol_mg_dl:    float
    systolic_bp:          float
    diastolic_bp:         float
    heart_rate:           float
    gender:               Literal['Male', 'Female']
    smoking:              Literal['Yes', 'No']
    alcohol_consumption:  Literal['Yes', 'No']
    physical_activity:    Literal['Low', 'Medium', 'High']
    family_history:       Literal['Yes', 'No']


#Endpoints

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.get('/model/info')
def model_info():
    if not os.path.exists('metrics.json'):
        raise HTTPException(status_code=404, detail='No trained model found. Call POST /train first.')
    with open('metrics.json') as f:
        metrics = json.load(f)
    return metrics


@app.post('/train')
def train(key: None = Security(verify_train_key)):
    from model import train_and_save  # imported here to avoid running on startup
    global model
    metrics = train_and_save()
    model = joblib.load('model.pkl')  # reload the freshly trained model into memory
    return {'status': 'training complete', 'metrics': metrics}


@app.post('/predict')
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Call POST /train first.')

    # Convert the incoming request into a single-row DataFrame.
    # The pipeline expects a DataFrame with the same column names it was trained on.
    input_df = pd.DataFrame([patient.model_dump()])

    prediction   = model.predict(input_df)[0]             # 0 or 1
    probability  = model.predict_proba(input_df)[0][1]    # probability of disease=Yes

    return {
        'prediction':  'Yes' if prediction == 1 else 'No',
        'probability': round(float(probability), 4)
    }
