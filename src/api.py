from fastapi import FastAPI, Request
import pandas as pd
import joblib
import uvicorn
from utils import *

config = load_config()

app = FastAPI()

@app.get('/')
async def root():
    response = {
        'status': 200,
        'message': 'Your Fraud Detection API is UP!'
    }
    return response


# loading model
def load_model():
    try:
        classifier = joblib.load('model/Decision_Tree_Classifier.pkl')
        return classifier
    except Exception as e:
        response = {
            'status': 204,
            'message': str(e)
        }
        return response
    

# predict data
@app.post('/predict')
async def predict(data: Request):

    # load request
    data = await data.json()
    data = pd.DataFrame(data, index = [0])

    # check data
    try:
        check_data(data)
    except AssertionError as e:
        return {
            'status': 204,
            'message': str(e)
        }

    # preprocess input data
    try:
        data_preprocess = preprocess_input(data)
        data_ohe = ohe_input(data_preprocess)
        data_clean = clean_input(data_preprocess, data_ohe)

        classifier = load_model()

        # predict data
        label = ['Not Fraud', 'Fraud']
        target_pred = classifier.predict(data_clean)

        return {
            'status': 200,
            'prediction': label[target_pred[0]]
        }

    except Exception as e:
        response = {
            'status': 204,
            'message': str(e)
        }
        return response

if __name__ == '__main__':
    uvicorn.run('api:app', host = '0.0.0.0', port = 8000, reload = True)