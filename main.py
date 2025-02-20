from typing import Union
from fastapi import FastAPI
import numpy as np

# Import the model module
import model

app = FastAPI()

trained_model = None  

@app.get("/")
def read_root():
    return "Please enter one of the following to train:", "AND, OR, NOT"

@app.post("/AND")
def mod_and():
    global trained_model
    trained_model = model.Model(
        X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        y=np.array([0, 0, 0, 1])
    )
    trained_model.train()
    return {"result": "trained AND successfully"}

@app.post("/OR")
def mod_or():
    global trained_model
    trained_model = model.Model(
        X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        y=np.array([0, 1, 1, 1])  # Fix shape issue
    )
    trained_model.train()
    return {"result": "trained OR successfully"}

@app.post("/NOT")
def mod_not():
    global trained_model
    trained_model = model.Model(
        X=np.array([[0], [1]]),
        y=np.array([1, 0]),
        w = 1
    )
    trained_model.train()
    return {"result": "trained NOT successfully"}

@app.get("/predict/{X}/{y}") 
def predict(X: int, y: Union[int, None] = None):
    if trained_model is None:
        return {"error": "No trained model found. Train a model first!"}
    
    input_data = [X] if y is None else [X, y]
    result = trained_model.predict([input_data])
    return {"result": result}
