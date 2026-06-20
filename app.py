from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Alfido Tech Model API",
    description="A simple API for serving an Iris classification model.",
    version="1.0.0"
)

# Define the request payload structure
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Global variable to hold the model
model = None

# Mapping array for class indices to names
class_names = ["setosa", "versicolor", "virginica"]

@app.on_event("startup")
def load_model():
    global model
    # Load the trained model from disk
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {model_path}. Make sure train_model.py has been run.")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Alfido Tech Model API!",
        "status": "Healthy",
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_iris(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Prepare data for prediction (2D array expected by scikit-learn)
        data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]]
        
        # Make prediction
        prediction = model.predict(data)
        class_idx = int(prediction[0])
        
        return {
            "prediction_class_index": class_idx,
            "prediction_class_name": class_names[class_idx]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
