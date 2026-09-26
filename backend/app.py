from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Fraud detection using StandardScaler + Tuned KNN",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------

model = joblib.load(
    "models/fraud_detection_pipeline.pkl"
)


# --------------------------------------------------
# Feature names
# --------------------------------------------------

FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: dict):

    # Convert input dictionary to DataFrame
    input_data = pd.DataFrame(
        [data],
        columns=FEATURES
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability if supported
    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        probability = float(probabilities[1])

    # Result
    if prediction == 1:
        result = "Fraud"
    else:
        result = "Normal"

    return {
        "prediction": int(prediction),
        "result": result,
        "fraud_probability": probability
    }