from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import joblib
import pandas as pd
import shap
import logging
from typing import Dict, Any

from src.config import MODEL_CREDIT, MODEL_FRAUD
from src.schemas import FraudRequest, CreditRequest

# Initialize app
app = FastAPI(title="Fraud Detector API", version="1.0")

# Setup logging
logger = logging.getLogger("fraud_api")
logging.basicConfig(level=logging.INFO)

# CORS configuration
origins = [
    "http://localhost:3000",  # React dev server
    "https://your-production-domain.com"  # Optional deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded models
_model_credit = None
_model_fraud = None

def load_models():
    global _model_credit, _model_fraud
    if _model_credit is None:
        logger.info("Loading credit model...")
        _model_credit = joblib.load(MODEL_CREDIT)
    if _model_fraud is None:
        logger.info("Loading fraud model...")
        _model_fraud = joblib.load(MODEL_FRAUD)

@app.on_event("startup")
def _startup():
    load_models()
    logger.info("Models loaded successfully.")

def _predict(model, payload: Dict[str, Any]):
    df = pd.DataFrame([payload])
    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": float(proba)}

@app.post("/predict/credit")
def predict_credit(req: CreditRequest):
    logger.info("Received credit prediction request")
    try:
        return _predict(_model_credit, req.features)
    except Exception as e:
        logger.error(f"Credit prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/fraud")
def predict_fraud(req: FraudRequest):
    logger.info("Received fraud prediction request")
    try:
        return _predict(_model_fraud, req.features)
    except Exception as e:
        logger.error(f"Fraud prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/shap/credit")
def shap_credit(req: CreditRequest):
    logger.info("Generating SHAP values for credit model")
    try:
        df = pd.DataFrame([req.features])
        X = _model_credit.named_steps["prep"].transform(df)
        explainer = shap.Explainer(_model_credit.named_steps["model"])
        values = explainer(X)
        return {"shap_values": values.values[0].tolist()}
    except Exception as e:
        logger.error(f"SHAP credit error: {e}")
        raise HTTPException(status_code=500, detail="SHAP explanation failed")

@app.post("/shap/fraud")
def shap_fraud(req: FraudRequest):
    logger.info("Generating SHAP values for fraud model")
    try:
        df = pd.DataFrame([req.features])
        X = _model_fraud.named_steps["prep"].transform(df)
        explainer = shap.Explainer(_model_fraud.named_steps["model"])
        values = explainer(X)
        return {"shap_values": values.values[0].tolist()}
    except Exception as e:
        logger.error(f"SHAP fraud error: {e}")
        raise HTTPException(status_code=500, detail="SHAP explanation failed")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "🚀 Fraud Detector API is live!",
        "endpoints": [
            "/predict/credit",
            "/predict/fraud",
            "/shap/credit",
            "/shap/fraud",
            "/health"
        ],
        "docs": "/docs"
    }