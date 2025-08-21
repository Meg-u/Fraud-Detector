import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.config import MODEL_CREDIT, MODEL_FRAUD, REPORTS_DIR
from src.data_preprocessing import get_creditcard_splits, get_fraud_splits
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
st.set_page_config(page_title="Fraud Detector Dashboard", layout="wide")

st.title("Fraud Detector — Model Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.header("Creditcard")
    if (REPORTS_DIR / "credit_roc.png").exists():
        st.image(str(REPORTS_DIR / "credit_roc.png"), caption="ROC")
    if (REPORTS_DIR / "credit_pr.png").exists():
        st.image(str(REPORTS_DIR / "credit_pr.png"), caption="PR Curve")
    if (REPORTS_DIR / "credit_shap_summary.png").exists():
        st.image(str(REPORTS_DIR / "credit_shap_summary.png"), caption="SHAP Summary")

with col2:
    st.header("Fraud (E-commerce)")
    if (REPORTS_DIR / "fraud_roc.png").exists():
        st.image(str(REPORTS_DIR / "fraud_roc.png"), caption="ROC")
    if (REPORTS_DIR / "fraud_pr.png").exists():
        st.image(str(REPORTS_DIR / "fraud_pr.png"), caption="PR Curve")
    if (REPORTS_DIR / "fraud_shap_summary.png").exists():
        st.image(str(REPORTS_DIR / "fraud_shap_summary.png"), caption="SHAP Summary")

st.markdown("---")
st.subheader("Quick Prediction (Credit)")

# Minimal quick form (user pastes V1..V28, Time, Amount)
example = {"Time":0, "Amount":10.0}
for i in range(1,29):
    example[f"V{i}"] = 0.0

with st.form("credit_form"):
    vals = {}
    for k,v in example.items():
        vals[k] = st.number_input(k, value=float(v))
    submitted = st.form_submit_button("Predict")
    if submitted:
        model = joblib.load(MODEL_CREDIT)
        df = pd.DataFrame([vals])
        proba = model.predict_proba(df)[:,1][0]
        st.success(f"Fraud probability: {proba:.4f}")
