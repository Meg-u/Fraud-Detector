import os
import joblib
import shap
import matplotlib.pyplot as plt
from .config import REPORTS_DIR, MODEL_CREDIT, MODEL_FRAUD
from .data_preprocessing import get_creditcard_splits, get_fraud_splits

def _save_shap_summary(model, X_sample, prefix: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    explainer = shap.Explainer(model.named_steps["model"], feature_names=None)
    # Pass preprocessed matrix:
    X_trans = model.named_steps["prep"].transform(X_sample)
    shap_values = explainer(X_trans)
    plt.close("all")
    shap.summary_plot(shap_values, X_trans, show=False)
    plt.gcf().savefig(REPORTS_DIR / f"{prefix}_shap_summary.png", bbox_inches="tight")
    plt.close("all")

def explain_credit(n=200):
    model = joblib.load(MODEL_CREDIT)
    X_tr, X_te, *_ = get_creditcard_splits()
    _save_shap_summary(model, X_te.sample(min(n, len(X_te)), random_state=42), "credit")

def explain_fraud(n=200):
    model = joblib.load(MODEL_FRAUD)
    X_tr, X_te, *_ = get_fraud_splits()
    _save_shap_summary(model, X_te.sample(min(n, len(X_te)), random_state=42), "fraud")

if __name__ == "__main__":
    explain_credit()
    explain_fraud()
