import joblib
import shap
import matplotlib.pyplot as plt
import os

# Ensure outputs folder exists
os.makedirs("outputs/plots", exist_ok=True)

print("Loading models and test data...")

# Load trained models
model_fraud = joblib.load("models/xgb_fraud_model.pkl")
model_card = joblib.load("models/xgb_credit_model.pkl")

# Load processed test sets
Xf_test = joblib.load("models/Xf_test.pkl")
Xc_test = joblib.load("models/Xc_test.pkl")

# ============= Fraud Model Explainability =============
print("Explaining Fraud Detection Model...")

explainer_fraud = shap.Explainer(model_fraud, Xf_test)
shap_values_fraud = explainer_fraud(Xf_test, check_additivity=False)

# Beeswarm plot
shap.summary_plot(shap_values_fraud, Xf_test, show=False)
plt.savefig("outputs/plots/shap_summary_fraud.png", bbox_inches='tight')
plt.close()

# Bar plot
shap.plots.bar(shap_values_fraud, show=False)
plt.savefig("outputs/plots/shap_bar_fraud.png", bbox_inches='tight')
plt.close()

# ============= Credit Card Model Explainability =============
print("Explaining Credit Card Detection Model...")

explainer_card = shap.Explainer(model_card, Xc_test)
shap_values_card = explainer_card(Xc_test, check_additivity=False)

# Beeswarm plot
shap.summary_plot(shap_values_card, Xc_test, show=False)
plt.savefig("outputs/plots/shap_summary_creditcard.png", bbox_inches='tight')
plt.close()

# Bar plot
shap.plots.bar(shap_values_card, show=False)
plt.savefig("outputs/plots/shap_bar_creditcard.png", bbox_inches='tight')
plt.close()

print("SHAP plots saved in outputs/plots/")
