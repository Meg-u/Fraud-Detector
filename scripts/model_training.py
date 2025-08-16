import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def load_data():
    creditcard_df = pd.read_csv("data/creditcard.csv")
    fraud_df = pd.read_csv("data/processed/fraud_data_processed.csv")
    return creditcard_df, fraud_df

def preprocess_creditcard(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def preprocess_fraud(df):
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["time_diff"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    df["ip_int"] = df["ip_address"].apply(lambda x: int("".join(x.split("."))) if isinstance(x, str) else 0)

    y = df["class"]

    df = df.drop(columns=["user_id", "signup_time", "purchase_time", "device_id", "ip_address", "class"])
    df = pd.get_dummies(df, drop_first=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print("⚠️ Non-numeric columns found:", non_numeric.tolist())

    return train_test_split(df, y, test_size=0.3, stratify=y, random_state=42)

def train_models(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    xgb = XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss', random_state=42)
    log_reg.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    return log_reg, xgb

def evaluate_model(y_true, y_pred, model_name, dataset_name):
    print(f"\n{model_name} on {dataset_name}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
    f1 = f1_score(y_true, y_pred)
    print("F1 Score:", round(f1, 4))

def plot_pr_curve(model, X_test, y_test, title):
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(recall, precision)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({title})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    creditcard_df, fraud_df = load_data()

    # Preprocess datasets
    Xc_train, Xc_test, yc_train, yc_test = preprocess_creditcard(creditcard_df)
    Xf_train, Xf_test, yf_train, yf_test = preprocess_fraud(fraud_df)

    # Train models
    log_reg_credit, xgb_credit = train_models(Xc_train, yc_train)
    log_reg_fraud, xgb_fraud = train_models(Xf_train, yf_train)

    # Evaluate models
    evaluate_model(yc_test, log_reg_credit.predict(Xc_test), "Logistic Regression", "Creditcard")
    evaluate_model(yc_test, xgb_credit.predict(Xc_test), "XGBoost", "Creditcard")
    evaluate_model(yf_test, log_reg_fraud.predict(Xf_test), "Logistic Regression", "Fraud_Data")
    evaluate_model(yf_test, xgb_fraud.predict(Xf_test), "XGBoost", "Fraud_Data")

    # Plot Precision-Recall curves
    plot_pr_curve(log_reg_credit, Xc_test, yc_test, "Logistic Regression - Creditcard")
    plot_pr_curve(xgb_credit, Xc_test, yc_test, "XGBoost - Creditcard")
    plot_pr_curve(log_reg_fraud, Xf_test, yf_test, "Logistic Regression - Fraud_Data")
    plot_pr_curve(xgb_fraud, Xf_test, yf_test, "XGBoost - Fraud_Data")

    # SHAP Explanation for Fraud Model
    import shap
    print("Explaining Fraud Detection Model...")

    # Force numeric dtype for SHAP compatibility
    Xf_test = Xf_test.astype(np.float64)

    explainer_fraud = shap.Explainer(xgb_fraud, Xf_test)
    shap_values_fraud = explainer_fraud(Xf_test)

    print(shap_values_fraud[1].shape, Xf_test.shape)
    shap.summary_plot(shap_values_fraud, Xf_test, show=False, plot_type="bar")
    plt.savefig("reports/shap_summary_fraud.png", bbox_inches='tight')
    plt.close()

    # Save models and test sets
    os.makedirs("models", exist_ok=True)
    joblib.dump(log_reg_credit, "models/logreg_credit_model.pkl")
    joblib.dump(xgb_credit, "models/xgb_credit_model.pkl")
    joblib.dump(log_reg_fraud, "models/logreg_fraud_model.pkl")
    joblib.dump(xgb_fraud, "models/xgb_fraud_model.pkl")
    joblib.dump(Xf_test, "models/Xf_test.pkl")
    joblib.dump(Xc_test, "models/Xc_test.pkl")

    print("Models saved in 'models/' directory.")