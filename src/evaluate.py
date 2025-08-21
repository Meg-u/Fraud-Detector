import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from .config import REPORTS_DIR, MODEL_CREDIT, MODEL_FRAUD
from .data_preprocessing import get_creditcard_splits, get_fraud_splits
from .utils_logging import get_logger

logger = get_logger("evaluate")

def _save_curves(model, X_test, y_test, prefix: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{prefix} - Confusion Matrix")
    plt.savefig(REPORTS_DIR / f"{prefix}_cm.png"); plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", color="gray"); plt.legend()
    plt.title(f"{prefix} - ROC"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.savefig(REPORTS_DIR / f"{prefix}_roc.png"); plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}"); plt.legend()
    plt.title(f"{prefix} - Precision-Recall"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.savefig(REPORTS_DIR / f"{prefix}_pr.png"); plt.close()

def evaluate_credit():
    model = joblib.load(MODEL_CREDIT)
    _, X_test, _, y_test, _ = get_creditcard_splits()
    _save_curves(model, X_test, y_test, "credit")

def evaluate_fraud():
    model = joblib.load(MODEL_FRAUD)
    _, X_test, _, y_test, _ = get_fraud_splits()
    _save_curves(model, X_test, y_test, "fraud")

if __name__ == "__main__":
    evaluate_credit()
    evaluate_fraud()
