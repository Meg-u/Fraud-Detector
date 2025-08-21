import joblib
from sklearn.metrics import classification_report, f1_score
from .config import (MODEL_CREDIT, MODEL_FRAUD, MODELS_DIR)
from .data_preprocessing import get_creditcard_splits, get_fraud_splits
from .models import build_pipeline
from .utils_logging import get_logger

logger = get_logger("train")

def train_and_save_credit(model_type="xgb"):
    X_tr, X_te, y_tr, y_te, preproc = get_creditcard_splits()
    pipe = build_pipeline(model_type, preproc, use_smote=True)
    pipe.fit(X_tr, y_tr)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_CREDIT)
    y_pred = pipe.predict(X_te)
    logger.info("CREDIT f1=%.4f", f1_score(y_te, y_pred))
    logger.info("\n%s", classification_report(y_te, y_pred, digits=4))

def train_and_save_fraud(model_type="xgb"):
    X_tr, X_te, y_tr, y_te, preproc = get_fraud_splits()
    pipe = build_pipeline(model_type, preproc, use_smote=True)
    pipe.fit(X_tr, y_tr)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_FRAUD)
    y_pred = pipe.predict(X_te)
    logger.info("FRAUD f1=%.4f", f1_score(y_te, y_pred))
    logger.info("\n%s", classification_report(y_te, y_pred, digits=4))

if __name__ == "__main__":
    train_and_save_credit("xgb")
    train_and_save_fraud("rf")  # or "xgb"
