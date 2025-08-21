from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CREDIT_PATH = DATA_DIR / "creditcard.csv"
FRAUD_PATH = PROCESSED_DIR / "fraud_data_processed.csv"

# Training config
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Model choices / hyperparams (simple defaults)
RF_PARAMS = {"n_estimators": 300, "class_weight": "balanced", "random_state": RANDOM_STATE}
LR_PARAMS = {"max_iter": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE}
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": 10,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss"
}

# Output model filenames
MODEL_CREDIT = MODELS_DIR / "xgb_credit_model.pkl"
MODEL_FRAUD = MODELS_DIR / "xgb_fraud_model.pkl"
