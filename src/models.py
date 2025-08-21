from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Literal, Tuple
from .config import LR_PARAMS, RF_PARAMS, XGB_PARAMS

def build_pipeline(model: Literal["lr", "rf", "xgb"], preprocessor, use_smote: bool = True):
    if model == "lr":
        clf = LogisticRegression(**LR_PARAMS)
    elif model == "rf":
        clf = RandomForestClassifier(**RF_PARAMS)
    elif model == "xgb":
        clf = XGBClassifier(**XGB_PARAMS)
    else:
        raise ValueError("Unknown model")

    steps = [("prep", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", clf))

    return ImbPipeline(steps)
