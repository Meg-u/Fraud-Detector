

from __future__ import annotations

import ipaddress
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# -----------------------------
# Helpers
# -----------------------------
def _ip_to_int(val) -> float:
    """Convert dotted IPv4 to integer; return NaN on failure."""
    try:
        if pd.isna(val):
            return np.nan
        return float(int(ipaddress.ip_address(str(val))))
    except Exception:
        return np.nan


# -----------------------------
# Fraud feature engineering
# -----------------------------
class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create stable numeric features from raw fraud dataframe columns.
    - Expects columns (when available): signup_time, purchase_time, ip_address
    - Will NOT fail if some columns are missing (it will just skip them).
    """

    def __init__(self, drop_high_cardinality: bool = True, high_cardinality_threshold: int = 50):
        self.drop_high_cardinality = drop_high_cardinality
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_card_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        Xc = X.copy()
        # Identify high-cardinality object columns (e.g., device_id)
        obj_cols = Xc.select_dtypes(include="object").columns.tolist()
        self.high_card_cols_ = [
            c for c in obj_cols if Xc[c].nunique(dropna=True) > self.high_cardinality_threshold
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Timestamps → engineered features
        if "signup_time" in X.columns:
            X["signup_time"] = pd.to_datetime(X["signup_time"], errors="coerce")
        if "purchase_time" in X.columns:
            X["purchase_time"] = pd.to_datetime(X["purchase_time"], errors="coerce")

        if "signup_time" in X.columns and "purchase_time" in X.columns:
            X["time_since_signup"] = (X["purchase_time"] - X["signup_time"]).dt.total_seconds()
        else:
            X["time_since_signup"] = np.nan

        if "purchase_time" in X.columns:
            X["purchase_hour"] = X["purchase_time"].dt.hour
            X["purchase_dayofweek"] = X["purchase_time"].dt.dayofweek
        else:
            X["purchase_hour"] = np.nan
            X["purchase_dayofweek"] = np.nan

        # IP → integer
        if "ip_address" in X.columns:
            X["ip_int"] = X["ip_address"].apply(_ip_to_int)
        else:
            X["ip_int"] = np.nan

        # Fill obvious numeric NaNs with 0 (safe for engineered features)
        for col in ["time_since_signup", "purchase_hour", "purchase_dayofweek", "ip_int"]:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # Optionally drop very high-cardinality columns (e.g., device_id)
        if self.drop_high_cardinality and self.high_card_cols_:
            to_drop = [c for c in self.high_card_cols_ if c in X.columns]
            X = X.drop(columns=to_drop, errors="ignore")

        # Drop raw timestamp columns—they’re no longer needed
        X = X.drop(columns=[c for c in ["signup_time", "purchase_time"] if c in X.columns], errors="ignore")

        return X


# -----------------------------
# Credit card preprocessing
# -----------------------------
def build_credit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessor for the credit card dataset:
    - Scale ['Amount','Time'] if present
    - Pass through V1..V28 (already PCA-like)
    """
    scale_cols = [c for c in ["Amount", "Time"] if c in X_train.columns]
    passthrough_cols = [c for c in X_train.columns if c not in scale_cols]

    transformers = []
    if scale_cols:
        transformers.append(("scale", StandardScaler(), scale_cols))
    if passthrough_cols:
        transformers.append(("passthrough", "passthrough", passthrough_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def get_creditcard_splits(
    path: str = "data/creditcard.csv",
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Load credit card data, split, and build a preprocessor (fit later in a Pipeline).
    """
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in credit card dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_credit_preprocessor(X_train)
    return X_train, X_test, y_train, y_test, preprocessor


# -----------------------------
# Fraud preprocessing
# -----------------------------
def build_fraud_preprocessor(
    drop_high_cardinality: bool = True,
    high_cardinality_threshold: int = 50,
) -> Pipeline:
    """
    Build a Pipeline for fraud data:
    1) Feature engineer (timestamps, ip, etc.)
    2) ColumnTransformer:
        - numeric → StandardScaler
        - low-cardinality categoricals → OneHotEncoder
        - (if not dropped) high-cardinality categoricals → OrdinalEncoder
    NOTE: high-cardinality cols are detected at fit() time by FraudFeatureEngineer.
    """
    # Stage 1: feature engineering
    fe = FraudFeatureEngineer(
        drop_high_cardinality=drop_high_cardinality,
        high_cardinality_threshold=high_cardinality_threshold,
    )

    # Stage 2: column-wise preprocessing (selectors evaluate after FE)
    numeric_selector = selector(dtype_include=["number", "bool"])
    categorical_selector = selector(dtype_include=object)

    # OneHot for all object columns (after FE). If you keep very high-card cols, consider OrdinalEncoder instead.
    coltx = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_selector),
            ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_selector)
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[("features", fe), ("prep", coltx)])


def get_fraud_splits(
    path: str = "data/processed/fraud_data_processed.csv",
    target_col: str = "class",
    test_size: float = 0.3,
    random_state: int = 42,
    drop_high_cardinality: bool = True,
    high_cardinality_threshold: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Load fraud data, split, and build a preprocessing Pipeline.
    - Works whether your CSV still has raw columns (signup_time, device_id, etc.) or
      already contains engineered/OHE columns. The FE step is robust and will skip missing pieces.
    """
    df = pd.read_csv(path, low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in fraud dataset.")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Fill common categorical NA early to stabilize encoders
    for c in ["source", "browser", "sex", "country", "device_id"]:
        if c in X.columns:
            X[c] = X[c].fillna("Unknown")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_fraud_preprocessor(
        drop_high_cardinality=drop_high_cardinality,
        high_cardinality_threshold=high_cardinality_threshold,
    )
    return X_train, X_test, y_train, y_test, preprocessor


# -----------------------------
# Quick smoke test
# -----------------------------
if __name__ == "__main__":
    # Credit card
    Xc_tr, Xc_te, yc_tr, yc_te, prep_c = get_creditcard_splits()
    print("[CREDIT] Train/Test:", Xc_tr.shape, Xc_te.shape)

    # Fraud
    Xf_tr, Xf_te, yf_tr, yf_te, prep_f = get_fraud_splits()
    print("[FRAUD]  Train/Test:", Xf_tr.shape, Xf_te.shape)
