import pandas as pd
from src.data_preprocessing import get_creditcard_splits, get_fraud_splits

def test_credit_split_shapes():
    X_tr, X_te, y_tr, y_te, pre = get_creditcard_splits()
    assert len(X_tr) > 0 and len(X_te) > 0
    assert pre is not None

def test_fraud_split_shapes():
    X_tr, X_te, y_tr, y_te, pre = get_fraud_splits()
    assert len(X_tr) > 0 and len(X_te) > 0
    assert pre is not None
