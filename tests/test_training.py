from src.model_training import train_and_save_credit, train_and_save_fraud
from src.config import MODEL_CREDIT, MODEL_FRAUD

def test_train_credit(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODEL_CREDIT", tmp_path / "credit.pkl")
    train_and_save_credit("lr")
    assert (tmp_path / "credit.pkl").exists()

def test_train_fraud(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODEL_FRAUD", tmp_path / "fraud.pkl")
    train_and_save_fraud("rf")
    assert (tmp_path / "fraud.pkl").exists()
