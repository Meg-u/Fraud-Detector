from pydantic import BaseModel
from typing import Dict, Any

class FraudRequest(BaseModel):
    features: Dict[str, Any]  # a single transaction as feature dict

class CreditRequest(BaseModel):
    features: Dict[str, Any]
