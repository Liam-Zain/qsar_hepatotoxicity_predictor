from pydantic import BaseModel
from typing import List, Optional

class SMILESRequest(BaseModel):
    smiles: str

class BatchSMILESRequest(BaseModel):
    smiles_list: List[str]

class PredictionResponse(BaseModel):
    smiles: str
    prediction: Optional[int]
    probability: Optional[float]
    is_hepatotoxic: Optional[bool]
    likelihood: Optional[str]