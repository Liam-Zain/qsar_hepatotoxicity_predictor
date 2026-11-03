import numpy as np
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from rdkit import Chem
from qsar_hepatotoxicity_predictor.features import get_morgan_fingerprint
from qsar_hepatotoxicity_predictor.api.schemas import SMILESRequest, BatchSMILESRequest, PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

class LogRequest(BaseModel):
    message: str
    level: str = "INFO"

class LogLevelRequest(BaseModel):
    level: str

def get_likelihood(probability: float) -> str:
    if probability >= 70: return "Most likely hepatotoxic"
    elif probability >= 50: return "Likely hepatotoxic"
    elif probability >= 30: return "Likely not hepatoxic"
    else: return "Most likely not hepatotoxic"

def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None

# Loading the model when the service starts
@router.get("/")
def read_root():
    return {"message": "Welcome to the QSAR Hepatotoxicity Predictor API"}

# Endpoint used to predict hepatotoxicity for a single molecule
@router.post("/predict/", response_model=PredictionResponse)
def predict_single(request: SMILESRequest, app_request: Request):
    model = app_request.app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    if not is_valid_smiles(request.smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string. Could not parse molecule.")

    try:
        fp= get_morgan_fingerprint(request.smiles)
        X_fp = fp.reshape(1, -1)
        prediction = int(model.predict(X_fp)[0])
        probability = float(np.round(model.predict_proba(X_fp)[0][1] * 100, 2))
            
        return PredictionResponse(
            smiles=request.smiles,
            prediction=prediction,
            probability=probability,
            is_hepatotoxic=(prediction == 1),
            likelihood=get_likelihood(probability)
        )
    except HTTPException as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}") 

# Endpoint used to predict hepatotoxicity for a list of SMILES strings.
@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchSMILESRequest, app_request: Request):
    model = app_request.app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    results = [None] * len(request.smiles_list)
    valid_smiles = []
    valid_indices = []

    for i, smile in enumerate(request.smiles_list):
        if is_valid_smiles(smile):
            valid_smiles.append(smile)
            valid_indices.append(i)
        else:
            logger.warning(f"Invalid SMILES skipped in batch: {smile}")

    if valid_smiles:
        fps= get_morgan_fingerprint(valid_smiles)   
        predictions = model.predict(fps)
        probabilities = np.round(model.predict_proba(fps)[:, 1] * 100, 2)

        for i, smile in enumerate(valid_smiles):
            pred = int(predictions[i])
            prob = float(probabilities[i])
            results[valid_indices[i]]= PredictionResponse(
                smiles=smile,
                prediction=pred,
                probability=prob,
                is_hepatotoxic=(pred == 1),
                likelihood=get_likelihood(prob)
        )

    for i, res in enumerate(results):
        if res is None:
            results[i] = PredictionResponse(
                smiles=request.smiles_list[i],
                prediction=None,
                probability=None,
                is_hepatotoxic=False,
                likelihood="Not applicable. Invalid SMILES."
            )
        
    return results

# Endpoint for log message
@router.post("/log")
def create_log(request: LogRequest):
    log_level = getattr(logging, request.level.upper(), None)
    if not isinstance(log_level, int):
        raise HTTPException(status_code=400, detail="Invalid log level. Options are DEBUG, INFO, WARNING, ERROR, CRITICAL.")

    logger.log(log_level, request.message)
    return {"message": "Log created successfully", "message": f"Log recorded at {request.level.upper()} level."}

# Endpoint for reading the last N lines from the log file
@router.get("/logs")
def read_logs(lines: int = 50):
    try:
        with open('app.log', 'r') as file:
            log_lines = file.readlines()
            return {"logs": [line.strip() for line in log_lines[-lines:]]}
    except FileNotFoundError:
        return {"message": "Log file not found.", "logs": []}

# Endpoint for dynamic log level change
@router.put("/log_level")
def update_log_level(request: LogLevelRequest):
    log_level = getattr(logging, request.level.upper(), None)
    if not isinstance(log_level, int):
        raise HTTPException(status_code=400, detail="Invalid log level. Options are DEBUG, INFO, WARNING, ERROR, CRITICAL.")

    logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)

    return {"status": "success", "message": f"Log level updated to {request.level.upper()}."}

# Endpoint for checking the API's health status
@router.get("/health")
def health_check():
    return {"status": "ok", "message": "API is healthy."}