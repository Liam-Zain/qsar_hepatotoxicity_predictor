import numpy as np
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qsar_hepatotoxicity_predictor.features import get_morgan_fingerprint
import joblib

from qsar_hepatotoxicity_predictor.api.schemas import SMILESRequest, BatchSMILESRequest, PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

class LogRequest(BaseModel):
    message: str
    level: str = "INFO"

class LogLevelRequest(BaseModel):
    level: str

# Loading the model when the service starts
try:
    model = joblib.load("models/hepatotoxicity_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@router.get("/")
def read_root():
    return {"message": "Welcome to the QSAR Hepatotoxicity Predictor API"}

# Endpoint used to predict hepatotoxicity for a single molecule
@router.post("/predict/", response_model=PredictionResponse)
def predict_single(request: SMILESRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        fp= get_morgan_fingerprint(request.smiles)
        X_fp = fp.reshape(1, -1)
        prediction = int(model.predict(X_fp)[0])
        probability = float(np.round(model.predict_proba(X_fp)[0][1] * 100, 2))

        if probability >= 70:
            likelihood = "Most likely hepatotoxic"
        elif probability >= 50:
            likelihood = "Likely hepatotoxic"
        elif probability >= 30:
            likelihood = "Likely not hepatoxic"
        else:
            likelihood = "Most likely not hepatotoxic"
            
        return PredictionResponse(
            smiles=request.smiles,
            prediction=prediction,
            probability=probability,
            is_hepatotoxic=(prediction == 1),
            likelihood=likelihood
        )
    except HTTPException as e:
           raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}") 

# Endpoint used to predict hepatotoxicity for a list of SMILES strings.
@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchSMILESRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    results = []
    for smile in request.smiles_list:
        try:
            fp= get_morgan_fingerprint(smile)   
            X_fp = fp.reshape(1, -1)
            prediction = int(model.predict(X_fp)[0])
            probability = float(np.round(model.predict_proba(X_fp)[0][1] * 100, 2))
            
            if probability >= 70:
                likelihood = "Most likely hepatotoxic"
            elif probability >= 50:
                likelihood = "Likely hepatotoxic"
            elif probability >= 30:
                likelihood = "Likely not hepatoxic"
            else:
                likelihood = "Most likely not hepatotoxic"

            results.append(PredictionResponse(
            smiles=smile,
            prediction=prediction,
            probability=probability,
            is_hepatotoxic=(prediction == 1),
            likelihood=likelihood
        ))

        except ValueError as e:
            print("Error processing SMILES:", e)
            results.append(PredictionResponse(
                    smiles=smile,
                    prediction=None,
                    probability=None,
                    is_hepatotoxic=False,
                    likelihood="Not applicable"
                ))
            
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