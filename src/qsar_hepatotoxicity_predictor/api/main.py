from fastapi import FastAPI
import logging
import joblib
from contextlib import asynccontextmanager
from qsar_hepatotoxicity_predictor.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers= [
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = joblib.load("models/hepatotoxicity_model.pkl")
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        app.state.model = None
    yield
    logging.info("Shutting down API")

app = FastAPI(
    title="QSAR Hepatotoxicity Predictor",
    description="API for predicting the potential hepatotoxicity of molecules using SMILES string",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)