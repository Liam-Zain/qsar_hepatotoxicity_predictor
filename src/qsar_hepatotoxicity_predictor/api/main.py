from fastapi import FastAPI
import logging

from qsar_hepatotoxicity_predictor.api.routes import router

app = FastAPI(
    title="QSAR Hepatotoxicity Predictor",
    description="API for predicting the potential hepatotoxicity of molecules using SMILES string",
    version="1.0.0"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers= [
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app.include_router(router)