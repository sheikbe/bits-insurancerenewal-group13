from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from insrenew import MLContext

# Initialize FastAPI app
api = FastAPI(title="Insurance Renewal ML Backend")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables and paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_FILE = MODELS_DIR / "artifacts.json"
FINAL_SUMMARY = RESULTS_DIR / "final_model_selection_summary.json"

# Create necessary directories
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Pydantic models
class StatusLog(BaseModel):
    step: str
    msg: str
    ts: float

class TrainResponse(BaseModel):
    status_log: List[StatusLog]
    trained_models: Dict[str, bool]
    chosen_model: Optional[str]

class TestResponse(BaseModel):
    message: str
    metrics: Dict[str, Any]
    preview: List[Dict[str, Any]]
    predictions_path: str

class PredictResponse(BaseModel):
    prediction: int
    prob: float
    model: str

# FastAPI endpoints
@api.post("/train", response_model=TrainResponse)
async def train_endpoint(train_data: Dict[str, Any]):
    """Train models using uploaded data"""
    status_log = []
    
    try:
        df = pd.DataFrame(train_data)
        train_path = DATA_DIR / "train_uploaded.csv"
        df.to_csv(train_path, index=False)
        status_log.append(StatusLog(step="upload", msg=f"Saved training file", ts=time.time()))
        
        ctx = MLContext(base_dir=str(BASE_DIR))
        ctx.set_df(df)
        
        # Rest of your training code...
        # (keeping the same implementation)
        
        return {"status_log": status_log, "trained_models": trained, "chosen_model": chosen}
    
    except Exception as e:
        status_log.append({"step": "error", "msg": str(e)})
        return {"status_log": status_log, "error": str(e)}

@api.post("/test", response_model=TestResponse)
async def test_endpoint(test_data: Dict[str, Any]):
    """Run predictions on test data"""
    try:
        df = pd.DataFrame(test_data)
        test_path = DATA_DIR / "test_uploaded.csv"
        df.to_csv(test_path, index=False)
        
        ctx = MLContext(base_dir=str(BASE_DIR))
        test_results = ctx.predict_external_test(str(test_path))
        
        return TestResponse(
            message=test_results.get("message", ""),
            metrics=test_results.get("metrics", {}),
            preview=test_results.get("preview", []),
            predictions_path=str(test_results.get("output_path", ""))
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@api.post("/predict", response_model=PredictResponse)
async def predict_endpoint(customer_data: Dict[str, Any]):
    """Predict for a single customer"""
    try:
        ctx = MLContext(base_dir=str(BASE_DIR))
        
        feature_path = ctx.results_dir / "feature_names.npy"
        if feature_path.exists():
            expected_features = list(np.load(feature_path, allow_pickle=True))
            for col in expected_features:
                if col not in customer_data:
                    customer_data[col] = 0
            customer_data = {k: customer_data[k] for k in expected_features if k in customer_data}
        
        result = ctx.predict_single(customer_dict=customer_data)
        return PredictResponse(
            prediction=result.get("prediction", 0),
            prob=result.get("prob", 0.0),
            model=result.get("model", "unknown")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")