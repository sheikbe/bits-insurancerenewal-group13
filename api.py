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
        
        # Run EDA
        try:
            ctx.run_eda()
            status_log.append(StatusLog(step="eda", msg="EDA completed", ts=time.time()))
        except Exception as e:
            status_log.append(StatusLog(step="eda", msg=f"EDA failed: {e}", ts=time.time()))
        
        # Feature engineering
        try:
            ctx.run_fe()
            ctx.prepare_feature_config()
            ctx.build_preprocessor()
            status_log.append(StatusLog(step="fe", msg="Feature engineering complete", ts=time.time()))
        except Exception as e:
            status_log.append(StatusLog(step="fe", msg=f"Feature engineering failed: {e}", ts=time.time()))
        
        # Train/test split
        ctx.split_train_test(test_size=0.2)
        
        # Train models
        trained = {}
        models_to_train = [
            ('lr', ctx.train_lr),
            ('xgb', ctx.train_xgboost),
            ('nn', ctx.train_neural_net),
            ('brf', ctx.train_balanced_random_forest),
            ('eec', ctx.train_easy_ensemble),
            ('lgb', ctx.train_lightgbm),
            ('tab', ctx.train_tabnet)
        ]
        
        for model_name, train_func in models_to_train:
            try:
                train_func()
                trained[model_name] = True
                status_log.append(StatusLog(step=f"train_{model_name}", msg=f"{model_name} trained successfully", ts=time.time()))
            except Exception as e:
                status_log.append(StatusLog(step=f"train_{model_name}", msg=f"{model_name} training failed: {e}", ts=time.time()))
        
        # Save models and artifacts
        ctx.prepare_artifacts()
        saved = ctx.save_models(prefix="trained")
        artifacts = {
            'artifacts': {k: str(v) for k, v in saved.items()},
            'feature_names': ctx.artifacts.get('feature_names', []),
        }
        with open(ARTIFACTS_FILE, "w") as f:
            json.dump(artifacts, f, indent=2, default=str)
        
        # Aggregate metrics and save final selection
        ctx.aggregate_model_comparison_metrics()
        summary_path = ctx.prepare_final_model_selection_summary()
        
        # Get chosen model
        chosen = None
        try:
            with open(FINAL_SUMMARY, "r") as f:
                final = json.load(f)
                chosen = final.get("recommendation", {}).get("recommended_model")
        except Exception:
            pass
        
        return TrainResponse(status_log=status_log, trained_models=trained, chosen_model=chosen)
    
    except Exception as e:
        status_log.append(StatusLog(step="error", msg=str(e), ts=time.time()))
        return TrainResponse(status_log=status_log, trained_models={}, chosen_model=None)
    
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
        
        # Handle NaN values in metrics and preview
        metrics = test_results.get("metrics", {})
        metrics_clean = {}
        for k, v in metrics.items():
            if isinstance(v, float) and np.isnan(v):
                metrics_clean[k] = None
            else:
                metrics_clean[k] = v
        
        preview = test_results.get("preview", [])
        for item in preview:
            for k, v in item.items():
                if isinstance(v, float) and np.isnan(v):
                    item[k] = None
        
        return TestResponse(
            message=test_results.get("message", ""),
            metrics=metrics_clean,
            preview=preview,
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