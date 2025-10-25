import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import plotly.express as px
from pathlib import Path
from typing import Dict, Any
import logging
import sys
from insrenew import MLContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("app")

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("app")



# Streamlit UI
def main():
    st.set_page_config(page_title="Insurance Renewal ML App", layout="wide")
    st.title("Insurance Renewal — Training / Testing / Single Prediction")

    # Initialize session state
    if "train_result" not in st.session_state:
        st.session_state["train_result"] = None
    if "test_df" not in st.session_state:
        st.session_state["test_df"] = None
    if "chosen_model" not in st.session_state:
        st.session_state["chosen_model"] = None

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Train", "Test", "Predict"])

    # Train Tab
    with tab1:
        st.header("1) Upload Training CSV and Train Models")
        train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train_file")

        if st.button("Start Training"):
            if train_file is None:
                st.error("Please upload a training CSV first.")
            else:
                with st.spinner("Training models... this may take several minutes"):
                    # Read CSV and call training endpoint
                    train_data = pd.read_csv(train_file)
                    # Convert DataFrame to dict for FastAPI
                    train_dict = train_data.to_dict(orient='list')
                    
                    try:
                        df = pd.DataFrame(train_dict)
                        train_path = DATA_DIR / "train_uploaded.csv"
                        df.to_csv(train_path, index=False)
                        
                        ctx = MLContext(base_dir=str(BASE_DIR))
                        ctx.set_df(df)
                        
                        # Run EDA
                        status_log = []
                        try:
                            ctx.run_eda()
                            status_log.append({"step": "eda", "msg": "EDA completed", "ts": time.time()})
                        except Exception as e:
                            status_log.append({"step": "eda", "msg": f"EDA failed: {e}", "ts": time.time()})
                        
                        # Feature engineering
                        try:
                            ctx.run_fe()
                            ctx.prepare_feature_config()
                            ctx.build_preprocessor()
                            status_log.append({"step": "fe", "msg": "Feature engineering complete", "ts": time.time()})
                        except Exception as e:
                            status_log.append({"step": "fe", "msg": f"Feature engineering failed: {e}", "ts": time.time()})
                        
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
                                status_log.append({"step": f"train_{model_name}", "msg": f"{model_name} trained successfully", "ts": time.time()})
                            except Exception as e:
                                status_log.append({"step": f"train_{model_name}", "msg": f"{model_name} training failed: {e}", "ts": time.time()})
                        
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
                        
                        result = {"status_log": status_log, "trained_models": trained, "chosen_model": chosen}
                        st.session_state["train_result"] = result
                        if chosen:
                            st.session_state["chosen_model"] = chosen
                        st.success("Training completed")
                        st.write("Status log:")
                        st.json(status_log)
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

        if st.session_state["train_result"]:
            st.subheader("Last training result")
            st.write("Chosen model:", st.session_state.get("chosen_model"))
            st.json(st.session_state["train_result"])

    # Test Tab
    with tab2:
        st.header("2) Upload Test CSV and Run Predictions")
        st.write("Chosen model from training:", st.session_state.get("chosen_model"))
        test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test_file")
        model_override = st.text_input("Model override (optional)", value="", key="model_override")

        if st.button("Run Test Predictions"):
            if test_file is None:
                st.error("Please upload a test CSV first.")
            else:
                with st.spinner("Running predictions..."):
                    test_data = pd.read_csv(test_file)
                    # Convert DataFrame to dict for FastAPI
                    test_dict = test_data.to_dict(orient='list')
                    
                    try:
                        df_test = pd.DataFrame(test_dict)
                        test_path = DATA_DIR / "test_uploaded.csv"
                        df_test.to_csv(test_path, index=False)
                        
                        ctx = MLContext(base_dir=str(BASE_DIR))
                        ctx.set_test_df(df_test)
                        
                        # Load artifacts and models
                        with open(ARTIFACTS_FILE, "r") as f:
                            artifacts = json.load(f)
                            
                        model_paths = {k: Path(v) for k, v in artifacts.get("artifacts", {}).items()}
                        feature_names = artifacts.get("feature_names", [])
                        
                        if not model_paths or not feature_names:
                            raise ValueError("Missing model paths or feature names")
                            
                        # Load preprocessor and models
                        ctx.load_preprocessor()
                        ctx.load_models(model_paths)
                        
                        # Run predictions for all models
                        predictions = {}
                        metrics = {}
                        
                        for model_name in ctx.models:
                            try:
                                preds = ctx.predict(model_name)
                                probs = ctx.predict_proba(model_name)
                                
                                predictions[model_name] = {
                                    'predictions': preds.tolist(),
                                    'probabilities': probs.tolist()
                                }
                                
                                # Save predictions
                                pred_df = pd.DataFrame({
                                    'prediction': preds,
                                    'probability': probs
                                })
                                pred_path = RESULTS_DIR / f"{model_name}_predictions.csv"
                                pred_df.to_csv(pred_path, index=False)
                                
                                # Preview
                                if model_name == st.session_state.get("chosen_model", "brf"):
                                    preview_df = pd.concat([
                                        df_test,
                                        pd.DataFrame({
                                            'prediction': preds,
                                            'probability': probs
                                        })
                                    ], axis=1)
                                    st.dataframe(preview_df.head())
                                
                            except Exception as e:
                                st.warning(f"Prediction failed for {model_name}: {e}")
                        
                        st.session_state["test_df"] = test_data
                        st.info("Test data stored for single predictions")
                        
                        # Display prediction summaries
                        st.write("Prediction files saved to results directory")
                    except Exception as e:
                        st.error(f"Testing failed: {str(e)}")

    # Predict Tab
    with tab3:
        st.header("3) Single Customer Prediction")
        st.write("Select an ID from uploaded test data or enter feature values manually")

        test_df = st.session_state.get("test_df")
        if test_df is not None:
            st.subheader("Select ID from uploaded test set")
            if 'id' in test_df.columns:
                selected_id = st.selectbox("Choose ID", test_df['id'].astype(str).tolist())
                if st.button("Use selected ID for prediction"):
                    row = test_df[test_df['id'].astype(str) == selected_id].iloc[0].to_dict()
                    st.write("Selected row data:")
                    st.json(row)
                    
                    try:
                        ctx = MLContext(base_dir=str(BASE_DIR))
                        df_single = pd.DataFrame([row])
                        ctx.set_test_df(df_single)
                        
                        with open(ARTIFACTS_FILE, "r") as f:
                            artifacts = json.load(f)
                        
                        model_paths = {k: Path(v) for k, v in artifacts.get("artifacts", {}).items()}
                        ctx.load_preprocessor()
                        ctx.load_models(model_paths)
                        
                        model = model_override if model_override else st.session_state.get("chosen_model", "brf")
                        if model not in ctx.models:
                            st.error(f"Model {model} not available. Using balanced random forest.")
                            model = "brf"
                        
                        pred = ctx.predict(model)[0]
                        prob = ctx.predict_proba(model)[0]
                        
                        st.success(f"Prediction: {pred} (prob={prob:.3f}) using model {model}")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

        st.subheader("Or: Manually enter customer features")
        st.write("Enter feature values as JSON")
        cust_text = st.text_area("Customer feature dict", height=200, value='{"age_years": 45, "Income": 42000}')
        
        if st.button("Predict for manual customer"):
            try:
                cust = json.loads(cust_text)
                ctx = MLContext(base_dir=str(BASE_DIR))
                df_single = pd.DataFrame([cust])
                ctx.set_test_df(df_single)
                
                with open(ARTIFACTS_FILE, "r") as f:
                    artifacts = json.load(f)
                
                model_paths = {k: Path(v) for k, v in artifacts.get("artifacts", {}).items()}
                ctx.load_preprocessor()
                ctx.load_models(model_paths)
                
                model = model_override if model_override else st.session_state.get("chosen_model", "brf")
                if model not in ctx.models:
                    st.error(f"Model {model} not available. Using balanced random forest.")
                    model = "brf"
                
                pred = ctx.predict(model)[0]
                prob = ctx.predict_proba(model)[0]
                
                st.success(f"Prediction: {pred} (prob={prob:.3f}) using model {model}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()