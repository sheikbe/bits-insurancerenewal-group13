"""
Streamlit app for Insurance Renewal Prediction.
This is an integrated version that combines the UI and backend logic into a single file for Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
import time
import json
import shutil
import logging
import sys
import math
import numpy as np
from pathlib import Path
import traceback

# Import your ML core
from insrenew import MLContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Initialize paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_FILE = MODELS_DIR / "artifacts.json"
FINAL_SUMMARY = RESULTS_DIR / "final_model_selection_summary.json"

# Create directories if they don't exist
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def sanitize_for_json(obj):
    """Recursively convert NaN, Inf, -Inf to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

def _read_final_selection():
    """Return parsed final model selection JSON or None."""
    if FINAL_SUMMARY.exists():
        try:
            with open(FINAL_SUMMARY, "r") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None

def save_uploaded_file(uploadedfile, dest_path):
    """Save uploaded file to destination path."""
    with open(dest_path, "wb") as f:
        f.write(uploadedfile.getvalue())
    return dest_path

def run_training(train_file):
    """Run the complete training pipeline."""
    status_log = []
    start_ts = time.time()
    
    # Save uploaded train file
    train_path = DATA_DIR / "train_uploaded.csv"
    try:
        save_uploaded_file(train_file, train_path)
        status_log.append({"step": "upload", "msg": f"Saved training file to {train_path}", "ts": time.time()})
    except Exception as e:
        st.error(f"Failed to save uploaded training file: {e}")
        return None

    # Run pipeline
    try:
        ctx = MLContext(base_dir=str(BASE_DIR))
        ctx.set_df(pd.read_csv(train_path))
        
        status_log.append({"step": "set_df", "msg": "Loaded training dataframe", "shape": ctx.df.shape, "ts": time.time()})

        # EDA
        with st.spinner("Running EDA..."):
            try:
                ctx.run_eda()
                status_log.append({"step": "eda", "msg": "EDA completed and plots saved", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "eda", "msg": f"EDA failed: {e}", "ts": time.time()})

        # Feature engineering
        with st.spinner("Running feature engineering..."):
            try:
                ctx.run_fe()
                status_log.append({"step": "fe", "msg": "Feature engineering complete", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "fe", "msg": f"Feature engineering failed: {e}", "ts": time.time()})

        # Feature config & preprocessor
        with st.spinner("Building preprocessor..."):
            try:
                ctx.prepare_feature_config()
                ctx.build_preprocessor()
                status_log.append({"step": "preproc", "msg": "Preprocessor built", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "preproc", "msg": f"Preprocessor build failed: {e}", "ts": time.time()})

        # Train/test split
        try:
            ctx.split_train_test(test_size=0.2)
            status_log.append({"step": "split", "msg": "Train/test split completed", "X_train": str(ctx.X_train.shape), "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "split", "msg": f"Train/test split failed: {e}", "ts": time.time()})
            raise

        # Train models
        trained = {}
        with st.spinner("Training models..."):
            try:
                # train LR
                try:
                    lr_model = ctx.train_lr()
                    trained['lr'] = True
                    status_log.append({"step": "train_lr", "msg": "Logistic Regression trained", "ts": time.time()})
                except Exception as e:
                    status_log.append({"step": "train_lr", "msg": f"LR train failed: {e}", "ts": time.time()})

                # train XGB
                try:
                    xgb_model = ctx.train_xgboost()
                    trained['xgb'] = True
                    status_log.append({"step": "train_xgb", "msg": "XGBoost/HGB trained", "ts": time.time()})
                except Exception as e:
                    status_log.append({"step": "train_xgb", "msg": f"XGB train failed: {e}", "ts": time.time()})

                # train NN
                try:
                    nn_model = ctx.train_neural_net(epochs=5)
                    trained['nn'] = True
                    status_log.append({"step": "train_nn", "msg": "Neural Net trained", "ts": time.time()})
                except Exception as e:
                    status_log.append({"step": "train_nn", "msg": f"NN train failed: {e}", "ts": time.time()})

                # Optional models
                try:
                    ctx.train_balanced_random_forest()
                    trained['brf'] = True
                    status_log.append({"step": "train_brf", "msg": "Balanced RF trained", "ts": time.time()})
                except Exception as e:
                    status_log.append({"step": "train_brf", "msg": f"BRF train failed or skipped: {e}", "ts": time.time()})

            except Exception as e:
                status_log.append({"step": "train_all", "msg": f"Model training failed: {e}", "ts": time.time()})
                raise

        # Save models and artifacts
        with st.spinner("Saving models and artifacts..."):
            try:
                ctx.prepare_artifacts()
                saved = ctx.save_models(prefix="trained")
                artifacts = {
                    'artifacts': {k: str(v) for k, v in saved.items()},
                    'feature_names': ctx.artifacts.get('feature_names', []),
                }
                with open(ARTIFACTS_FILE, "w") as fh:
                    json.dump(artifacts, fh, indent=2)
                status_log.append({"step": "save_models", "msg": f"Saved model artifacts: {list(saved.keys())}", "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "save_models", "msg": f"Save models failed: {e}", "ts": time.time()})

        # Metrics and final selection
        with st.spinner("Aggregating metrics..."):
            try:
                ctx.aggregate_model_comparison_metrics()
                summary_path = ctx.prepare_final_model_selection_summary()
                status_log.append({"step": "aggregate_metrics", "msg": "Aggregated model metrics and created final selection summary", "path": str(summary_path), "ts": time.time()})
            except Exception as e:
                status_log.append({"step": "aggregate_metrics", "msg": f"Aggregation/selection failed: {e}", "ts": time.time()})

        # SHAP values
        try:
            ctx.save_shap_values_csv(model_name='xgb')
            status_log.append({"step": "shap", "msg": "Saved SHAP values", "ts": time.time()})
        except Exception as e:
            status_log.append({"step": "shap", "msg": f"SHAP save failed: {e}", "ts": time.time()})

        # Finalize
        elapsed = time.time() - start_ts
        status_log.append({"step": "done", "msg": "Training pipeline finished", "elapsed_sec": elapsed, "ts": time.time()})

        # Get chosen model
        final = _read_final_selection()
        chosen = None
        if final:
            try:
                chosen = final.get('chosen_model')
            except Exception:
                pass

        return {"status_log": status_log, "trained_models": trained, "chosen_model": chosen}

    except Exception as e:
        tb = traceback.format_exc()
        status_log.append({"step": "error", "msg": str(e), "traceback": tb})
        return {"status_log": status_log, "error": str(e)}

def run_test_predictions(test_file, model_override=None):
    """Run predictions on test data."""
    try:
        # Save test file
        file_path = DATA_DIR / test_file.name
        save_uploaded_file(test_file, file_path)
        logger.info(f"Uploaded test file saved to: {file_path}")

        # Run predictions
        ctx = MLContext(base_dir=str(BASE_DIR))
        test_results = ctx.predict_external_test(str(file_path))
        
        # Save test dataset for later single predictions
        ctx.test_df = pd.read_csv(file_path)
        
        return sanitize_for_json(test_results)
    
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Test prediction failed: {e}\n{tb}")
        raise Exception(f"Prediction failed: {e}")

def predict_single(customer_dict):
    """Predict for a single customer."""
    try:
        ctx = MLContext(base_dir=str(BASE_DIR))

        # Align to training schema if available
        feature_path = ctx.results_dir / "feature_names.npy"
        if feature_path.exists():
            expected_features = list(np.load(feature_path, allow_pickle=True))
            logger.info(f"Aligning input to training schema: {len(expected_features)} features")
            for col in expected_features:
                if col not in customer_dict:
                    customer_dict[col] = None
            # Remove extra keys not seen during training
            customer_dict = {k: customer_dict[k] for k in expected_features if k in customer_dict}
        else:
            logger.warning("No feature schema found; using raw input as-is.")

        # Make prediction
        result = ctx.predict_single(customer_dict=customer_dict)
        return sanitize_for_json(result)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Prediction failed: {e}\n{tb}")
        raise Exception(f"Prediction failed: {e}")

# Streamlit UI
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
tab = st.tabs(["Train", "Test", "Predict"])

# TRAIN TAB
with tab[0]:
    st.header("1) Upload Training CSV and Train Models")
    train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train_file")

    if st.button("Start Training"):
        if train_file is None:
            st.error("Please upload a training CSV first.")
        else:
            with st.spinner("Training models... this may take several minutes"):
                try:
                    result = run_training(train_file)
                    if result:
                        st.session_state["train_result"] = result
                        if "chosen_model" in result:
                            st.session_state["chosen_model"] = result["chosen_model"]
                        st.success("Training completed")
                        st.write("Status log:")
                        st.json(result["status_log"])
                except Exception as e:
                    st.error(f"Training failed: {e}")

    # Show last train result
    if st.session_state["train_result"]:
        st.subheader("Last training result")
        st.write("Chosen model:", st.session_state.get("chosen_model"))
        st.json(st.session_state["train_result"])

# TEST TAB
with tab[1]:
    st.header("2) Upload Test CSV and Run Predictions")
    st.write("Chosen model from training:", st.session_state.get("chosen_model"))
    test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test_file")
    model_override = st.text_input("Model override (optional, e.g., xgb, lr, nn, brf, eec, tab)", value="", key="model_override")

    if st.button("Run Test Predictions"):
        if test_file is None:
            st.error("Please upload a test CSV first.")
        else:
            with st.spinner("Running predictions..."):
                try:
                    result = run_test_predictions(test_file, model_override if model_override else None)
                    st.success("Test predictions created")
                    st.write("Model used:", result.get("model"))
                    st.write("Predictions file:", result.get("output_path"))
                    preview = result.get("preview", [])
                    if preview:
                        st.dataframe(pd.DataFrame(preview))
                    
                    # Store test_df for Predict tab
                    st.session_state["test_df"] = pd.read_csv(test_file)
                    st.info("Uploaded test file stored for single-ID predictions.")
                except Exception as e:
                    st.error(f"Test predictions failed: {e}")

# PREDICT TAB
with tab[2]:
    st.header("3) Single Customer Prediction")
    st.write("You can either select an ID from uploaded test CSV or manually enter feature values.")

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
                    result = predict_single(row)
                    st.success(f"Prediction: {result.get('prediction')} (prob={result.get('prob')}) using model {result.get('model')}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("Uploaded test CSV doesn't have an 'id' column.")

    st.subheader("Or: Manually enter a customer's features")
    st.write("Enter JSON-like dictionary of feature_name: value (e.g., {\"age_years\": 45, \"Income\": 42000, ...})")
    cust_text = st.text_area("Customer feature dict", height=200, value='{"age_years": 45, "Income": 42000}')
    if st.button("Predict for manual customer"):
        try:
            cust = json.loads(cust_text)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
        else:
            try:
                result = predict_single(cust)
                st.success(f"Prediction: {result.get('prediction')} (prob={result.get('prob')}) using model {result.get('model')}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
