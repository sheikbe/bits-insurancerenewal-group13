import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import requests
from typing import Dict, Any
import logging
import sys
from insrenew import MLContext

# Constants
API_URL = "http://localhost:8000"  # FastAPI server URL

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
                        response = requests.post(f"{API_URL}/train", json=train_dict)
                        response.raise_for_status()  # Raise an exception for bad status codes
                        result = response.json()
                        
                        if "error" in result:
                            st.error(f"Training failed: {result['error']}")
                        else:
                            st.session_state["train_result"] = result
                            if result.get("chosen_model"):
                                st.session_state["chosen_model"] = result["chosen_model"]
                            st.success("Training completed")
                            st.write("Status log:")
                            st.json(result["status_log"])
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
                        response = requests.post(f"{API_URL}/test", json=test_dict)
                        response.raise_for_status()
                        result = response.json()
                        
                        st.success("Test predictions created")
                        st.write("Predictions file:", result["predictions_path"])
                        
                        if result.get("preview"):
                            st.dataframe(pd.DataFrame(result["preview"]))
                        
                        st.session_state["test_df"] = test_data
                        st.info("Test data stored for single predictions")

                        # Display metrics
                        st.write("Model evaluation metrics:")
                        st.json(result["metrics"])
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
                        response = requests.post(f"{API_URL}/predict", json=row)
                        response.raise_for_status()
                        result = response.json()
                        st.success(f"Prediction: {result['prediction']} (prob={result['prob']:.3f}) using model {result['model']}")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

        st.subheader("Or: Manually enter customer features")
        st.write("Enter feature values as JSON")
        cust_text = st.text_area("Customer feature dict", height=200, value='{"age_years": 45, "Income": 42000}')
        
        if st.button("Predict for manual customer"):
            try:
                cust = json.loads(cust_text)
                response = requests.post(f"{API_URL}/predict", json=cust)
                response.raise_for_status()
                result = response.json()
                st.success(f"Prediction: {result['prediction']} (prob={result['prob']:.3f}) using model {result['model']}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()