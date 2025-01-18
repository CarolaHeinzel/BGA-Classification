# This code was built with TabPFN.

import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.base import BaseEstimator
from tabpfn import TabPFNClassifier

# Function to load data
def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[int]]:
    """Load data: from upload or repository."""
    st.sidebar.write("### Select Data Source")
    data_source = st.sidebar.radio(
        "Load training and test data from:",
        ("Example data from repository", "Upload your own CSV files"),
    )

    if data_source == "Upload your own CSV files":
        train_file = st.sidebar.file_uploader("Upload training data (CSV)", type="csv")
        test_file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")

        if train_file is not None and test_file is not None:
            data_train = pd.read_csv(train_file).astype("category")
            data_test = pd.read_csv(test_file).astype("category")
        else:
            st.warning("Please upload both training and test data.")
            return None, None, None, None
    else:
        # Example data paths (replace with actual repository paths)
        train_file = "data_train.csv"
        test_file = "data_test.csv"
        data_train = pd.read_csv(train_file).astype("category")
        data_test = pd.read_csv(test_file).astype("category")

    # Separate features and target for training data
    X_train = data_train.drop(columns=["Population"])
    y_train = data_train["Population"]

    # Use all test data as features (remove target column if it exists)
    if "Population" in data_test.columns:
        X_test = data_test.drop(columns=["Population"])
    else:
        X_test = data_test

    # Identify indices of categorical features
    categorical_features_indices = [
        i for i, col in enumerate(X_train.columns) if X_train[col].dtype.name == "category"
    ]

    return X_train, y_train, X_test, categorical_features_indices


# Function to initialize models
def get_models(categorical_features_indices: list[int]) -> dict[str, BaseEstimator]:
    """Initialize models for comparison."""
    return {
        "TabPFN": TabPFNClassifier(
            random_state=np.random.RandomState(42),  # Set random seed
            categorical_features_indices=categorical_features_indices,  # Specify categorical features
        ),
    }


# Function to train and predict
def run_single_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, BaseEstimator],
) -> list[dict]:
    """Train and evaluate models on a single train-test split."""
    raw_predictions = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        raw_predictions.append(
            {
                "Model": model_name,
                "Predicted Class": y_pred.tolist(),
                "Prediction Probabilities": y_pred_proba.tolist(),
            },
        )

    return raw_predictions


# Streamlit interface
def main():
    st.title("TabPFN Classifier Experiment")

    # Load data
    st.sidebar.write("## Step 1: Load Data")
    data = load_data()

    if data is None:
        return

    X_train, y_train, X_test, categorical_features_indices = data

    # Display data previews
    st.write("### Training Data Preview")
    st.dataframe(X_train)

    st.write("### Test Data Preview")
    st.dataframe(X_test)

    # Initialize models
    st.sidebar.write("## Step 2: Run Experiment")
    if st.sidebar.button("Run Experiment"):
        st.write("### Results")
        models = get_models(categorical_features_indices)
        predictions = run_single_split(X_train, X_test, y_train, models)

        # Display predictions
        for result in predictions:
            st.write(f"#### Model: {result['Model']}")
            st.write("Predicted Classes:", result["Predicted Class"])
            st.write("Prediction Probabilities:", result["Prediction Probabilities"])

        # Option to save results
        if st.button("Save Results"):
            with open("results_predictions.json", "w") as f:
                json.dump(predictions, f)
            st.success("Results saved as `results_predictions.json`")


if __name__ == "__main__":
    main()

