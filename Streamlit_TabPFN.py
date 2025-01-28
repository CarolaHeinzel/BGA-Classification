# This code was built with TabPFN.

import streamlit as st
import pandas as pd
import numpy as np
# import json
from sklearn.base import BaseEstimator
from tabpfn import TabPFNClassifier
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}") 

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
        # Get the class order
        classes = model.classes_
        raw_predictions.append(
            {
                "Model": model_name,
                "Classes": classes.tolist(),
                "Predicted Class": y_pred.tolist(),
                "Prediction Probabilities": y_pred_proba.tolist(),
            },
        )

    return raw_predictions

def display_progress_bar(probability, class_name):
    """Display progress bar with probability and class name."""
    # Display probability as text above the progress bar
    st.markdown(f"<div style='text-align: center; color: blue;'> Prediction Probability of {class_name}: {probability * 100:.2f}% </div>", unsafe_allow_html=True)


# Streamlit interface
def main():
    st.title("TabPFN Classifier Experiment")
    
    st.write("This is a graphical user interface to classify individuals into populations based on their genotypes.")
    st.write("It uses TabPFN by Hollmann et al. (2025), which was published in *Nature*.")
    st.write("If a GPU is available, the code will automatically utilize it, significantly speeding up the process.")
    st.write("It is important that you use the correct format as presented in the Training Data Preview and Test Data Preview.")
    st.write("The comparison of TabPFN to other classification methods has been described in [our paper](https://example.com).")
        
    # Load data
    st.sidebar.write("## Step 1: Load Data")
    data = load_data()

    if data is None:
        return

    X_train, y_train, X_test, categorical_features_indices = data

    # Display data previews
    st.write("### Training Data Preview")
    data_all = pd.concat([y_train, X_train], axis=1)
    st.dataframe(data_all)

    st.write("### Test Data Preview")
    st.dataframe(X_test)

    # Initialize models
    st.sidebar.write("## Step 2: Run Experiment")
    if st.sidebar.button("Run Experiment"):
        with st.spinner("The computer is calculating..."):
            st.write("### Results")
            models = get_models(categorical_features_indices)
            predictions = run_single_split(X_train, X_test, y_train, models)
            all_predictions = []
            for result in predictions:
                st.write("Predicted Ancestral Populations:")
                prob_data =[]
                for i, (pred_class, probabilities) in enumerate(zip(result["Predicted Class"], result["Prediction Probabilities"])):
                    st.write(f"TabPFN classifies individual {i} into {pred_class}.")
            # Display predictions
            for result in predictions:
                st.write("Predicted Probabilities:")
                prob_data =[]
                for i, (pred_class, probabilities) in enumerate(zip(result["Predicted Class"], result["Prediction Probabilities"])):
                    st.write(f"Prediction Probabilities for individual {i}:")
                    for j, prob in enumerate(probabilities):  # Iterate over class probabilities
                        st.write(f"Population {result['Classes'][j]}:")
                        st.progress(int(prob * 100))  
                        prob_data.append({
                            "Index of the individual": i,
                            "Class": result['Classes'][j],
                            "Probability": prob
                                })
                        class_name = result['Classes'][j]
                        display_progress_bar(prob, class_name)
    
                all_predictions.append(prob_data)
                # Flatten the list of dictionaries (one dictionary per model prediction)
            flat_predictions = [item for sublist in all_predictions for item in sublist]
        
        # Convert the flattened list of dictionaries to a DataFrame
            predictions_df = pd.DataFrame(flat_predictions)
            st.download_button(
                label="Download Prediction Probabilities as CSV",
                data=predictions_df.to_csv(index=False),
                file_name="prediction_probabilities.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()

