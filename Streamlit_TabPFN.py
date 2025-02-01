# This code was built with TabPFN.

import streamlit as st
import pandas as pd
import numpy as np
# import json
from sklearn.base import BaseEstimator
from tabpfn import TabPFNClassifier
import torch
#import matplotlib.pyplot as plt

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
        if train_file is None or test_file is None:
            st.stop()
        if train_file is not None and test_file is not None:
            data_train = pd.read_csv(train_file).astype("category")
            data_test = pd.read_csv(test_file).astype("category")
        else:
            st.warning("Please upload both training and test data.")
            return None, None, None, None
    else:
        # Example data paths (replace with actual repository paths)
        train_file = "train_data_eur.csv"
        test_file = "test_data_eur.csv"
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
    pp = []
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
        pp.append(y_pred_proba.tolist())
    return raw_predictions, classes, pp

def display_progress_bar(probability, class_name):
    """Display progress bar with probability and class name."""
    # Display probability as text above the progress bar
    st.markdown(f"<div style='text-align: center; color: blue;'> Prediction Probability of {class_name}: {probability * 100:.2f}% </div>", unsafe_allow_html=True)


def main():
    st.title("The TabPFN Classifier for Biogeographical Ancestry")
    st.write("This is a Graphical User Interface for the usage of [TabPFN](https://github.com/PriorLabs/tabpfn#-license) as a classifier as performed in [Advancing Biogeographical Ancestry Predictions Through Machine Learning](xxx) by Heinzel, Purucker, Hutter and Pfaffelhuber.")
    st.write("The full code, which also contains the comparisons to other classifiers, can be found on [github](https://github.com/CarolaHeinzel/BGA-Classification/).")
    st.write("This is a graphical user interface to classify individuals into populations based on their genotypes.")
    st.write("The test and training data are based on data from different data sets which can be found in the Supplemental (1-s2.0-S1872497323000285-mmc5.xlsx) of [Ruiz-Ramirez et al, *Development and evaluations of the ancestry informative markers of the visage enhanced tool for appearance and ancestry*](https://pubmed.ncbi.nlm.nih.gov/36917866/).")
    st.write("The runtime of the code with the example data from the repository is about two minutes without a GPU.")
    st.write("If a GPU is available, the code will automatically utilize it, significantly speeding up the process.")
    st.write("It is important that you use the correct format as presented in the Training Data Preview and Test Data Preview.")
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
            predictions, classes,pp = run_single_split(X_train, X_test, y_train, models)
            st.write(pp)
            all_predictions = []
            df = pd.DataFrame(pp[0], columns=classes)

            for result in predictions:
                st.write("Predicted Ancestral Populations:")
                
                unique_classes = sorted(set(result["Predicted Class"]))  
                prob_data = []
                
                for i, (pred_class, probabilities) in enumerate(zip(result["Predicted Class"], result["Prediction Probabilities"])):
                    st.write(f"TabPFN classifies individual {i} into {pred_class}.")
                    
                    row = {"Individual": i, "Predicted Class": pred_class}
                    for pop, prob in zip(unique_classes, probabilities):
                        row[pop] = prob 
                    
                    prob_data.append(row)
                
                all_predictions.append(prob_data)  #


            st.session_state["predictions_df"] = df
            st.session_state["predictions_display"] = all_predictions

    if "predictions_display" in st.session_state:

        st.download_button(
            label="Download Prediction Probabilities as CSV",
            data=st.session_state["predictions_df"].to_csv(index=False),
            file_name="prediction_probabilities.csv",
            mime="text/csv"
        )



if __name__ == "__main__":
    main()

