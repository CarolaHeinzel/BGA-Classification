
import json
import io
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
!pip install tabpfn
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

def get_dataset() -> [pd.DataFrame, pd.Series, list[int]]:
    """Read in data for experiment."""
    #file_content = uploaded['df_new_results.csv']
    file_content = uploaded['df_new_results.csv']
    # Create a file-like object using io.BytesIO
    file_object = io.BytesIO(file_content)
    # Read the CSV data using pd.read_csv, passing the file-like object
    data = pd.read_csv(file_object)
    # Convert features to categorical but leave the target variable as is
    X = data.drop(columns=["Population"]).astype("category")  # Only convert features
    y = data["Population"]  # Leave target variable as is
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Convert y back to a Pandas Series
    y = pd.Series(y, index=data.index) #This line was added
    categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype.name == "category"]
    return X, y, categorical_features_indices


def get_models(categorical_features_indices: list[int]) -> dict[str, BaseEstimator]:
    """Sklearn models to compare."""
    return {
        "TabPFN": TabPFNClassifier(
           random_state=np.random.RandomState(42) # Control Random Seed
        ),
                    # Random Forest Classifier
          "Random Forest": Pipeline(
                [
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values from ordinal encoding
                    ("classifier", RandomForestClassifier(random_state=42)),  # Control Random Seed
                ],
            ),
            # XGBoost Classifier
            "XGBoost": Pipeline(
                [
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values from ordinal encoding
                    ("classifier", XGBClassifier(random_state=42, verbosity=0)),
                ],
            ),
    }


def run_cross_val(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, BaseEstimator],
) -> tuple[pd.DataFrame, list[dict]]:
    """Run cross-validation for the models and compute various metrics."""
    # Cross-validation setup
    # Folds: 5, because this allows us to have for the minor class (Russian) at least 5 sample in each test fold
    # Repeats: 10, to have a more robust estimate of the model performance
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    # Store results
    results = []
    raw_predictions = []
    print(results)
    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        print(f"==== Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        for model_name, model in models.items():
            print(f"Running {model_name}")
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            # Store the experiment results
            results.append(
                {
                    "Fold": fold,
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                    "ROC AUC": roc_auc_score(y_test, y_pred_proba, multi_class="ovr"),
                    "Log Loss": log_loss(y_test, y_pred_proba),
                },
            )
            raw_predictions.append(
                {
                    "Fold": fold,
                    "Model": model_name,
                    "Predictions": y_pred.tolist(),
                    "Predictions Probabilities": y_pred_proba.tolist(),
                    "True Labels": y_test.tolist(),
                },
            )

    return pd.DataFrame(results), raw_predictions


def run_experiments():
    """Run our experiments."""
    X, y, categorical_features_indices = get_dataset()
    models = get_models(categorical_features_indices=categorical_features_indices)
    results_df, raw_predictions = run_cross_val(X=X, y=y, models=models)

    # Save results to disk
    results_df.to_csv("/content/eva_XGBoos_RF_eur.csv", index=False)
    with open("/content/eva_XGBoos_RF_eur.json", "w") as f:
        json.dump(raw_predictions, f)


if __name__ == "__main__":
    run_experiments()
