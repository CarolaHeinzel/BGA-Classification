
import json
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.base import BaseEstimator


def get_dataset() -> [pd.DataFrame, pd.Series, list[int]]:
    """Read in data for experiment."""

    train_data_path = "data_train.csv"  # Change it to your training data
    test_data_path = "data_test.csv"  # Change it to your data with unkown labels

    data_train = pd.read_csv(train_data_path)
    data_test = pd.read_csv(test_data_path)

    data_train = data_train.astype("category")
    data_test = data_test.astype("category")
    X_test = data_test
    X_train = data_train.drop(columns=["Population"])
    y_train = data_train["Population"]
    print(X_train, y_train)
    print(X_test)
    # Indizes der kategorischen Merkmale finden
    categorical_features_indices = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype.name == "category"]
    print(categorical_features_indices)
    return X_train, y_train, X_test, categorical_features_indices


def get_models(categorical_features_indices: list[int]) -> dict[str, BaseEstimator]:
    """Sklearn models to compare."""
    return {
         # TabPFN Modell aus dem Nature Paper
        "TabPFN": TabPFNClassifier(
            random_state=np.random.RandomState(42),  # Kontrolle des Zufallsgenerators
            categorical_features_indices=categorical_features_indices  # Definiere, welche Merkmale kategorisch sind
        ),
    }


def run_single_split(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, BaseEstimator],
) -> tuple[pd.DataFrame, list[dict]]:
    """Train and evaluate models on a single train-test split."""

    raw_predictions = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)


        raw_predictions.append(
            {
                "Model": model_name,
                "Predictions": y_pred.tolist(),
                "Predictions Probabilities": y_pred_proba.tolist(),
            },
        )

    return raw_predictions


def run_experiments():
    """Run our experiments."""
    X_train, y_train, X_test, categorical_features_indices = get_dataset()

    models = get_models(categorical_features_indices=categorical_features_indices)
    raw_predictions = run_single_split(X_train=X_train, X_test=X_test, y_train=y_train, models=models)
    print(raw_predictions)
    with open("results_examples.json", "w") as f:
        json.dump(raw_predictions, f)



if __name__ == "__main__":
    run_experiments()