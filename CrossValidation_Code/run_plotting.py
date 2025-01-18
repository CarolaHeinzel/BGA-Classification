"""Code for plotting the results from our experiments."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_metric(df, metric):
    """Plots a barplot comparing models based on the specified metric."""
    # Ensure the 'plots' directory exists
    Path("plots").mkdir(parents=True, exist_ok=True)

    # Create the barplot
    plt.figure(figsize=(6, 6))
    sns.barplot(data=df, x="Model", y=metric, hue="Model")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.tight_layout()

    # Save the plot to file
    plot_path = Path("plots") / f"{metric.replace(' ', '_')}_comparison.pdf"
    plt.savefig(plot_path)
    plt.show()



if __name__ == "__main__":
    results_df = pd.read_csv("results/results.csv")
    with open("results/raw_predictions.json") as f:
        json.load(f)

    metrics = ["Accuracy", "Balanced Accuracy", "ROC AUC", "Log Loss"]
    for metric in metrics:
        plot_metric(results_df, metric)