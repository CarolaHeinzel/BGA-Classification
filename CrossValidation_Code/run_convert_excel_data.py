"""Requires openpyxl and pandas to be installed."""

from __future__ import annotations

import pandas as pd

df = pd.DataFrame(pd.read_excel("data/filtered_population_eur.xlsx"))
df = df.drop(
    columns=[
        "Unnamed: 1",
        "SNP-Indel with complex genotype listing in VCF (not compiled)",
        "Unnamed: 3",
        "Unnamed: 4",
        "ID",
    ],
)
df = df.sample(frac=1, random_state=42)
df.to_csv("data/filtered_population_eur.csv", index=False)
