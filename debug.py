import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/model_features.parquet")

print("Units sold distribution:")
print(df["units_sold"].describe())

print("\nZero sale rows %:", (df["units_sold"] == 0).mean() * 100)

print("\nSell price nulls:", df["sell_price"].isna().sum())
print("Units sold lag1 nulls:", df["units_sold_lag1"].isna().sum())
print("Units sold roll4 nulls:", df["units_sold_roll4"].isna().sum())