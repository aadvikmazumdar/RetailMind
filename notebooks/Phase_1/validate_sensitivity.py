# validate_sensitivity.py
# PURPOSE: Prove pricing_sensitivity_score predicts real price response,
# and diagnose which signals drive tier assignment (explainability, not tuning)

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.etl.m5_utils import load_raw, melt_sales

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

PROCESSED = Path(cfg["paths"]["processed"])


def load_melted():
    sales, calendar, prices = load_raw()
    melted = melt_sales(sales, calendar, prices)
    return melted


def compute_real_price_response(melted):
    melted = melted.sort_values(["item_id", "store_id", "date"])

    melted["price_lag1"] = melted.groupby(["item_id", "store_id"])["sell_price"].shift(1)
    melted["sales_lag1"] = melted.groupby(["item_id", "store_id"])["units_sold"].shift(1)

    melted["price_pct_change"] = (melted["sell_price"] - melted["price_lag1"]) / (melted["price_lag1"])
    melted["sales_pct_change"] = (melted["units_sold"] - melted["sales_lag1"]) / (melted["sales_lag1"] + 1e-6)

    # 0.02 threshold: real median price change is ~2%, 0.1% let in noise
    # price_lag1 > 0.50: excludes near-zero-price rows producing absurd ratios
    real_price_changes = melted[
        (melted["price_pct_change"].abs() > 0.02) &
        (melted["price_lag1"] > 0.50)
    ].copy()

    print(f"Rows with real, meaningful price change: {len(real_price_changes)}")

    real_price_changes["observed_real_elasticity"] = (
        abs(real_price_changes["sales_pct_change"]) / abs(real_price_changes["price_pct_change"])
    )
    real_price_changes["observed_real_elasticity"] = (
        real_price_changes["observed_real_elasticity"].clip(0, 20)
    )

    return real_price_changes


def join_sensitivity_tiers(real_price_changes):
    sensitivity = pd.read_parquet(PROCESSED / "sensitivity_scores.parquet")

    merged = real_price_changes.merge(
        sensitivity[[
            "item_id", "store_id", "sensitivity_tier",
            "pricing_sensitivity_score",
            "elasticity_norm", "snap_norm", "volatility_norm", "category_norm"
        ]],
        on=["item_id", "store_id"],
        how="left"
    )

    return merged


def run_validation(merged):
    validation = merged.groupby("sensitivity_tier", observed=True)["observed_real_elasticity"].agg(
        median_elasticity="median",
        mean_elasticity="mean",
        n_observations="count"
    )

    print("\n=== VALIDATION RESULTS ===")
    print(validation)

    print("\n=== TIER COUNTS ===")
    print(merged["sensitivity_tier"].value_counts())

    print("\n=== TIER SCORE SEPARATION ===")
    print(merged.groupby("sensitivity_tier", observed=True)["pricing_sensitivity_score"].describe())

    print("\n=== SIGNAL CONTRIBUTION BY TIER (normalized signal means) ===")
    signal_cols = ["elasticity_norm", "snap_norm", "volatility_norm", "category_norm"]
    print(merged.groupby("sensitivity_tier", observed=True)[signal_cols].mean())

    print("\n=== WEIGHTED CONTRIBUTION BY TIER ===")
    merged["elasticity_contrib"] = 0.20 * merged["elasticity_norm"]
    merged["snap_contrib"] = 0.25 * merged["snap_norm"]
    merged["volatility_contrib"] = 0.25 * merged["volatility_norm"]
    merged["category_contrib"] = 0.30 * merged["category_norm"]
    contrib_cols = ["elasticity_contrib", "snap_contrib", "volatility_contrib", "category_contrib"]
    print(merged.groupby("sensitivity_tier", observed=True)[contrib_cols].mean())

    print("\n=== PRICE CHANGE MAGNITUDE BY TIER (confounder check) ===")
    print(merged.groupby("sensitivity_tier", observed=True)["price_pct_change"].apply(lambda x: x.abs().median()))

    return validation


def run():
    print("Loading melted data...")
    melted = load_melted()

    print("Computing real price response...")
    real_price_changes = compute_real_price_response(melted)

    print("Joining sensitivity tiers...")
    merged = join_sensitivity_tiers(real_price_changes)

    print("Running validation...")
    validation = run_validation(merged)

    out_path = PROCESSED / "validation_results_v2.csv"
    validation.to_csv(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()