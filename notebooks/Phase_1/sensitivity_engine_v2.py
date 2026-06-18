# sensitivity_engine.py
# PURPOSE: Combine 4 weak signals into one pricing_sensitivity_score per item/store
# This REPLACES raw mean_elasticity as the pricing guardrail input

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.etl.m5_utils import load_raw, melt_sales 

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

PROCESSED = Path(cfg["paths"]["processed"])
MODELS = Path("models")

# WHY these specific priors: published retail economics elasticity ranges
# per M5 category (FOODS/HOBBIES/HOUSEHOLD), used when no other signal is strong
# negative = normal economic behavior (price up -> demand down)
CATEGORY_PRIORS = {
    "FOODS": -0.5,
    "HOBBIES": -1.6,
    "HOUSEHOLD": -0.7,
}

# WHY these weights: SNAP and category prior trusted more than raw elasticity
# (which we know is noisy due to 1.47% price change frequency)
WEIGHTS = {
    "elasticity": 0.30,
    "snap": 0.10,
    "volatility": 0.10,
    "category_prior": 0.50,
}


def load_melted_with_cat():
    # WHY recompute melt again: need cat_id + snap flags at row level
    # which only exist in the melted (not aggregated) data
    # YOUR TASK: copy load_raw() + melt_sales() from etl_m5.py here
    # (same pattern as etl_model_features.py did)
    sales , calendar, prices = load_raw()
    melted = melt_sales(sales,calendar,prices)
    return melted
    
def compute_snap_sensitivity(melted):
    # WHY map state from store_id: snap_CA/snap_TX/snap_WI are state-specific
    # but we need ONE snap flag per row matching that row's actual store state
    
    # YOUR TASK: extract state from store_id
    # hint: store_id.str.split('_').str[0]  -> gives "CA", "TX", "WI"
    melted["state"] = melted.store_id.str.split("_").str[0]
    
    # YOUR TASK: pick the correct snap column value per row
    # hint: use np.select or a loop with np.where for each state
    # if state == "CA": use snap_CA column value
    # if state == "TX": use snap_TX column value  
    # if state == "WI": use snap_WI column value
    melted["snap_flag"] = np.select(
        [melted["state"] == "CA", melted["state"] == "TX", melted["state"] == "WI"],
        [melted["snap_CA"], melted["snap_TX"], melted["snap_WI"]],
        default=0
    )
    
    # WHY ratio not difference: ratio is scale-independent across products
    # of very different volumes (a product selling 5/day vs 500/day)
    snap_stats = melted.groupby(["item_id", "store_id", "snap_flag"])["units_sold"].mean().unstack()
    snap_stats = snap_stats.rename(
    columns={
        0: "non_snap_avg",
        1: "snap_avg"
    })
    snap_stats["non_snap_avg"] = snap_stats["non_snap_avg"].fillna(0)
    snap_stats["snap_avg"] = snap_stats["snap_avg"].fillna(0)
    snap_stats = snap_stats.reset_index()
    
    # YOUR TASK: compute snap_sensitivity = snap_avg / (non_snap_avg + 1e-6)
    snap_stats["snap_sensitivity"] = snap_stats.snap_avg / (snap_stats.non_snap_avg + 1e-6)
    
    return snap_stats[["item_id", "store_id", "snap_sensitivity"]]


def compute_residual_volatility(melted):
    # WHY load the trained model: we need ACTUAL prediction errors,
    # not raw demand variance, to isolate "unexplained" volatility
    
    # This requires loading model_features.parquet (has all engineered features)
    # and running the trained LightGBM model on it to get residuals
    
    model = lgb.Booster(model_file=str(MODELS / "demand_model.lgb"))
    model_features = pd.read_parquet(PROCESSED / "model_features.parquet")
    
    FEATURE_COLS = [
        "sell_price", "units_sold_lag1", "units_sold_lag2", "units_sold_lag4",
        "sell_price_lag1", "units_sold_roll4", "units_sold_roll8", "units_sold_std4",
        "price_vs_lag_ratio", "price_change_pct", "demand_growth_pct",
        "mean_elasticity", "peak_demand_month", "zero_sales_rate",
        "first_month", "first_year",
    ]
    
    # YOUR TASK: get predictions, reverse log transform, compute residual
    # hint: preds = np.expm1(model.predict(model_features[FEATURE_COLS]))
    # residual = actual - predicted
    preds = np.expm1(model.predict(model_features[FEATURE_COLS]))
    actual = model_features['units_sold']
    model_features["residual"] = actual - preds
    
    # WHY std of residual not mean: we care about HOW INCONSISTENT 
    # the errors are, not their average direction
    volatility = model_features.groupby(["item_id", "store_id"])["residual"].std().reset_index()
    volatility.columns = ["item_id", "store_id", "residual_volatility"]
    
    return volatility


def add_category_prior(product_features):
    # WHY map from cat_id: category priors are per M5 category, 
    # need to attach to every item_id/store_id row
    
    # YOUR TASK: map product_features["cat_id"] to CATEGORY_PRIORS dict
    # hint: product_features["cat_id"].map(CATEGORY_PRIORS)
    product_features["category_prior"] = product_features['cat_id'].map(CATEGORY_PRIORS)
    
    return product_features


def normalize_signal(series):
    # WHY min-max not z-score: we want everything strictly in [0,1]
    # so weighted sum is interpretable and bounded
    
    # YOUR TASK: min-max normalize
    # formula: (x - min) / (max - min + 1e-6)
    return ( series - series.min()) / (series.max() - series.min() + 1e-6)


def combine_signals(elasticity_df, snap_df, volatility_df):
    # merge all 4 signal dataframes on item_id, store_id
    merged = elasticity_df.merge(snap_df, on=["item_id", "store_id"], how="left")
    merged = merged.merge(volatility_df, on=["item_id", "store_id"], how="left")
    merged = merged.fillna(0)
    
    # WHY normalize elasticity as abs(): more negative = more elastic,
    # but we want "more sensitive = higher score" consistently across all signals
    merged["elasticity_norm"] = normalize_signal(merged["mean_elasticity"].abs())
    merged["snap_norm"] = normalize_signal(merged["snap_sensitivity"])
    merged["volatility_norm"] = normalize_signal(merged["residual_volatility"])
    merged["category_norm"] = normalize_signal(merged["category_prior"].abs())
    
    # YOUR TASK: compute weighted sum using WEIGHTS dict
    # pricing_sensitivity_score = 
    #   WEIGHTS["elasticity"] * elasticity_norm +
    #   WEIGHTS["snap"] * snap_norm +
    #   WEIGHTS["volatility"] * volatility_norm +
    #   WEIGHTS["category_prior"] * category_norm
    merged["pricing_sensitivity_score"] = (WEIGHTS["elasticity"] * merged['elasticity_norm'] +
    WEIGHTS["snap"] * merged['snap_norm'] +
    WEIGHTS["volatility"] * merged['volatility_norm'] +
    WEIGHTS["category_prior"] * merged['category_norm'])
    
    # YOUR TASK: assign sensitivity_tier based on tercile (33%/66% cutoffs)
    # hint: use pd.qcut(merged["pricing_sensitivity_score"], q=3, labels=["LOW","HIGH"])
    threshold = merged['pricing_sensitivity_score'].quantile(0.67)
    merged['sensitivity_tier'] = np.where(merged['pricing_sensitivity_score'] >= threshold,"HIGH","LOW")
    
    return merged


def run():
    print("Loading melted data...")
    melted = load_melted_with_cat()
    
    print("Computing SNAP sensitivity...")
    snap_df = compute_snap_sensitivity(melted)
    
    print("Computing residual volatility...")
    volatility_df = compute_residual_volatility(melted)
    
    print("Loading elasticity + category data...")
    product_features = pd.read_parquet(PROCESSED / "product_features.parquet")
    
    print("Adding category priors...")
    product_features = add_category_prior(product_features)
    
    print("Combining all signals...")
    sensitivity = combine_signals(product_features, snap_df, volatility_df)
    
    out_path = PROCESSED / "sensitivity_scores.parquet"
    sensitivity.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | shape: {sensitivity.shape}")
    print(sensitivity["sensitivity_tier"].value_counts())


if __name__ == "__main__":
    run()