# inspect_sensitivity.py
# PURPOSE: Business-sense sanity check on sensitivity tiers
# NOTE: category_prior already feeds the score directly (weight 0.50),
# so the category crosstab partially confirms the score is doing what we 
# built it to do — not fully independent validation, but useful for 
# explainability and catching anything that looks economically wrong

import pandas as pd
from pathlib import Path
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

PROCESSED = Path(cfg["paths"]["processed"])


def load_data():
    sensitivity = pd.read_parquet(PROCESSED / "sensitivity_scores.parquet")
    product_features = pd.read_parquet(PROCESSED / "product_features.parquet")
    return sensitivity, product_features


def merge_with_category(sensitivity, product_features):
    # YOUR TASK: merge sensitivity with product_features[["item_id","store_id","cat_id"]]
    # on item_id, store_id
    prod = product_features[['item_id','store_id']]
    merged = sensitivity.merge(prod, on =['item_id','store_id'], how = 'left')
    return merged


def show_top_products(merged, n=15):
    cols = ["item_id", "store_id", "cat_id", "pricing_sensitivity_score", 
            "sensitivity_tier", "mean_elasticity", "snap_sensitivity", "category_prior"]
    
    print(f"=== TOP {n} HIGH SENSITIVITY PRODUCTS ===")
    print(merged[cols].sort_values("pricing_sensitivity_score", ascending=False).head(n).to_string())
    
    print(f"\n=== TOP {n} LOW SENSITIVITY PRODUCTS ===")
    print(merged[cols].sort_values("pricing_sensitivity_score", ascending=True).head(n).to_string())


def category_breakdown(merged):
    # YOUR TASK: build a crosstab of cat_id vs sensitivity_tier
    # hint: pd.crosstab(merged["cat_id"], merged["sensitivity_tier"])
    crosstab = pd.crosstab(merged["cat_id"], merged["sensitivity_tier"])
    
    print("\n=== CATEGORY BREAKDOWN BY TIER (counts) ===")
    print(crosstab)
    
    print("\n=== CATEGORY BREAKDOWN BY TIER (% within category) ===")
    print(crosstab.div(crosstab.sum(axis=1), axis=0).round(3))


def run():
    print("Loading data...")
    sensitivity, product_features = load_data()
    
    print("Merging with category info...")
    merged = merge_with_category(sensitivity, product_features)
    
    show_top_products(merged)
    category_breakdown(merged)


if __name__ == "__main__":
    run()