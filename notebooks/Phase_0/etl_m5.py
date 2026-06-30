# etl_m5.py
# PURPOSE: Load raw M5 data, engineer features, save to parquet
# This is the foundation — every downstream model reads from this output

import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW = PROJECT_ROOT / cfg["paths"]["raw_m5"]
OUT = PROJECT_ROOT / cfg["paths"]["processed"]
OUT.mkdir(parents=True, exist_ok=True)

# WHY yaml config: hardcoding paths breaks when others clone your repo
# config.yaml is the single source of truth for all paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW = PROJECT_ROOT / cfg["paths"]["raw_m5"]
OUT = PROJECT_ROOT / cfg["paths"]["processed"]
OUT.mkdir(parents=True, exist_ok=True)

def download_m5():
    # WHY kaggle CLI: programmatic download is reproducible
    # anyone can clone repo and run this to get exact same data
    # BEFORE THIS WORKS: accept competition terms at kaggle.com/competitions/m5-forecasting-accuracy
    os.system(f'kaggle competitions download -c m5-forecasting-accuracy -p {RAW}')
    os.system(f'unzip -q {RAW}/m5-forecasting-accuracy.zip -d {RAW}')


def load_raw():
    # WHY these three files specifically:
    # sales_train_evaluation — actual unit sales per product per day (1941 days)
    # sell_prices — price per product per store per week (not every day, weekly granularity)
    # calendar — maps day columns (d_1, d_2...) to actual dates + events + SNAP flags
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    prices = pd.read_csv(RAW / "sell_prices.csv")
    calendar = pd.read_csv(RAW / "calendar.csv")
    return sales, prices, calendar


def melt_sales(sales, calendar, prices):
    # WHY melt: sales data is wide format — one column per day (d_1 to d_1941)
    # ML models need long format — one row per product per day
    # this is the single most expensive operation in the ETL — 3M+ rows after melt
    
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    
    # YOUR TASK: write the melt operation here
    # hint: pd.melt with id_vars=id_cols, value_vars=day_cols
    # name the new columns "d" and "units_sold"
    melted = pd.melt(
        sales,
        id_vars = id_cols,
        value_vars = day_cols,
        var_name = 'd',
        value_name = 'units_sold'
    )
 
    # WHY merge calendar: we need actual dates, weekday, month, event info
    # wm_yr_wk is the walmart week number — needed to join prices
    melted = melted.merge(
        calendar[["d", "date", "wday", "month", "year", "wm_yr_wk",
                  "event_name_1", "snap_CA", "snap_TX", "snap_WI"]],
        on="d"
    )
    
    # WHY merge prices here not separately:
    # price is weekly not daily — joining on store_id + item_id + wm_yr_wk
    # gives us the price that was active during that week for that product
    melted = melted.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    
    melted["date"] = pd.to_datetime(melted["date"])
    melted["units_sold"] = melted["units_sold"].fillna(0)
    
    return melted


def build_product_features(melted, day_cols):
    # WHY these aggregations:
    # mean_daily_sales — baseline demand signal
    # std_daily_sales — demand volatility, high std = unpredictable product
    # zero_sale_rate — % of days with no sales, proxy for deadstock risk
    # price stats — how much has this product's price moved historically
    
    # YOUR TASK: write the groupby aggregation
    # group by item_id and store_id
    # aggregate: mean, std, sum of units_sold
    # aggregate: count of zero sales days
    # aggregate: mean, std, min, max of sell_price
    product_stats = melted.groupby(['item_id','store_id']).agg(
        units_sold_mean = ('units_sold','mean'),
        units_sold_std = ('units_sold','std'),
        units_sold_sum = ('units_sold','sum'),
        zero_sales_days = ('units_sold',lambda x:(x == 0).sum()),
        sell_price_mean = ('sell_price','mean'),
        sell_price_std = ('sell_price','std'),
        sell_price_min = ('sell_price','min'),
        sell_price_max = ('sell_price','max')
    ).reset_index()
    # WHY merge cat_id here: category is a property of item_id, constant across
    # stores — needed downstream for category-prior elasticity assumptions
    cat_lookup = melted[['item_id', 'cat_id']].drop_duplicates()
    product_stats = product_stats.merge(cat_lookup, on='item_id', how='left')
    
    # WHY these derived features:
    # zero_sale_rate normalizes zero_sale_days to [0,1] regardless of history length
    # price_volatility = std/mean (coefficient of variation) — scale independent measure
    # price_range_pct — how wide has the price band been, proxy for markdown history
    product_stats["zero_sales_rate"] = product_stats["zero_sales_days"] / len(day_cols)
    product_stats["price_volatility"] = (
        product_stats["sell_price_std"] / product_stats["sell_price_mean"]
    )
    product_stats["price_range_pct"] = (
        (product_stats["sell_price_max"] - product_stats["sell_price_min"])
        / product_stats["sell_price_mean"]
    )
    
    return product_stats


def build_elasticity_features(melted):
    # WHY elasticity matters for pricing engine:
    # elasticity tells us how sensitive demand is to price changes
    # high elasticity (< -1) = customers will leave if you raise price
    # low elasticity (> -1) = customers will buy regardless of small price changes
    # this directly drives how aggressively we recommend markdowns
    
    melted = melted.sort_values(["item_id", "store_id", "date"])
    
    # WHY shift(1): we need previous week's price to compute % change
    # groupby ensures we don't leak across products or stores
    melted["price_lag1"] = melted.groupby(["item_id", "store_id"])["sell_price"].shift(1)
    melted["sales_lag1"] = melted.groupby(["item_id", "store_id"])["units_sold"].shift(1)
    
    # YOUR TASK: compute price_pct_change and sales_pct_change
    # price_pct_change = (current - lag) / lag
    # sales_pct_change = (current - lag) / (lag + 1e-6)
    # WHY 1e-6: avoid division by zero when previous sales were 0
    melted["price_pct_change"] = ((melted['sell_price'] - melted['price_lag1'])/melted['price_lag1']) 
    melted["sales_pct_change"] = (melted['units_sold'] - melted['sales_lag1'])/(melted['sales_lag1']+ 1e-6)
    
    # WHY median not mean for elasticity:
    # point elasticity is noisy — outlier price changes skew the mean badly
    # median is robust to those outliers
    # WHY clip(-10, 0): elasticity should be negative (price up = demand down)
    # values beyond -10 are data artifacts not real behavior
    melted["point_elasticity"] = np.where(
        melted["price_pct_change"].abs() > 0.001,
        melted["sales_pct_change"] / melted["price_pct_change"],
        np.nan  # WHY nan: ignore days with no price change — elasticity undefined
    )
    
    elasticity = melted.groupby(["item_id", "store_id"]).agg(
        mean_elasticity=("point_elasticity", "median"),
        elasticity_std=("point_elasticity", "std"),
        n_price_changes=("price_pct_change", lambda x: (x.abs() > 0.001).sum())
    ).reset_index()
    
    elasticity["mean_elasticity"] = elasticity["mean_elasticity"].clip(-10, 0)
    
    return elasticity


def build_seasonality_features(melted):
    # WHY seasonality: retail demand is heavily seasonal
    # knowing peak demand month per product helps markdown timing
    # don't markdown milk in November if December is peak month
    
    # YOUR TASK: 
    # 1. group by item_id, store_id, month — get mean units_sold per month
    # 2. group by item_id, store_id — get overall mean units_sold
    # 3. seasonality_index = monthly_avg / overall_avg
    # 4. find peak_demand_month = month where seasonality_index is highest
    monthly_avg = melted.groupby(['item_id','store_id','month'])['units_sold'].mean().reset_index()
    monthly_avg = monthly_avg.rename(columns ={'units_sold':'monthly_avg_sales'})

    overall_avg = melted.groupby(['item_id','store_id'])['units_sold'].mean().reset_index()
    overall_avg = overall_avg.rename(columns ={'units_sold':'overall_avg_sales'})

    merged = monthly_avg.merge(overall_avg, on=['item_id','store_id'])
    merged['seasonality_index'] = merged["monthly_avg_sales"] / (merged["overall_avg_sales"] + 1e-6)

    peak_month = (
        merged.sort_values("seasonality_index", ascending=False)
        .groupby(["item_id", "store_id"])
        .first()
        .reset_index()[["item_id", "store_id", "month"]]
        .rename(columns={"month": "peak_demand_month"})
    )
    
    return peak_month


def run():
    if not (RAW / "sales_train_evaluation.csv").exists():
        download_m5()

    print("Loading raw M5...")
    sales, prices, calendar = load_raw()
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    print("Melting sales to long format...")
    melted = melt_sales(sales, calendar, prices)

    print("Building product features...")
    product_stats = build_product_features(melted, day_cols)

    print("Building elasticity features...")
    elasticity = build_elasticity_features(melted)

    print("Building seasonality features...")
    peak_month = build_seasonality_features(melted)

    # WHY multiple merges not one big join:
    # each function is independently testable
    # if elasticity breaks you don't rerun the whole pipeline
    product_features = product_stats.merge(elasticity, on=["item_id", "store_id"])
    product_features = product_features.merge(peak_month, on=["item_id", "store_id"])

    # WHY parquet not csv:
    # columnar storage — downstream models read only the columns they need
    # 5-10x smaller than CSV for this data
    # preserves dtypes — no re-casting on load
    out_path = OUT / "product_features.parquet"
    product_features.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | shape: {product_features.shape}")


if __name__ == "__main__":
    run()