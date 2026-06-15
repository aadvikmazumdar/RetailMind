import pandas as pd
import numpy as np
from pathlib import Path
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW = Path(cfg["paths"]["raw_m5"])
PROCESSED = Path(cfg["paths"]["processed"])

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


def load_melted():
    # WHY recompute melt: we didn't save the 59M row melted table in Phase 0
    # (too large to store as-is). Re-run the same melt + merge logic here.
    # YOUR TASK: reuse load_raw() and melt_sales() logic from etl_m5.py
    # Either import them directly, or copy the two functions here
    sales, prices, calendar = load_raw()
    melted = melt_sales(sales, calendar, prices)
    return melted


def resample_weekly(melted):
    # WHY weekly: daily data is 59M rows, too large for fast iteration
    # weekly aggregation reduces this to ~8M rows while preserving 
    # enough signal for demand prediction
    
    # YOUR TASK:
    # group by item_id, store_id, wm_yr_wk (the week ID already in the data)
    # aggregate: sum of units_sold (total weekly demand)
    #            mean of sell_price (price during that week)
    #            first of month, year (just need one value per week)
    weekly = melted.groupby(['item_id','store_id','wm_yr_wk']).agg(
        units_sold = ('units_sold','sum'),
        sell_price = ('sell_price','mean'),
        first_month = ('month','first'),
        first_year = ('year','first') 
    ).reset_index()
    return weekly


def add_lag_features(weekly):
    # WHY lag features: the model needs to know "what happened last week"
    # to predict "what happens this week" — demand is autocorrelated
    
    weekly = weekly.sort_values(["item_id", "store_id", "wm_yr_wk"])
    
    # YOUR TASK:
    # create units_sold_lag1 = previous week's units_sold (groupby + shift)
    # create sell_price_lag1 = previous week's price (groupby + shift)
    weekly["units_sold_lag1"] = weekly.groupby(['item_id','store_id'])['units_sold'].shift(1)
    weekly['units_sold_lag2'] = weekly.groupby(['item_id','store_id'])['units_sold'].shift(2)
    weekly['units_sold_lag4'] = weekly.groupby(['item_id','store_id'])['units_sold'].shift(4)
    weekly["sell_price_lag1"] = weekly.groupby(['item_id','store_id'])['sell_price'].shift(1)
    weekly["price_vs_lag_ratio"] = weekly["sell_price"] / (weekly["sell_price_lag1"] + 1e-6)
    weekly ['price_change_pct'] = (weekly['sell_price'] - weekly['sell_price_lag1'])/(weekly['sell_price_lag1']+ 1e-6)
    weekly['demand_growth_pct'] = (weekly['units_sold_lag1'] - weekly['units_sold_lag2']) / (weekly['units_sold_lag2'] + 1e-6)
    return weekly


def add_rolling_features(weekly):
    # WHY rolling average: smooths out noisy week-to-week fluctuations,
    # gives the model a sense of recent trend not just last week's value
    
    # YOUR TASK:
    # create units_sold_roll4 = 4-week rolling mean of units_sold
    # (groupby item_id/store_id, then rolling(4).mean(), watch for shift 
    #  to avoid leakage — rolling mean should only use PAST weeks)
    weekly["units_sold_roll4"] = weekly.groupby(['item_id','store_id'])['units_sold'].shift(1).rolling(4).mean()
    weekly['units_sold_roll8'] = weekly.groupby(['item_id','store_id'])['units_sold'].shift(1).rolling(8).mean()
    weekly['units_sold_std4'] = weekly.groupby(['item_id','store_id'])['units_sold'].shift(1).rolling(4).std()
    return weekly


def merge_static_features(weekly):
    # WHY merge: bring in elasticity and peak_demand_month from Phase 0's 
    # product_features.parquet — these don't change over time, so they're 
    # "static" features per product/store
    product_features = pd.read_parquet(PROCESSED / "product_features.parquet")
    
    # YOUR TASK: merge weekly with product_features on item_id, store_id
    # only bring in: mean_elasticity, peak_demand_month, zero_sales_rate
    selective = product_features[[
        'item_id',
        'store_id',
        'mean_elasticity',
        'peak_demand_month',
        'zero_sales_rate'
    ]]
    merged = weekly.merge(selective,on = ['item_id','store_id'],how = 'left')
    merged["is_peak_month"] = (merged["first_month"] == merged["peak_demand_month"]).astype(int)
    return merged


def run():
    print("Loading melted sales...")
    melted = load_melted()
    
    print("Resampling to weekly...")
    weekly = resample_weekly(melted)
    
    print("Adding lag features...")
    weekly = add_lag_features(weekly)
    
    print("Adding rolling features...")
    weekly = add_rolling_features(weekly)
    
    print("Merging static features...")
    weekly = merge_static_features(weekly)
    
    # drop rows with NaN from lag/rolling features (first few weeks per product)
    weekly = weekly.dropna()
    
    out_path = PROCESSED / "model_features.parquet"
    weekly.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | shape: {weekly.shape}")


if __name__ == "__main__":
    run()