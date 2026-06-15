# train_demand_model.py
# PURPOSE: Train LightGBM to predict weekly units_sold per product per store
# Input: model_features.parquet (from etl_model_features.py)
# Output: trained model saved to disk + evaluation metrics

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import yaml
import mlflow
import json

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

PROCESSED = Path(cfg["paths"]["processed"])
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

# features the model uses to predict demand
# WHY these: price + lag + rolling give current context
# elasticity + peak_month + zero_sales_rate give product-level context
# month + year capture seasonality
FEATURE_COLS = [
    "sell_price",
    "units_sold_lag1",
    "units_sold_lag2",
    "units_sold_lag4",
    "sell_price_lag1", 
    "units_sold_roll4",
    "units_sold_roll8",
    "units_sold_std4",
    "price_vs_lag_ratio",
    "price_change_pct",
    "demand_growth_pct",
    "mean_elasticity",
    "peak_demand_month",
    "zero_sales_rate",
    "first_month",
    "first_year",
]

# WHY units_sold as target: we predict demand, then use that to optimize price
TARGET_COL = "units_sold"


def load_features():
    df = pd.read_parquet(PROCESSED / "model_features.parquet")
    return df


def chronological_split(df):
    # WHY chronological not random: time series data — 
    # randomly splitting leaks future data into training
    # use last 8 weeks as test, everything before as train
    
    # YOUR TASK: write the three lines
    # cutoff = max week - 8
    # train = rows where wm_yr_wk <= cutoff
    # test = rows where wm_yr_wk > cutoff
    cutoff = df['wm_yr_wk'].max() - 8
    train = df[df['wm_yr_wk'] <= cutoff]
    test = df[df['wm_yr_wk'] > cutoff]
    
    print(f"Train weeks: {df['wm_yr_wk'].min()} to {cutoff} | rows: {len(train)}")
    print(f"Test weeks: {cutoff+1} to {df['wm_yr_wk'].max()} | rows: {len(test)}")
    
    return train, test


def build_matrices(train, test):
    # WHY LightGBM Dataset: lgb.Dataset is LightGBM's optimized data structure
    # more memory efficient than passing raw pandas DataFrames
    # bins continuous features automatically which speeds up tree building
    
    X_train = train[FEATURE_COLS]
    y_train = np.log1p(train[TARGET_COL])
    X_test = test[FEATURE_COLS]
    y_test_raw = test[TARGET_COL].values
    y_test_log = np.log1p(test[TARGET_COL])
    
    # YOUR TASK: create lgb.Dataset for train and test
    # hint: lgb.Dataset(data, label=target)
    # for test set, pass reference_dataset=train_dataset 
    # WHY reference: ensures test uses same binning as train
    train_dataset = lgb.Dataset(X_train,label = y_train)
    test_dataset = lgb.Dataset(X_test, label = y_test_log, reference = train_dataset) 
    
    return train_dataset, test_dataset, X_test, y_test_raw


def train_model(train_dataset, test_dataset):
    # WHY these params:
    # objective=regression: predicting continuous units_sold not a class
    # metric=rmse: root mean squared error, standard for regression
    # num_leaves=64: controls tree complexity, 64 is safe default
    # learning_rate=0.05: slow but stable convergence
    # feature_fraction=0.8: use 80% of features per tree, reduces overfitting
    # early_stopping_rounds=50: stop if test RMSE doesn't improve for 50 rounds
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 128,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "min_child_samples": 50
    }
    
    # YOUR TASK: train the model
    # hint: lgb.train(params, train_dataset, num_boost_round=500,
    #                 valid_sets=[test_dataset],
    #                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    model = lgb.train(params, train_dataset,num_boost_round=1000,valid_sets=[test_dataset],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(50)])
    
    return model


def evaluate(model, X_test, y_test_raw,train,test):
    # WHY RMSE and MAPE both:
    # RMSE — absolute error in units, tells you average prediction error
    # MAPE — % error, tells you relative accuracy (easier to explain to business)
    
    # YOUR TASK:
    # 1. get predictions: model.predict(X_test)
    # 2. clip predictions to 0 (demand can't be negative)
    # 3. compute RMSE = sqrt(mean((y_test - preds)^2))
    # 4. compute MAPE = mean(|y_test - preds| / (y_test + 1e-6)) * 100
    
    baseline_preds = X_test['units_sold_lag1'].values
    baseline_rmse = np.sqrt(np.mean((y_test_raw - baseline_preds ) ** 2))
    preds = np.expm1(
        model.predict(
            X_test,
            num_iteration = model.best_iteration
        )
    )
    preds_log = model.predict(
        X_test,
        num_iteration = model.best_iteration
    )
    log_rmse = np.sqrt(
        np.mean(
            (np.log1p(y_test_raw) - preds_log) ** 2
        )
    )
    print(f"Log RMSE: {log_rmse:.4f}")

    preds = np.clip(preds,0,None)  # clip negatives

    rmse = np.sqrt(np.mean((y_test_raw - preds) ** 2))
    test_copy = test.copy()
    test_copy['pred'] = preds
    print(f"Baseline RMSE lag(1): {baseline_rmse:.4f}")
    improvement = (
        (baseline_rmse - rmse)
        /baseline_rmse
    )*100
    print(
        f"improvement over baseline:"
        f"{improvement:.2f}%"
    )

    rmsse_values = []
    for(item_id,store_id),train_grp in train.groupby(
        ['item_id','store_id']
    ):
        test_grp = test_copy[(test_copy['item_id'] == item_id) & (test_copy['store_id'] == store_id)]

        if len(test_grp) == 0:
            continue

        train_series = train_grp.sort_values(
            'wm_yr_wk'
        )['units_sold'].values

        if len(train_series) < 2:
            continue

        scale = np.mean(
            np.diff(train_series) **2
        )

        if np.isnan(scale) or scale <= 1e-12:
            continue

        mse = np.mean(
            (
                test_grp['units_sold'].values - test_grp['pred'].values
            ) ** 2
        )
        rmsse_values.append(np.sqrt(mse/scale))
    if len(rmsse_values) > 0:
        rmsse = np.mean(rmsse_values)
    else:
        rmsse = np.nan
    print(f"Test RMSSE: {rmsse:.4f}")

    mask = y_test_raw > 0
    mape = np.mean(np.abs((y_test_raw[mask] - preds[mask])/ (y_test_raw[mask]))) * 100
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test RMSSE: {rmsse:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
    
    return {"rmse": rmse, "mape": mape,"rmsse":rmsse}


def save_model(model, metrics):
    # WHY save both model and metrics together:
    # when you load the model later in pricing_engine.py
    # you want to know how accurate it is before trusting its predictions
    model_path = MODELS / "demand_model.lgb"
    metrics_path = MODELS / "demand_model_metrics.json"
    
    # YOUR TASK:
    # save model: model.save_model(str(model_path))
    # save metrics: json.dump(metrics, open(metrics_path, "w"))
    model.save_model(str(model_path))
    json.dump(metrics,open(metrics_path,'w'))
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")


def run():
    print("Loading features...")
    df = load_features()
    print("\nDemand distribution:")
    print(df[TARGET_COL].describe())
    
    print("Splitting chronologically...")
    train, test = chronological_split(df)
    print("\nTest set zero-sales rate:")
    print(round((test[TARGET_COL]==0).mean() * 100 ,2), "%")
    print("\nBefore filtering:")
    print("Train rows:", len(train))
    print("Zero Sales rows:", (train[TARGET_COL]==0).sum())
    print("Zero sales%:", round((train[TARGET_COL]==0).mean() * 100,2))
    train_full = train.copy()

    print("\nAfter filtering:")
    print("Train rows:", len(train))


    print("Building LightGBM datasets...")
    train_dataset, test_dataset, X_test, y_test_raw = build_matrices(train, test)
    
    print("Training model...")
    model = train_model(train_dataset, test_dataset)
    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importance()
    }).sort_values('importance',ascending = False)
    importance.to_csv(
        MODELS/'feature_importance.csv',index = False
    )
    print("\nFeature Importance:")
    print(importance)
    print("Evaluating...")
    metrics = evaluate(model, X_test, y_test_raw,train_full,test)
    
    print("Saving...")
    save_model(model, metrics)
    
    print("Done.")


if __name__ == "__main__":
    run()