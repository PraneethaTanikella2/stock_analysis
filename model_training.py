# model_training.py
"""
Train ML models to predict daily sales for (store, product, date).
Saves best model and mappings using joblib.
Run: python model_training.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json

CLEANED_PATH = "data/cleaned_data.csv"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

print("Loading cleaned data...")
df = pd.read_csv(CLEANED_PATH, parse_dates=["date"])
print("Rows:", len(df))

# Aggregate if dataset is transactional (multiple records per date/store/product)
agg = df.groupby(["date", "store_id", "product_id"], as_index=False)["sales"].sum()

# feature engineering - date parts
agg["year"] = agg["date"].dt.year
agg["month"] = agg["date"].dt.month
agg["day"] = agg["date"].dt.day
agg["dayofweek"] = agg["date"].dt.dayofweek
agg["weekofyear"] = agg["date"].dt.isocalendar().week.astype(int)

# Create mapping for categorical features
def create_mapping(series):
    cats = series.astype("category")
    mapping = dict(enumerate(cats.cat.categories))
    # inverse mapping useful later (value->code)
    value_to_code = {v: int(k) for k,v in dict(enumerate(cats.cat.categories)).items()}
    codes = series.astype("category").cat.codes
    return codes, value_to_code

agg["store_code"], store_map = create_mapping(agg["store_id"])
agg["product_code"], product_map = create_mapping(agg["product_id"])

# Features and target
feature_cols = ["store_code", "product_code", "year", "month", "day", "dayofweek"]
X = agg[feature_cols]
y = agg["sales"]

# Train-test split (time-agnostic split; for more rigorous use time-based split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def evaluate(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred) ** 0.5
    return mae, rmse

# Linear Regression baseline
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae, lr_rmse = evaluate(y_test, lr_pred)
print(f"LinearRegression -> MAE: {lr_mae:.3f}, RMSE: {lr_rmse:.3f}")

# Random Forest
print("Training RandomForest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae, rf_rmse = evaluate(y_test, rf_pred)
print(f"RandomForest -> MAE: {rf_mae:.3f}, RMSE: {rf_rmse:.3f}")

# Choose best model (by RMSE here)
if rf_rmse <= lr_rmse:
    best_model = rf
    best_name = "random_forest"
    best_metrics = (rf_mae, rf_rmse)
else:
    best_model = lr
    best_name = "linear_regression"
    best_metrics = (lr_mae, lr_rmse)

# Save model and mappings
model_file = os.path.join(MODEL_PATH, f"{best_name}.joblib")
joblib.dump(best_model, model_file)
print("Saved best model to:", model_file)

# Save mappings for store/product (value->code)
with open(os.path.join(MODEL_PATH, "mappings.json"), "w") as f:
    json.dump({"store_map": store_map, "product_map": product_map}, f)
print("Saved mappings to models/mappings.json")

# Also save feature columns for reference
with open(os.path.join(MODEL_PATH, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f)

print("Training finished. Best model:", best_name, "MAE/RMSE:", best_metrics)
