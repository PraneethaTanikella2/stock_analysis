# eda_analysis.py
"""
EDA and basic cleaning for INFOSYS_STOCK project.
Run: python eda_analysis.py
Outputs:
 - cleaned_data.csv (in data/)
 - saved plots in outputs/ (png)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# --- Config ---
DATA_PATH = "retail_store_inventory.csv"   # your CSV in project root or data/
CLEANED_PATH = "data/cleaned_data.csv"
OUT_DIR = "outputs"
os.makedirs("data", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print("Initial rows:", len(df))
print(df.columns.tolist())

# --- Basic cleaning ---
# Try to detect common column names
possible_date_cols = [c for c in df.columns if "date" in c.lower()]
date_col = possible_date_cols[0] if possible_date_cols else None
if date_col is None:
    raise SystemExit("No date column found. Please check your CSV for a date column.")

# Convert to datetime
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
print(f"Using '{date_col}' as date column. Null dates:", df[date_col].isna().sum())

# Common names: store, product, sales/quantity
def pick_column(keywords):
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keywords):
            return c
    return None

store_col = pick_column(["store", "location", "branch"])
product_col = pick_column(["product", "item", "sku"])
sales_col = pick_column(["sales", "quantity", "qty", "sold"])

print("Detected columns -> store:", store_col, "product:", product_col, "sales:", sales_col)

if store_col is None or product_col is None or sales_col is None:
    print("Columns could not be automatically detected. Please inspect columns and update code.")
    print("Columns present:", df.columns.tolist())
    # proceed but require user to check

# Drop rows without date or sales
df = df.dropna(subset=[date_col, sales_col])
# Convert sales to numeric
df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
df = df.dropna(subset=[sales_col])
# Remove negative sales if any
df = df[df[sales_col] >= 0].copy()

# Standardize column names
df = df.rename(columns={
    date_col: "date",
    store_col: "store_id" if store_col else store_col,
    product_col: "product_id" if product_col else product_col,
    sales_col: "sales"
})

# If some of the required columns are missing after rename, add placeholders
if "store_id" not in df.columns:
    df["store_id"] = "store_0"
if "product_id" not in df.columns:
    df["product_id"] = "product_0"

# --- Feature: date parts ---
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["weekofyear"] = df["date"].dt.isocalendar().week

# --- Save cleaned data ---
df.to_csv(CLEANED_PATH, index=False)
print("Saved cleaned data to:", CLEANED_PATH)
print("Clean data sample:")
print(df.head())

# --- EDA: basic stats ---
print("\nBasic stats:")
print(df["sales"].describe())
print("\nTop 10 products by total sales:")
print(df.groupby("product_id")["sales"].sum().sort_values(ascending=False).head(10))

# --- Plots (Matplotlib & Plotly) ---
# 1. Daily total sales (matplotlib)
daily = df.groupby("date")["sales"].sum().reset_index()
plt.figure(figsize=(10,4))
plt.plot(daily["date"], daily["sales"])
plt.title("Daily Total Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "daily_total_sales.png"))
plt.close()
print("Saved plot:", os.path.join(OUT_DIR, "daily_total_sales.png"))

# 2. Top 10 products bar chart (plotly interactive saved as HTML)
top_products = df.groupby("product_id")["sales"].sum().reset_index().sort_values("sales", ascending=False).head(10)
fig = px.bar(top_products, x="product_id", y="sales", title="Top 10 Products by Sales")
fig.write_html(os.path.join(OUT_DIR, "top_products.html"))
print("Saved interactive plot:", os.path.join(OUT_DIR, "top_products.html"))

# 3. Sales per store (matplotlib)
store_sales = df.groupby("store_id")["sales"].sum().sort_values(ascending=False)
plt.figure(figsize=(8,4))
store_sales.plot(kind="bar")
plt.title("Total Sales per Store")
plt.xlabel("Store")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sales_per_store.png"))
plt.close()
print("Saved plot:", os.path.join(OUT_DIR, "sales_per_store.png"))

print("\nEDA finished. Check the outputs/ folder and data/cleaned_data.csv")
