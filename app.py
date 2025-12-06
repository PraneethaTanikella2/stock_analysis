# app.py
"""
Infosys_SmartStock - Streamlit dashboard (polished UI)
Run:
    streamlit run app.py
Requirements: pandas, numpy, plotly, streamlit, joblib
Files expected in project root:
 - data/cleaned_data.csv
 - models/{linear_regression.joblib or random_forest.joblib}
 - models/mappings.json
 - models/feature_columns.json
"""

import os
import json
from datetime import timedelta

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

# --------------------------- Config ---------------------------
DATA_CLEANED = "data/cleaned_data.csv"
MODEL_DIR = "models"

st.set_page_config(page_title="Infosys_SmartStock", layout="wide", page_icon="📦")

# --------------------------- Utility loaders ---------------------------
@st.cache_data
def load_data(path=DATA_CLEANED):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

@st.cache_data
def load_model_and_mappings(model_dir=MODEL_DIR):
    model_file = None
    for fname in os.listdir(model_dir):
        if fname.endswith(".joblib"):
            model_file = os.path.join(model_dir, fname)
            break
    if model_file is None:
        return None, None, None
    model = joblib.load(model_file)
    with open(os.path.join(model_dir, "mappings.json"), "r") as f:
        mappings = json.load(f)
    with open(os.path.join(model_dir, "feature_columns.json"), "r") as f:
        feature_cols = json.load(f)
    return model, mappings, feature_cols

# --------------------------- Load everything ---------------------------
df = load_data()
model, mappings, feature_cols = load_model_and_mappings()

# --------------------------- Minimal checks ---------------------------
if df is None or df.empty:
    st.error("Cleaned data not found. Run eda_analysis.py first to create data/cleaned_data.csv.")
    st.stop()

if model is None:
    st.error("Trained model not found in /models. Run model_training.py first.")
    st.stop()

# --------------------------- Small CSS (optional) ---------------------------
CUSTOM_CSS = """
<style>
body .main { font-size: 15px; }
[data-testid="stSidebar"] { background-color: #f8fafc; padding: 10px 12px; }
.stButton>button { border-radius: 8px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------- Header ---------------------------
logo_path = "assets/logo.png"  # optional - create assets/logo.png if you want a logo
col1, col2 = st.columns([1, 6])
with col1:
    try:
        st.image(logo_path, width=80)
    except Exception:
        st.markdown("<h2 style='margin:0'>📦</h2>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin:0'>Infosys_SmartStock</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray; margin:0'>Inventory optimization & demand forecasting for retail devices</p>", unsafe_allow_html=True)
st.write("---")

# --------------------------- Sidebar controls ---------------------------
st.sidebar.header("🔎 Filters & Forecast Settings")
st.sidebar.markdown("Choose a store and product to view forecasts and recommended stock levels.")

stores = sorted(df["store_id"].unique().tolist())
products = sorted(df["product_id"].unique().tolist())

selected_store = st.sidebar.selectbox("Store", stores)
selected_product = st.sidebar.selectbox("Product", products)
forecast_days = st.sidebar.slider("Forecast days (ahead)", 1, 30, 7)
safety_pct = st.sidebar.slider("Safety stock (%)", 0, 100, 30)
st.sidebar.markdown("---")
st.sidebar.markdown("📌 Tip: choose a product with adequate historical data (30+ days) for better forecasts.")

# --------------------------- Filter historical data ---------------------------
history = df[(df["store_id"] == selected_store) & (df["product_id"] == selected_product)].copy()
history = history.sort_values("date")

st.markdown(f"### Historical & Forecast — Store `{selected_store}` • Product `{selected_product}`")

if history.empty:
    st.warning("No historical data for this store/product combination. Try a different selection.")
    st.stop()

# --------------------------- Prepare mapping codes ---------------------------
# mappings file contains mappings like {"store_map": {"store_A": 0, ...}, "product_map": {...}}
store_map = mappings.get("store_map", {}) if mappings else {}
product_map = mappings.get("product_map", {}) if mappings else {}

def get_code(value, mapping):
    """
    Return integer code for a categorical value using saved mapping.
    If value not found, return 0 (fallback).
    """
    try:
        return int(mapping.get(value, 0))
    except Exception:
        return 0

store_code = get_code(selected_store, store_map)
product_code = get_code(selected_product, product_map)

# --------------------------- Build future DataFrame for predictions ---------------------------
last_date = history["date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

future_df = pd.DataFrame({"date": future_dates})
# standard date parts used in training
future_df["year"] = future_df["date"].dt.year
future_df["month"] = future_df["date"].dt.month
future_df["day"] = future_df["date"].dt.day
future_df["dayofweek"] = future_df["date"].dt.dayofweek

# add codes (model expects store_code & product_code)
future_df["store_code"] = store_code
future_df["product_code"] = product_code

# Make sure feature_cols exist (defensive)
missing = [c for c in feature_cols if c not in future_df.columns]
if missing:
    # create missing columns as zeros (shouldn't usually happen)
    for c in missing:
        future_df[c] = 0

# Ensure correct column order
X_future = future_df[feature_cols]

# Predict using loaded model
try:
    preds = model.predict(X_future)
except Exception as e:
    st.error(f"Model prediction failed: {e}")
    st.stop()

future_df["predicted_sales"] = preds
total_pred = float(future_df["predicted_sales"].sum())
recommended_stock = int(total_pred * (1 + safety_pct / 100.0))

# --------------------------- KPIs ---------------------------
last_30_cutoff = history["date"].max() - pd.Timedelta(days=30)
last_30 = history[history["date"] >= last_30_cutoff]
avg_30 = float(last_30["sales"].mean()) if not last_30.empty else 0.0
# percentage change (pred avg per day vs last_30 avg)
pred_avg_per_day = total_pred / max(forecast_days, 1)
change_pct = (pred_avg_per_day - avg_30) / (avg_30 + 1e-9) * 100

k1, k2, k3 = st.columns([1.5, 1.5, 1.5])
k1.metric("Avg daily (last 30d)", f"{avg_30:.1f}")
k2.metric(f"Predicted total (next {forecast_days} days)", f"{total_pred:.1f}")
k3.metric("Recommended stock", f"{recommended_stock}", delta=f"{change_pct:.1f}%")

st.write("---")

# --------------------------- Tabs: History & Forecast ---------------------------
tab1, tab2 = st.tabs(["📈 History", "🔮 Forecast"])

with tab1:
    st.subheader("Historical Sales (aggregated by date)")
    hist_plot = history.groupby("date")["sales"].sum().reset_index()
    fig_hist = px.line(hist_plot, x="date", y="sales", title="Historical Daily Sales", template="plotly_white")
    fig_hist.update_traces(mode="lines+markers", marker=dict(size=4), hovertemplate="%{x}<br>Sales: %{y:.0f}")
    fig_hist.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("**Recent historical records**")
    st.dataframe(history.sort_values("date", ascending=False).head(20))

with tab2:
    st.subheader("Forecast (next days)")
    display_df = future_df[["date", "predicted_sales"]].copy()
    display_df["predicted_sales"] = display_df["predicted_sales"].round(2)
    st.dataframe(display_df)

    fig_fore = px.line(future_df, x="date", y="predicted_sales", title="Forecasted Daily Sales", template="plotly_white")
    fig_fore.update_traces(mode="lines+markers", marker=dict(size=6), hovertemplate="%{x}<br>Predicted: %{y:.2f}")
    fig_fore.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_fore, use_container_width=True)

    csv = display_df.to_csv(index=False)
    st.download_button("📥 Download forecast CSV", csv, file_name="forecast.csv", mime="text/csv")

# --------------------------- Additional info & notes ---------------------------
st.write("---")
st.markdown(
    """
    **Notes & limitations**
    - This is a baseline model for educational purposes.
    - For production: use time-series cross-validation, include lead-time, promotions, price/holiday effects, and refine safety stock calculation.
    - If a product/store is new (no mapping), model uses fallback code 0 — predictions in that case may be inaccurate.
    """
)
