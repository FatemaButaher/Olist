import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

model = joblib.load("delivery_model.pkl")
df = pd.read_csv("products.csv")

st.title("📦 Olist Delivery Predictor")

product = st.selectbox(
    "Choose a product category",
    sorted(df["product_category_name_english"].dropna().unique())
)

state = st.selectbox(
    "Choose your state",
    sorted(df["customer_state"].dropna().unique())
)

if st.button("Predict Delivery"):
    row = df[
        (df["product_category_name_english"] == product) &
        (df["customer_state"] == state)
    ]

    if row.empty:
        st.warning("No data available for this selection.")
    else:
        features = [
            "customer_state", "seller_state",
            "product_category_name_english", "freight_value",
            "product_weight_g", "vol", "dist_km", "quantity",
            "same_state", "customer_lat", "customer_lng",
            "seller_lat", "seller_lng", "month", "dow",
            "day", "hour", "customer_zip_code_prefix",
            "seller_zip_code_prefix"
        ]

        prediction = model.predict(row[features].iloc[[0]])[0]
        today = datetime.today()
        delivery_date = today + timedelta(days=float(prediction))
        st.success(f"Expected delivery date: {delivery_date.strftime('%B %d, %Y')}")