import numpy as np

import pandas as pd

import streamlit as st

import joblib

import math

from datetime import datetime, timedelta

import os

model = joblib.load("streamlit/delivery_model.pkl")

df = pd.read_csv("streamlit/products.csv")

st.title("📦 When My Order Come?")

product = st.selectbox(
    "Choose a product category",
    sorted(df["product_category_name_english"].dropna().unique())
)

state = st.selectbox(
    "Choose your state",
    sorted(df["customer_state"].dropna().unique())
)

if st.button("Predict Delivery"):

    row = df[df["product_category_name_english"] == product].iloc[[0]].copy()

    row["customer_state"] = state

    features = [
        "customer_state", "seller_state",
        "product_category_name_english", "freight_value",
        "product_weight_g", "vol", "dist_km", "quantity",
        "same_state", "customer_lat", "customer_lng",
        "seller_lat", "seller_lng", "month", "dow",
        "day", "hour", "customer_zip_code_prefix",
        "seller_zip_code_prefix"
    ]

    prediction = model.predict(row[features])[0]

    st.success(f"Estimated delivery time: {prediction:.1f} days")


# def sine_km(cust_lat, cust_lng, sell_lat, sell_lng):

#     cust_lat, cust_lng, sell_lat, sell_lng = map(np.radians, [cust_lat, cust_lng, sell_lat, sell_lng])



#     dlat = sell_lat - cust_lat

#     dlng = sell_lng - cust_lng



#     a = np.sin(dlat / 2) ** 2 + np.cos(cust_lat) * np.cos(sell_lat) * np.sin(dlng / 2) ** 2

#     return 2 * 6371 * np.arcsin(np.sqrt(a))





# @st.cache_data

# def load_data():

#     df = pd.read_csv('olist_clean.csv')



#     customers = (df.dropna(subset=['customer_lat', 'customer_lng'])

#                    .drop_duplicates('customer_zip_code_prefix')

#                    .set_index('customer_zip_code_prefix')[['customer_lat', 'customer_lng']])



#     sellers = (df.dropna(subset=['seller_id_new', 'seller_lat', 'seller_lng'])

#                  .drop_duplicates('seller_id_new')

#                  .set_index('seller_id_new')[['seller_lat', 'seller_lng']])



#     return customers, sellers





# def main():

#     st.set_page_config(page_title="Seller Distance", page_icon="📦")

#     st.title("📦 Distance to Seller")



#     customers, sellers = load_data()



#     customer_zip = st.selectbox("Your zip code prefix", sorted(customers.index))

#     seller = st.selectbox("Choose a seller", sorted(sellers.index))



#     if st.button("Calculate distance"):

#         c = customers.loc[customer_zip]

#         s = sellers.loc[seller]

#         dist = sine_km(c['customer_lat'], c['customer_lng'], s['seller_lat'], s['seller_lng'])

#         st.success(f"Distance between you and {seller}: {dist:.0f} km")





# if __name__ == '__main__':

#     main()
