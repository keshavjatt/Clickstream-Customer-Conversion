import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Conversion Predictor")

# Load model
model = joblib.load("model/model.pkl")

st.title("🛒 Clickstream Customer Conversion Predictor")

price = st.slider("Product Price ($)", 1, 100, 30)
price_2 = st.selectbox("Price Above Category Average?", [1, 2])
page = st.slider("Pages Visited", 1, 5, 2)

input_df = pd.DataFrame({
    "price": [price],
    "price 2": [price_2],
    "page": [page]
})

if st.button("Predict"):
    result = model.predict(input_df)[0]
    st.success("✅ Conversion Likely" if result else "❌ No Conversion")