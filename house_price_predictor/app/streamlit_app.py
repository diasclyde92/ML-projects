
import streamlit as st
import joblib

st.title("House Price Predictor")

sqft = st.number_input("Square Feet")
bedrooms = st.number_input("Bedrooms")

if st.button("Predict"):

    model = joblib.load("../models/house_model.pkl")

    price = model.predict([[sqft,bedrooms]])[0]

    st.success(f"Predicted Price: ${price}")
