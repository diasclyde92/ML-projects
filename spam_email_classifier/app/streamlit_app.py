
import streamlit as st
from src.predict import predict_email

st.title("Spam Email Classifier")

email = st.text_area("Enter Email")

if st.button("Predict"):

    pred = predict_email(email)

    st.success(pred)
