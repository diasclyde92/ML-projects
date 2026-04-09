
import streamlit as st
import joblib
import os

st.title("Machine Learning Playground")

choice = st.sidebar.selectbox(
    "Select Model",
    ["House Price Predictor", "Spam Email Classifier"]
)

# ---------------- HOUSE PRICE ----------------

if choice == "House Price Predictor":

    st.header("House Price Predictor")

    sqft = st.number_input("Square Feet", min_value=0)
    bedrooms = st.number_input("Bedrooms", min_value=0)

    if st.button("Predict Price"):

        model_path = "../house-price-predictor/models/house_model.pkl"

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            price = model.predict([[sqft, bedrooms]])[0]
        else:
            price = sqft * 200 + bedrooms * 10000

        st.success(f"Predicted Price: ${price}")


# ---------------- SPAM CLASSIFIER ----------------

if choice == "Spam Email Classifier":

    st.header("Spam Email Classifier")

    email = st.text_area("Enter Email Text")

    if st.button("Predict Spam"):

        model_path = "../spam-email-classifier/models/spam_classifier.pkl"
        vectorizer_path = "../spam-email-classifier/models/vectorizer.pkl"

        if os.path.exists(model_path):
            import joblib
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)

            vec = vectorizer.transform([email])
            pred = model.predict(vec)[0]
        else:
            pred = "spam" if "free" in email.lower() or "win" in email.lower() else "ham"

        st.success(f"Prediction: {pred}")
