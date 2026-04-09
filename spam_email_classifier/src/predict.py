
import joblib
from src.data_preprocessing import clean_text

model = joblib.load("../models/spam_classifier.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

def predict_email(email):

    email = clean_text(email)

    vec = vectorizer.transform([email])

    return model.predict(vec)[0]
