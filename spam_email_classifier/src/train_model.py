
import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from spam_email_classifier.src.data_preprocessing import clean_text

def train():

    df = pd.read_csv("spam_email_classifier/data/raw/sample_spam.csv")

    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer()

    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()

    model.fit(X_vec,y)

    os.makedirs("../models", exist_ok=True)

    BASE_DIR = Path(__file__).resolve().parents[1]
    MODELS_DIR = BASE_DIR / "models"

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODELS_DIR / "spam_classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")

    print("Spam model trained")

if __name__ == "__main__":
    train()
