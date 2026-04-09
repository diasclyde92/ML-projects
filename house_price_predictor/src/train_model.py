import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from pathlib import Path

def train():

    data = {
        "sqft":[1000,1500,2000,2500],
        "bedrooms":[2,3,3,4],
        "price":[200000,300000,400000,500000]
    }

    df = pd.DataFrame(data)

    X = df[["sqft","bedrooms"]]
    y = df["price"]

    model = LinearRegression()
    model.fit(X,y)

    # Correct path handling
    BASE_DIR = Path(__file__).resolve().parents[1]
    MODELS_DIR = BASE_DIR / "models"

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODELS_DIR / "house_model.pkl")

    print("Model trained and saved")

if __name__ == "__main__":
    train()