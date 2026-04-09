
import joblib

model = joblib.load("../models/house_model.pkl")

def predict_price(sqft,bedrooms):

    return model.predict([[sqft,bedrooms]])[0]
