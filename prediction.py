import joblib

def predict(data):
    model = joblib.load('mobile_price_classifier.sav')
    return model.predict(data)
