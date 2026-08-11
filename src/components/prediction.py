import joblib
import pandas as pd


class RFMPredictor:

    def __init__(
        self,
        model_path="artifacts/random_forest.pkl",
        scaler_path="artifacts/scaler.pkl",
        encoder_path="artifacts/label_encoder.pkl"
    ):

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

    def predict(self, customer_features):

        feature_array = customer_features.values.reshape(1, -1)

        scaled_features = self.scaler.transform(feature_array)

        prediction = self.model.predict(scaled_features)

        segment = self.encoder.inverse_transform(prediction)

        return segment[0]