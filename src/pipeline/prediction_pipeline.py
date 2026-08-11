from pathlib import Path
import joblib
import pandas as pd

from src.components.feature_engineering import CustomerFeatureEngineer


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODEL_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Products_Purchased",
    "Unique_Products_Purchased",
    "Avg_Transaction_Value",
    "Customer_Tenure_Days",
    "Revenue_Per_Product",
    "Avg_Days_Between_Purchases",
    "Std_Days_Between_Purchases",
    "Purchase_Regularity",
    "Spending_Trend",
    "Monthly_Spending_Mean",
    "Monthly_Spending_Std",
    "Cancellation_Rate",
    "Is_UK"
]


class PredictionPipeline:

    def __init__(self):

        # ---------------------------------------------
        # Artifact paths
        # ---------------------------------------------

        model_path = (
            PROJECT_ROOT
            / "artifacts"
            / "random_forest.pkl"
        )

        scaler_path = (
            PROJECT_ROOT
            / "artifacts"
            / "scaler.pkl"
        )

        encoder_path = (
            PROJECT_ROOT
            / "artifacts"
            / "label_encoder.pkl"
        )

        # ---------------------------------------------
        # Historical data
        # ---------------------------------------------

        data_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "online_retail_II.csv"
        )

        # ---------------------------------------------
        # Check files
        # ---------------------------------------------

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found:\n{scaler_path}"
            )

        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found:\n{encoder_path}"
            )

        if not data_path.exists():
            raise FileNotFoundError(
                f"Historical data not found:\n{data_path}"
            )

        # ---------------------------------------------
        # Load artifacts
        # ---------------------------------------------

        self.model = joblib.load(model_path)

        self.scaler = joblib.load(scaler_path)

        self.label_encoder = joblib.load(
            encoder_path
        )

        # ---------------------------------------------
        # Load historical transactions
        # ---------------------------------------------

        self.historical_data = pd.read_csv(
            data_path
        )

    def predict(self, transaction):

        # Convert transaction to DataFrame

        new_transaction = pd.DataFrame(
            [transaction]
        )

        # Convert date

        new_transaction["InvoiceDate"] = pd.to_datetime(
            new_transaction["InvoiceDate"],
            errors="coerce"
        )

        # Validate CustomerID

        if new_transaction["CustomerID"].isna().any():

            raise ValueError(
                "CustomerID cannot be empty."
            )

        customer_id = new_transaction[
            "CustomerID"
        ].iloc[0]

        # ---------------------------------------------
        # Combine historical + new transaction
        # ---------------------------------------------

        combined_data = pd.concat(
            [
                self.historical_data,
                new_transaction
            ],
            ignore_index=True
        )

        # ---------------------------------------------
        # Feature Engineering
        # ---------------------------------------------

        feature_engineer = CustomerFeatureEngineer(
            snapshot_date=new_transaction[
                "InvoiceDate"
            ].iloc[0]
        )

        customer_features = (
            feature_engineer.transform(
                combined_data
            )
        )

        # ---------------------------------------------
        # Get customer
        # ---------------------------------------------

        customer_data = customer_features[
            customer_features["CustomerID"]
            == customer_id
        ]

        if customer_data.empty:

            raise ValueError(
                f"Customer {customer_id} "
                "was not found."
            )

        # ---------------------------------------------
        # Select model features
        # ---------------------------------------------

        X = customer_data[
            MODEL_FEATURES
        ]

        # ---------------------------------------------
        # Scale
        # ---------------------------------------------

        X_scaled = self.scaler.transform(X)

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        prediction = self.model.predict(
            X_scaled
        )

        # ---------------------------------------------
        # Decode segment
        # ---------------------------------------------

        segment = self.label_encoder.inverse_transform(
            prediction
        )[0]

        return segment