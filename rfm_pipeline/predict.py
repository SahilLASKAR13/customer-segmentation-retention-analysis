"""
Prediction interface.

Loads the artifacts saved by `train.py` (model, scaler, label encoder,
feature column order) and exposes a single class, `RFMSegmentPredictor`,
with three entry points depending on what shape of input you have:

- `predict_from_raw_transactions`  : raw line-item transactions (Invoice,
  StockCode, Quantity, InvoiceDate, Price, Customer ID, Country, ...) for
  one or more customers -> cleans + featurizes + predicts.
- `predict_from_customer_features` : a DataFrame that already has one row
  per customer with (at least) the 16 model features -> predicts directly.
- `predict_one`                    : a single customer as a plain dict of
  feature_name -> value -> a single predicted segment (+ optional
  probabilities).
"""

from __future__ import annotations

from typing import Optional

import joblib
import pandas as pd

from . import config
from .data_cleaning import clean_transactions
from .feature_engineering import compute_customer_features


class RFMSegmentPredictor:
    def __init__(self, artifacts_dir=None):
        artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
        model_path = config.MODEL_PATH if artifacts_dir == config.ARTIFACTS_DIR else artifacts_dir / "random_forest.pkl"
        scaler_path = config.SCALER_PATH if artifacts_dir == config.ARTIFACTS_DIR else artifacts_dir / "scaler.pkl"
        encoder_path = (
            config.LABEL_ENCODER_PATH if artifacts_dir == config.ARTIFACTS_DIR else artifacts_dir / "label_encoder.pkl"
        )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_columns = config.FEATURE_COLUMNS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict_from_feature_frame(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """feature_df must contain every column in self.feature_columns."""
        missing = set(self.feature_columns) - set(feature_df.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        X = feature_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)

        pred_encoded = self.model.predict(X_scaled)
        pred_labels = self.label_encoder.inverse_transform(pred_encoded)

        proba = self.model.predict_proba(X_scaled)
        confidence = proba.max(axis=1)

        result = feature_df.copy()
        result["Predicted_RFM_Segment"] = pred_labels
        result["Prediction_Confidence"] = confidence
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_from_raw_transactions(
        self,
        raw_transactions: pd.DataFrame,
        snapshot_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Full pipeline for new/incoming raw transactions: clean -> engineer
        features -> predict.

        Parameters
        ----------
        raw_transactions : DataFrame with the raw Online Retail schema
            (Invoice, StockCode, Description, Quantity, InvoiceDate,
            Price, Customer ID, Country).
        snapshot_date : the "as of" date for Recency/Tenure. Defaults to
            the max InvoiceDate found in `raw_transactions` if not given
            -- pass an explicit date (e.g. `pd.Timestamp.now()`) if the
            batch you're scoring doesn't represent "today".

        Returns
        -------
        One row per Customer ID with the engineered features plus
        `Predicted_RFM_Segment` and `Prediction_Confidence`.
        """
        clean_df = clean_transactions(raw_transactions)
        feature_df = compute_customer_features(clean_df, snapshot_date=snapshot_date)
        return self._predict_from_feature_frame(feature_df)

    def predict_from_customer_features(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict directly from an already-aggregated, one-row-per-customer
        feature table (e.g. loaded from a data warehouse feature store).
        Must contain every column in config.FEATURE_COLUMNS.
        """
        return self._predict_from_feature_frame(customer_features)

    def predict_one(self, features: dict) -> dict:
        """
        Predict a single customer's RFM segment from a plain dict of
        feature_name -> value.

        Example
        -------
        >>> predictor.predict_one({
        ...     "Recency": 12, "Frequency": 8, "Monetary": 2500.0,
        ...     "Total_Products_Purchased": 340, "Unique_Products_Purchased": 45,
        ...     "Avg_Transaction_Value": 312.5, "Customer_Tenure_Days": 400,
        ...     "Revenue_Per_Product": 55.6, "Avg_Days_Between_Purchases": 30,
        ...     "Std_Days_Between_Purchases": 5, "Purchase_Regularity": 5.0,
        ...     "Spending_Trend": 15.2, "Monthly_Spending_Mean": 300,
        ...     "Monthly_Spending_Std": 40, "Cancellation_Rate": 0.02, "Is_UK": 1,
        ... })
        {'Predicted_RFM_Segment': 'Champions', 'Prediction_Confidence': 0.87}
        """
        row = pd.DataFrame([features])
        result = self._predict_from_feature_frame(row)
        return {
            "Predicted_RFM_Segment": result["Predicted_RFM_Segment"].iloc[0],
            "Prediction_Confidence": float(result["Prediction_Confidence"].iloc[0]),
        }
