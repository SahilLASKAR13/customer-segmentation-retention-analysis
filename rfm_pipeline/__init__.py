"""
rfm_pipeline
============
A modular pipeline for cleaning Online Retail transactional data,
engineering RFM (Recency / Frequency / Monetary) based customer features,
training a classifier that predicts a customer's RFM segment, and
serving predictions on new/unseen transactional data.

Modules
-------
config          : central configuration (paths, columns, hyper-params)
data_cleaning   : raw transaction cleaning steps (mirrors CSRA.ipynb EDA/cleaning)
feature_engineering : builds customer-level features + rule-based RFM labels
train           : trains & saves the RandomForest classifier + artifacts
predict         : loads saved artifacts and predicts RFM segment for new data
"""

from .predict import RFMSegmentPredictor  # noqa: F401

__all__ = ["RFMSegmentPredictor"]
