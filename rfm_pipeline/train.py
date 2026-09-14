"""
End-to-end training: raw transactions -> cleaned data -> customer features
-> rule-based RFM labels -> tuned RandomForestClassifier -> saved artifacts.

Usage
-----
    python -m rfm_pipeline.train --input data/online_retail_II.csv
    python -m rfm_pipeline.train --input data/online_retail_II.csv --quick
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config
from .data_cleaning import clean_transactions
from .feature_engineering import add_rfm_labels, compute_customer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_training_table(raw_csv_path: str) -> pd.DataFrame:
    """Raw CSV -> cleaned -> customer features -> RFM labels, in one call."""
    logger.info("Loading raw data from %s", raw_csv_path)
    raw_df = pd.read_csv(raw_csv_path)

    logger.info("Cleaning transactions...")
    clean_df = clean_transactions(raw_df)

    logger.info("Engineering customer-level features...")
    customer_features = compute_customer_features(clean_df)

    logger.info("Assigning rule-based RFM segment labels...")
    labeled = add_rfm_labels(customer_features)

    return labeled


def train_model(labeled_df: pd.DataFrame, quick: bool = False) -> dict:
    """
    Fit (optionally hyper-parameter-tuned) RandomForestClassifier to
    predict RFM_Segment from config.FEATURE_COLUMNS, then persist all
    artifacts needed for prediction.

    Parameters
    ----------
    labeled_df : output of `build_training_table`
    quick : if True, skip RandomizedSearchCV and fit a single default
        RandomForest (much faster -- useful for smoke-testing the pipeline).

    Returns
    -------
    dict of evaluation metrics.
    """
    X = labeled_df[config.FEATURE_COLUMNS]
    y_raw = labeled_df[config.TARGET_COLUMN]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if quick:
        logger.info("Quick mode: fitting a single default RandomForestClassifier...")
        best_model = RandomForestClassifier(random_state=config.RANDOM_STATE)
        best_model.fit(X_train_scaled, y_train)
        best_params = best_model.get_params()
    else:
        logger.info(
            "Running RandomizedSearchCV (%d iters, %d-fold CV)...",
            config.RANDOM_SEARCH_N_ITER,
            config.RANDOM_SEARCH_CV,
        )
        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=config.RANDOM_STATE),
            param_distributions=config.RF_PARAM_GRID,
            n_iter=config.RANDOM_SEARCH_N_ITER,
            scoring="f1_weighted",
            cv=config.RANDOM_SEARCH_CV,
            verbose=1,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        logger.info("Best params: %s", best_params)

    y_pred = best_model.predict(X_test_scaled)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    cv_scores = cross_val_score(
        best_model, X_train_scaled, y_train, cv=5, scoring="f1_weighted", n_jobs=-1
    )

    metrics = {
        "test_f1_weighted": test_f1,
        "cv_f1_mean": float(np.mean(cv_scores)),
        "cv_f1_std": float(np.std(cv_scores)),
        "classification_report": report,
        "best_params": {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                         for k, v in (best_params.items() if isinstance(best_params, dict) else {})},
    }

    logger.info("Test weighted F1: %.4f", test_f1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    _save_artifacts(best_model, scaler, label_encoder, metrics)
    return metrics


def _save_artifacts(model, scaler, label_encoder, metrics: dict) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    with open(config.FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(config.FEATURE_COLUMNS, f, indent=2)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Artifacts saved to %s", config.ARTIFACTS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Train the RFM segment classifier.")
    parser.add_argument("--input", required=True, help="Path to raw online_retail_II.csv")
    parser.add_argument(
        "--quick", action="store_true", help="Skip hyper-parameter search (fast smoke test)."
    )
    args = parser.parse_args()

    labeled_df = build_training_table(args.input)
    train_model(labeled_df, quick=args.quick)


if __name__ == "__main__":
    main()
