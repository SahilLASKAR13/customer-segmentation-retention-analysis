"""
Command-line entry point.

Train:
    python main.py train --input data/online_retail_II.csv
    python main.py train --input data/online_retail_II.csv --quick

Predict (on a CSV of raw transactions, writes a CSV of results):
    python main.py predict --input data/new_transactions.csv --output predictions.csv
"""

import argparse

import pandas as pd

from rfm_pipeline.predict import RFMSegmentPredictor
from rfm_pipeline.train import build_training_table, train_model


def run_train(args):
    labeled_df = build_training_table(args.input)
    train_model(labeled_df, quick=args.quick)


def run_predict(args):
    predictor = RFMSegmentPredictor()
    raw = pd.read_csv(args.input)
    snapshot_date = pd.Timestamp(args.snapshot_date) if args.snapshot_date else None
    results = predictor.predict_from_raw_transactions(raw, snapshot_date=snapshot_date)
    out_cols = ["Customer ID", "Predicted_RFM_Segment", "Prediction_Confidence"]
    results[out_cols].to_csv(args.output, index=False)
    print(f"Saved {len(results)} predictions to {args.output}")
    print(results[out_cols].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="RFM segment training / prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model from raw transactions.")
    train_parser.add_argument("--input", required=True, help="Path to raw transactions CSV.")
    train_parser.add_argument("--quick", action="store_true", help="Skip hyper-parameter search.")
    train_parser.set_defaults(func=run_train)

    predict_parser = subparsers.add_parser("predict", help="Predict RFM segments for new transactions.")
    predict_parser.add_argument("--input", required=True, help="Path to new raw transactions CSV.")
    predict_parser.add_argument("--output", default="predictions.csv", help="Where to write predictions CSV.")
    predict_parser.add_argument(
        "--snapshot-date", default=None, help="Reference 'as of' date, e.g. 2026-09-14. Defaults to max date in input."
    )
    predict_parser.set_defaults(func=run_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
