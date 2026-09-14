"""
Raw transaction cleaning.

Mirrors, step for step, the cleaning decisions made in CSRA.ipynb's EDA
section, so that a model trained on data produced by this module sees
the same data distribution whether it's being trained or asked to score
new transactions.

Steps applied (in order):
1. Parse InvoiceDate to datetime.
2. Fill missing Description with "Unknown"; drop rows with no Customer ID
   (transactions cannot be attributed to a customer without one).
3. Drop exact duplicate rows.
4. Tag each row as Completed / Cancelled based on the Invoice prefix "C".
5. Drop "anomalous" stock codes -- codes with 4 or fewer digit characters,
   which in the source data are mostly non-product administrative codes
   (postage, bank charges, manual entries, etc.).
6. Drop pure service-line descriptions (carriage / image fees) and
   normalise Description casing.
7. Drop non-positive prices (returns/adjustments recorded as 0 or refund
   rows already captured by the Cancelled flag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _flag_anomalous_stock_codes(stock_codes: pd.Series) -> list:
    """A stock code is treated as anomalous if it has 4 or fewer digits."""
    unique_codes = stock_codes.unique()
    return [
        code
        for code in unique_codes
        if sum(c.isdigit() for c in str(code)) <= 4
    ]


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw Online Retail-style transactions DataFrame.

    Parameters
    ----------
    df : raw transactions, must contain config.RAW_REQUIRED_COLUMNS

    Returns
    -------
    A cleaned copy of the DataFrame, one row per line item, ready to be
    fed into `feature_engineering.compute_customer_features`.
    """
    missing_cols = set(config.RAW_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    data = df.copy()

    # 1. Parse dates
    data[config.COL_INVOICE_DATE] = pd.to_datetime(data[config.COL_INVOICE_DATE])

    # 2. Handle missing values
    data[config.COL_DESCRIPTION] = data[config.COL_DESCRIPTION].fillna("Unknown")
    data = data.dropna(subset=[config.COL_CUSTOMER_ID])

    # 3. Drop exact duplicates
    data = data.drop_duplicates()

    # 4. Cancellation flag
    data["Transaction_Status"] = np.where(
        data[config.COL_INVOICE].astype(str).str.startswith("C"),
        "Cancelled",
        "Completed",
    )

    # 5. Drop anomalous stock codes
    anomalous_codes = _flag_anomalous_stock_codes(data[config.COL_STOCKCODE])
    data = data[~data[config.COL_STOCKCODE].isin(anomalous_codes)]

    # 6. Drop service-line descriptions & normalise casing
    data = data[~data[config.COL_DESCRIPTION].isin(config.SERVICE_DESCRIPTIONS)]
    data[config.COL_DESCRIPTION] = data[config.COL_DESCRIPTION].str.upper()

    # 7. Keep only strictly positive prices
    data = data[data[config.COL_PRICE] > 0]

    data = data.reset_index(drop=True)
    return data
