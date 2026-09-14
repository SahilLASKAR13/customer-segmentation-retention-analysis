"""
Feature engineering.

Split into two clearly separated responsibilities:

1. `compute_customer_features` -- turns cleaned line-item transactions into
   one row per customer with the exact 16 numeric features the model
   consumes (config.FEATURE_COLUMNS). This is used at BOTH training time
   and prediction time, so a new/unseen customer is featurized identically
   to how the training data was built.

2. `add_rfm_labels` -- reproduces the *rule-based* Recency/Frequency/
   Monetary quintile scoring and segment-naming logic from CSRA.ipynb.
   This only makes sense across a whole population (it uses `qcut`, i.e.
   quantiles), so it is only used to build the training target
   (`RFM_Segment`). It is deliberately NOT part of the prediction path --
   the whole point of training a classifier is to let it approximate this
   rule for new customers without needing to recompute population
   quantiles every time a single new customer needs a segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from . import config


def _calc_slope(monthly_spend: pd.Series) -> float:
    """Linear trend of a customer's month-over-month spend."""
    if len(monthly_spend) <= 1:
        return 0.0
    return float(linregress(np.arange(len(monthly_spend)), monthly_spend.values).slope)


def compute_customer_features(
    df_clean: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Aggregate cleaned line-item transactions into one row per customer,
    with the 16 numeric features the model was trained on.

    Parameters
    ----------
    df_clean : output of `data_cleaning.clean_transactions`
    snapshot_date : the "as of" date used to compute Recency / Tenure.
        - During training this should be left as None, which defaults to
          the max InvoiceDate seen in the training data (matches
          CSRA.ipynb).
        - During prediction on new/incoming data you should usually pass
          an explicit `snapshot_date` (e.g. `pd.Timestamp.now()`), since a
          small prediction batch may not contain a representative "most
          recent" date.

    Returns
    -------
    DataFrame indexed by row (one row per "Customer ID"), containing the
    Customer ID column plus every column in config.FEATURE_COLUMNS.
    """
    df = df_clean.copy()
    df[config.COL_INVOICE_DATE] = pd.to_datetime(df[config.COL_INVOICE_DATE])
    df["Total_Spend"] = df[config.COL_QUANTITY] * df[config.COL_PRICE]

    if snapshot_date is None:
        snapshot_date = df[config.COL_INVOICE_DATE].max()
    else:
        snapshot_date = pd.Timestamp(snapshot_date)

    cust_col = config.COL_CUSTOMER_ID

    # --- core RFM + purchase-volume aggregates ---------------------------
    rfm = (
        df.groupby(cust_col)
        .agg(
            Recency=(config.COL_INVOICE_DATE, lambda x: (snapshot_date - x.max()).days),
            Frequency=(config.COL_INVOICE, "nunique"),
            Monetary=("Total_Spend", "sum"),
            Total_Products_Purchased=(config.COL_QUANTITY, "sum"),
            Unique_Products_Purchased=(config.COL_STOCKCODE, "nunique"),
            First_Purchase=(config.COL_INVOICE_DATE, "min"),
            Last_Purchase=(config.COL_INVOICE_DATE, "max"),
        )
        .reset_index()
    )

    rfm["Avg_Transaction_Value"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["Customer_Tenure_Days"] = (snapshot_date - rfm["First_Purchase"]).dt.days
    rfm["Revenue_Per_Product"] = (rfm["Monetary"] / rfm["Unique_Products_Purchased"]).fillna(0)

    # --- inter-purchase gap stats -----------------------------------------
    purchase_gaps = (
        df.sort_values([cust_col, config.COL_INVOICE_DATE])
        .groupby(cust_col)[config.COL_INVOICE_DATE]
        .diff()
        .dt.days
    )
    gap_stats = (
        purchase_gaps.groupby(df[cust_col])
        .agg(Avg_Days_Between_Purchases="mean", Std_Days_Between_Purchases="std")
        .reset_index()
    )
    gap_stats["Std_Days_Between_Purchases"] = gap_stats["Std_Days_Between_Purchases"].fillna(0)
    gap_stats["Purchase_Regularity"] = (
        gap_stats["Avg_Days_Between_Purchases"] / (gap_stats["Std_Days_Between_Purchases"] + 1)
    ).fillna(0)

    # --- monthly spend trend & volatility ----------------------------------
    monthly = (
        df.assign(YearMonth=df[config.COL_INVOICE_DATE].dt.to_period("M"))
        .groupby([cust_col, "YearMonth"])["Total_Spend"]
        .sum()
        .reset_index()
    )
    trend = (
        monthly.groupby(cust_col)["Total_Spend"]
        .apply(_calc_slope)
        .reset_index(name="Spending_Trend")
    )
    monthly_stats = (
        monthly.groupby(cust_col)["Total_Spend"]
        .agg(Monthly_Spending_Mean="mean", Monthly_Spending_Std="std")
        .reset_index()
    )
    monthly_stats["Monthly_Spending_Std"] = monthly_stats["Monthly_Spending_Std"].fillna(0)

    # --- cancellation rate ---------------------------------------------------
    total_tx = df.groupby(cust_col)[config.COL_INVOICE].nunique().reset_index(name="Total_Transactions")
    cancelled = (
        df[df["Transaction_Status"] == "Cancelled"]
        .groupby(cust_col)[config.COL_INVOICE]
        .nunique()
        .reset_index(name="Cancelled_Transactions")
    )
    cancel_features = total_tx.merge(cancelled, on=cust_col, how="left")
    cancel_features["Cancelled_Transactions"] = cancel_features["Cancelled_Transactions"].fillna(0)
    cancel_features["Cancellation_Rate"] = (
        cancel_features["Cancelled_Transactions"] / cancel_features["Total_Transactions"]
    )

    # --- country (Is_UK) ------------------------------------------------------
    country_mode = (
        df.groupby([cust_col, config.COL_COUNTRY])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .drop_duplicates(cust_col)
    )
    country_mode["Is_UK"] = (country_mode[config.COL_COUNTRY] == "United Kingdom").astype(int)

    # --- assemble ---------------------------------------------------------
    customer_data = rfm.merge(
        gap_stats[[cust_col, "Avg_Days_Between_Purchases", "Std_Days_Between_Purchases", "Purchase_Regularity"]],
        on=cust_col,
        how="left",
    )
    customer_data = customer_data.merge(trend, on=cust_col, how="left")
    customer_data = customer_data.merge(monthly_stats, on=cust_col, how="left")
    customer_data = customer_data.merge(cancel_features[[cust_col, "Cancellation_Rate"]], on=cust_col, how="left")
    customer_data = customer_data.merge(country_mode[[cust_col, "Is_UK"]], on=cust_col, how="left")

    numeric_cols = customer_data.select_dtypes(include=["number"]).columns
    customer_data[numeric_cols] = customer_data[numeric_cols].fillna(0)

    return customer_data


def _get_rfm_segment(r: int, f: int, m: int) -> str:
    """Rule-based segment naming from R/F/M quintile scores (1-5 each)."""
    if r >= 4 and f >= 4:
        return "Champions"
    elif r >= 4 and f >= 3:
        return "Loyal Customers"
    elif r >= 3 and f >= 4:
        return "Best Customers"
    elif r >= 3 and f >= 3:
        return "Good Customers"
    elif r >= 4 and m >= 4:
        return "At Risk - Big Spenders"
    elif r <= 2 and f >= 4:
        return "Cant Lose Them"
    elif r <= 2 and m >= 4:
        return "Lost - Big Spenders"
    elif r <= 2:
        return "Lost Customers"
    elif f <= 2 and r <= 3:
        return "Hibernating"
    elif m <= 2:
        return "Low-Value Customers"
    else:
        return "Potential"


def add_rfm_labels(customer_data: pd.DataFrame, churn_days: int = config.CHURN_DAYS) -> pd.DataFrame:
    """
    Add population-relative R/F/M quintile scores, the rule-based
    RFM_Segment label, and the Churn flag.

    This is a TRAINING-TIME ONLY step: qcut needs a representative
    population to build meaningful quantile boundaries, so this should
    never be called on a single new customer or a small prediction batch.
    It is what generates the `y` target that the classifier learns to
    predict directly from the raw features at prediction time.
    """
    data = customer_data.copy()

    data["Churn"] = (data["Recency"] > churn_days).astype(int)

    data["R_Score"] = pd.qcut(data["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    data["F_Score"] = pd.qcut(
        data["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)
    data["M_Score"] = pd.qcut(data["Monetary"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)

    data[config.TARGET_COLUMN] = data.apply(
        lambda x: _get_rfm_segment(x["R_Score"], x["F_Score"], x["M_Score"]), axis=1
    )

    return data
