import pandas as pd
import numpy as np


class CustomerFeatureEngineer:

    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

    def transform(self, df):

        df = df.copy()

        # ---------------------------------------------------------
        # 1. DATA TYPE CONVERSION
        # ---------------------------------------------------------

        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

        df["Quantity"] = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        df["Price"] = pd.to_numeric(
            df["Price"],
            errors="coerce"
        )

        df["CustomerID"] = pd.to_numeric(
            df["CustomerID"],
            errors="coerce"
        )

        # Remove rows without customer/date
        df = df.dropna(
            subset=["CustomerID", "InvoiceDate"]
        )

        # ---------------------------------------------------------
        # 2. TOTAL TRANSACTION VALUE
        # ---------------------------------------------------------

        df["TotalAmount"] = (
            df["Quantity"] * df["Price"]
        )

        # ---------------------------------------------------------
        # 3. CANCELLATION FLAG
        # ---------------------------------------------------------

        df["Is_Cancelled"] = (
            df["Invoice"].astype(str).str.startswith("C")
        ).astype(int)

        # ---------------------------------------------------------
        # 4. SNAPSHOT DATE
        # ---------------------------------------------------------

        if self.snapshot_date is None:
            snapshot_date = df["InvoiceDate"].max()
        else:
            snapshot_date = pd.to_datetime(
                self.snapshot_date
            )

        # ---------------------------------------------------------
        # 5. BASIC CUSTOMER AGGREGATION
        # ---------------------------------------------------------

        customer = df.groupby("CustomerID").agg(

            First_Purchase=(
                "InvoiceDate",
                "min"
            ),

            Last_Purchase=(
                "InvoiceDate",
                "max"
            ),

            Frequency=(
                "Invoice",
                "nunique"
            ),

            Monetary=(
                "TotalAmount",
                "sum"
            ),

            Total_Products_Purchased=(
                "Quantity",
                "sum"
            ),

            Unique_Products_Purchased=(
                "StockCode",
                "nunique"
            ),

            Avg_Transaction_Value=(
                "TotalAmount",
                "mean"
            ),

            Cancellation_Rate=(
                "Is_Cancelled",
                "mean"
            )
        ).reset_index()

        # ---------------------------------------------------------
        # 6. RECENCY
        # ---------------------------------------------------------

        customer["Recency"] = (
            snapshot_date -
            customer["Last_Purchase"]
        ).dt.days

        # ---------------------------------------------------------
        # 7. CUSTOMER TENURE
        # ---------------------------------------------------------

        customer["Customer_Tenure_Days"] = (
            customer["Last_Purchase"] -
            customer["First_Purchase"]
        ).dt.days

        # Avoid zero tenure
        customer["Customer_Tenure_Days"] = (
            customer["Customer_Tenure_Days"]
            .clip(lower=1)
        )

        # ---------------------------------------------------------
        # 8. REVENUE PER PRODUCT
        # ---------------------------------------------------------

        customer["Revenue_Per_Product"] = (
            customer["Monetary"] /
            customer["Total_Products_Purchased"]
            .replace(0, np.nan)
        )

        customer["Revenue_Per_Product"] = (
            customer["Revenue_Per_Product"]
            .fillna(0)
        )

        # ---------------------------------------------------------
        # 9. PURCHASE INTERVAL FEATURES
        # ---------------------------------------------------------

        df = df.sort_values(
            ["CustomerID", "InvoiceDate"]
        )

        df["Days_Between_Purchases"] = (
            df.groupby("CustomerID")["InvoiceDate"]
            .diff()
            .dt.total_seconds()
            .div(86400)
        )

        purchase_gap = df.groupby("CustomerID")[
            "Days_Between_Purchases"
        ].agg(
            Avg_Days_Between_Purchases="mean",
            Std_Days_Between_Purchases="std"
        ).reset_index()

        customer = customer.merge(
            purchase_gap,
            on="CustomerID",
            how="left"
        )

        customer["Avg_Days_Between_Purchases"] = (
            customer[
                "Avg_Days_Between_Purchases"
            ].fillna(0)
        )

        customer["Std_Days_Between_Purchases"] = (
            customer[
                "Std_Days_Between_Purchases"
            ].fillna(0)
        )

        # ---------------------------------------------------------
        # 10. PURCHASE REGULARITY
        # ---------------------------------------------------------

        customer["Purchase_Regularity"] = (
            1 /
            (
                1 +
                customer["Std_Days_Between_Purchases"]
            )
        )

        # ---------------------------------------------------------
        # 11. MONTHLY SPENDING
        # ---------------------------------------------------------

        df["YearMonth"] = (
            df["InvoiceDate"]
            .dt.to_period("M")
        )

        monthly_spending = (
            df.groupby(
                ["CustomerID", "YearMonth"]
            )["TotalAmount"]
            .sum()
            .reset_index()
        )

        monthly_stats = (
            monthly_spending
            .groupby("CustomerID")["TotalAmount"]
            .agg(
                Monthly_Spending_Mean="mean",
                Monthly_Spending_Std="std"
            )
            .reset_index()
        )

        customer = customer.merge(
            monthly_stats,
            on="CustomerID",
            how="left"
        )

        customer["Monthly_Spending_Mean"] = (
            customer["Monthly_Spending_Mean"]
            .fillna(0)
        )

        customer["Monthly_Spending_Std"] = (
            customer["Monthly_Spending_Std"]
            .fillna(0)
        )

        # ---------------------------------------------------------
        # 12. SPENDING TREND
        # ---------------------------------------------------------

        def calculate_spending_trend(group):

            if len(group) < 2:
                return 0.0

            x = np.arange(len(group))
            y = group["TotalAmount"].values

            slope = np.polyfit(x, y, 1)[0]

            return slope

        spending_trend = (
            monthly_spending
            .sort_values(["CustomerID", "YearMonth"])
            .groupby("CustomerID")
            .apply(
                calculate_spending_trend,
                include_groups=False
            )
            .reset_index(
                name="Spending_Trend"
            )
        )

        customer = customer.merge(
            spending_trend,
            on="CustomerID",
            how="left"
        )

        customer["Spending_Trend"] = (
            customer["Spending_Trend"]
            .fillna(0)
        )

        # ---------------------------------------------------------
        # 13. IS UK
        # ---------------------------------------------------------

        country = (
            df.groupby("CustomerID")["Country"]
            .first()
            .reset_index()
        )

        country["Is_UK"] = (
            country["Country"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("united kingdom")
            .astype(int)
        )

        customer = customer.merge(
            country[
                ["CustomerID", "Is_UK"]
            ],
            on="CustomerID",
            how="left"
        )

        # ---------------------------------------------------------
        # 14. CLEAN NUMERICAL FEATURES
        # ---------------------------------------------------------

        feature_columns = [
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

        customer[feature_columns] = (
            customer[feature_columns]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        return customer