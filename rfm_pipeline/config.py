"""
Central configuration for the RFM pipeline.
Keeping every path / column name / hyper-parameter here means the rest
of the codebase never hardcodes a "magic" string, and future-you only
edits one file when something changes (a new raw column name, a new
artifacts folder, etc.).
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = ARTIFACTS_DIR / "random_forest.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# --------------------------------------------------------------------------
# Raw column names (Online Retail II schema, as produced by CSRA.ipynb)
# --------------------------------------------------------------------------
COL_INVOICE = "Invoice"
COL_STOCKCODE = "StockCode"
COL_DESCRIPTION = "Description"
COL_QUANTITY = "Quantity"
COL_INVOICE_DATE = "InvoiceDate"
COL_PRICE = "Price"
COL_CUSTOMER_ID = "Customer ID"
COL_COUNTRY = "Country"

RAW_REQUIRED_COLUMNS = [
    COL_INVOICE,
    COL_STOCKCODE,
    COL_DESCRIPTION,
    COL_QUANTITY,
    COL_INVOICE_DATE,
    COL_PRICE,
    COL_CUSTOMER_ID,
    COL_COUNTRY,
]

# Descriptions that are service/shipping line items rather than products
# (identified in CSRA.ipynb's EDA of lowercase descriptions).
SERVICE_DESCRIPTIONS = ["Next Day Carriage", "High Resolution Image"]

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
# A customer is considered "Churned" (rule-based label used only for the
# training-time RFM_Segment target, not fed to the model as an input).
CHURN_DAYS = 90

# The exact 16 engineered features the model was trained on
# (must match model.ipynb cell that builds `X`). Order matters for
# the StandardScaler / RandomForest that were fit on this order.
FEATURE_COLUMNS = [
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
    "Is_UK",
]

TARGET_COLUMN = "RFM_Segment"
CUSTOMER_ID_OUTPUT_COLUMN = "Customer ID"

# --------------------------------------------------------------------------
# Model training
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

RANDOM_SEARCH_N_ITER = 20
RANDOM_SEARCH_CV = 5
