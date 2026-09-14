"""
Streamlit UI for the RFM Segment Predictor.

Two ways to get a prediction:
1. "Manual Input" tab -- type in a single customer's feature values and
   get an instant predicted segment + confidence.
2. "Upload Transactions" tab -- upload a CSV of raw line-item transactions
   (Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
   Customer ID, Country) and get predicted segments for every customer
   in the file, downloadable as CSV.

Run with:
    streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st

from rfm_pipeline import config
from rfm_pipeline.predict import RFMSegmentPredictor

st.set_page_config(page_title="RFM Segment Predictor", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_predictor():
    return RFMSegmentPredictor()


def artifacts_missing():
    return not (
        config.MODEL_PATH.exists() and config.SCALER_PATH.exists() and config.LABEL_ENCODER_PATH.exists()
    )


st.title("📊 RFM Segment Predictor")
st.caption(
    "Predict a customer's RFM segment (Champions, Loyal Customers, At Risk, "
    "Hibernating, etc.) from either raw transactions or pre-computed features."
)

if artifacts_missing():
    st.error(
        "No trained model found in `artifacts/`. Train one first:\n\n"
        "```\npython main.py train --input data/online_retail_II.csv\n```"
    )
    st.stop()

predictor = load_predictor()

tab_manual, tab_upload = st.tabs(["🧮 Manual Input", "📁 Upload Transactions"])

# ---------------------------------------------------------------------------
# Tab 1: manual single-customer input
# ---------------------------------------------------------------------------
with tab_manual:
    st.subheader("Enter a single customer's features")
    st.caption("Fill in the customer's RFM-related metrics and get an instant segment prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
        frequency = st.number_input("Frequency (number of invoices)", min_value=0, value=5)
        monetary = st.number_input("Monetary (total spend, £)", min_value=0.0, value=500.0, step=10.0)
        total_products = st.number_input("Total Products Purchased", min_value=0, value=50)
        unique_products = st.number_input("Unique Products Purchased", min_value=0, value=20)
        avg_txn_value = st.number_input("Avg Transaction Value (£)", min_value=0.0, value=100.0, step=5.0)

    with col2:
        tenure_days = st.number_input("Customer Tenure (days)", min_value=0, value=200)
        revenue_per_product = st.number_input("Revenue Per Product (£)", min_value=0.0, value=25.0, step=1.0)
        avg_days_between = st.number_input("Avg Days Between Purchases", min_value=0.0, value=30.0, step=1.0)
        std_days_between = st.number_input("Std Dev Days Between Purchases", min_value=0.0, value=10.0, step=1.0)
        purchase_regularity = st.number_input("Purchase Regularity", min_value=0.0, value=3.0, step=0.5)
        spending_trend = st.number_input("Spending Trend (slope, £/month)", value=5.0, step=1.0)

    with col3:
        monthly_mean = st.number_input("Monthly Spending Mean (£)", min_value=0.0, value=150.0, step=5.0)
        monthly_std = st.number_input("Monthly Spending Std (£)", min_value=0.0, value=40.0, step=5.0)
        cancellation_rate = st.slider("Cancellation Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        is_uk = st.selectbox("Is UK customer?", options=["Yes", "No"])

    if st.button("Predict Segment", type="primary"):
        features = {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "Total_Products_Purchased": total_products,
            "Unique_Products_Purchased": unique_products,
            "Avg_Transaction_Value": avg_txn_value,
            "Customer_Tenure_Days": tenure_days,
            "Revenue_Per_Product": revenue_per_product,
            "Avg_Days_Between_Purchases": avg_days_between,
            "Std_Days_Between_Purchases": std_days_between,
            "Purchase_Regularity": purchase_regularity,
            "Spending_Trend": spending_trend,
            "Monthly_Spending_Mean": monthly_mean,
            "Monthly_Spending_Std": monthly_std,
            "Cancellation_Rate": cancellation_rate,
            "Is_UK": 1 if is_uk == "Yes" else 0,
        }
        result = predictor.predict_one(features)

        st.success(f"### Predicted Segment: **{result['Predicted_RFM_Segment']}**")
        st.metric("Prediction Confidence", f"{result['Prediction_Confidence']*100:.1f}%")

# ---------------------------------------------------------------------------
# Tab 2: bulk prediction from raw transactions CSV
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload raw transactions")
    st.caption(
        "CSV must contain columns: Invoice, StockCode, Description, Quantity, "
        "InvoiceDate, Price, Customer ID, Country -- one row per line item."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    snapshot_date_input = st.date_input("Snapshot / 'as of' date (used for Recency & Tenure)", value=None)

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(raw_df):,}** rows.")
        st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("Run Predictions", type="primary"):
            with st.spinner("Cleaning data, engineering features, and predicting..."):
                try:
                    snapshot_ts = pd.Timestamp(snapshot_date_input) if snapshot_date_input else None
                    results = predictor.predict_from_raw_transactions(raw_df, snapshot_date=snapshot_ts)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            display_cols = ["Customer ID", "Predicted_RFM_Segment", "Prediction_Confidence"]
            st.success(f"Predicted segments for {len(results):,} customers.")
            st.dataframe(results[display_cols], use_container_width=True)

            st.bar_chart(results["Predicted_RFM_Segment"].value_counts())

            csv_bytes = results[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=csv_bytes,
                file_name="rfm_segment_predictions.csv",
                mime="text/csv",
            )
