import streamlit as st

from src.pipeline.prediction_pipeline import PredictionPipeline


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer RFM Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("📊 Customer RFM Segment Prediction")

st.write(
    "Enter a customer's transaction details to predict "
    "their RFM customer segment."
)


# ---------------------------------------------------------
# Create Prediction Pipeline
# ---------------------------------------------------------

@st.cache_resource
def load_prediction_pipeline():

    return PredictionPipeline()


try:

    pipeline = load_prediction_pipeline()

except Exception as e:

    st.error(
        f"Unable to load prediction pipeline: {e}"
    )

    st.stop()


# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------

st.subheader("🧾 Transaction Details")


with st.form("customer_transaction_form"):

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Column 1
    # -----------------------------------------------------

    with col1:

        invoice = st.text_input(
            "Invoice",
            placeholder="Example: 536365"
        )

        stock_code = st.text_input(
            "Stock Code",
            placeholder="Example: 85123A"
        )

        description = st.text_input(
            "Description",
            placeholder="Example: WHITE HANGING HEART T-LIGHT HOLDER"
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=1,
            step=1
        )

    # -----------------------------------------------------
    # Column 2
    # -----------------------------------------------------

    with col2:

        invoice_date = st.date_input(
            "Invoice Date"
        )

        price = st.number_input(
            "Price",
            min_value=0.0,
            value=1.0,
            step=0.01
        )

        customer_id = st.number_input(
            "Customer ID",
            min_value=1,
            step=1
        )

        country = st.text_input(
            "Country",
            value="United Kingdom"
        )

    # -----------------------------------------------------
    # Submit Button
    # -----------------------------------------------------

    submitted = st.form_submit_button(
        "🔮 Predict RFM Segment"
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    if not invoice.strip():

        st.warning("Please enter Invoice.")

        st.stop()

    if not stock_code.strip():

        st.warning("Please enter Stock Code.")

        st.stop()

    if not description.strip():

        st.warning("Please enter Description.")

        st.stop()

    if not country.strip():

        st.warning("Please enter Country.")

        st.stop()


    # -----------------------------------------------------
    # Create Transaction Dictionary
    # -----------------------------------------------------

    transaction = {

        "Invoice": invoice,

        "StockCode": stock_code,

        "Description": description,

        "Quantity": quantity,

        "InvoiceDate": str(invoice_date),

        "Price": price,

        "CustomerID": customer_id,

        "Country": country
    }


    # -----------------------------------------------------
    # Make Prediction
    # -----------------------------------------------------

    try:

        prediction = pipeline.predict(
            transaction
        )


        # -------------------------------------------------
        # Display Result
        # -------------------------------------------------

        st.success(
            f"Predicted RFM Segment: **{prediction}**"
        )


        # -------------------------------------------------
        # Business Recommendation
        # -------------------------------------------------

        recommendations = {

            "Champions":
                "High-value and highly engaged customer. "
                "Focus on loyalty programs, premium offers, "
                "and customer retention.",

            "Loyal Customers":
                "Customer shows strong engagement. "
                "Use personalized offers and loyalty rewards.",

            "Best Customers":
                "Valuable customer with good purchasing "
                "behavior. Focus on retention and upselling.",

            "Good Customers":
                "Customer has healthy purchasing activity. "
                "Encourage repeat purchases with personalized promotions.",

            "At Risk":
                "Customer engagement is declining. "
                "Consider targeted re-engagement campaigns.",

            "Lost Customers":
                "Customer has very low recent engagement. "
                "Consider win-back campaigns and special offers."
        }


        recommendation = recommendations.get(
            prediction,
            "Analyze the customer's purchasing behavior "
            "and create a suitable retention strategy."
        )


        st.subheader("💡 Business Recommendation")

        st.info(recommendation)


    except ValueError as e:

        st.error(
            f"Prediction error: {e}"
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )
