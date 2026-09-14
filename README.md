# Customer Segmentation & Retention Analysis

## 📌 Project Overview

**Customer Segmentation & Retention Analysis (CSRA)** is an end-to-end
machine learning project that analyzes customer purchasing behavior and
assigns each customer to a meaningful **RFM-based customer segment**.

The project uses transaction-level retail data to understand:

-   How recently a customer purchased
-   How frequently a customer purchases
-   How much a customer spends
-   How many products they purchase
-   How regularly they purchase
-   Whether their spending is increasing or decreasing
-   Their cancellation behavior
-   Their customer tenure and geographic information

A **Random Forest Classifier** is then trained to learn the RFM segment
definitions and predict the segment of new customers.

The project also includes a **Streamlit application** that allows users
to enter customer features manually or upload raw transaction data and
receive segment predictions with a confidence score.

------------------------------------------------------------------------

## 🎯 Business Problem

Businesses often have thousands of customers with very different
purchasing behaviors. Treating every customer in the same way can lead
to:

-   Poorly targeted marketing campaigns
-   Loss of high-value customers
-   Inefficient retention campaigns
-   Missed opportunities for cross-selling and upselling
-   Difficulty identifying customers who are becoming inactive
-   Limited understanding of customer lifetime behavior

### Problem this project solves

This project converts raw transaction data into **customer-level
behavioral features**, segments customers according to their purchasing
behavior, and provides a machine-learning-based prediction system that
can classify new customers into the appropriate segment.

This helps a business answer questions such as:

-   Who are our most valuable customers?
-   Which customers are loyal?
-   Which high-value customers may be at risk?
-   Which customers have become inactive?
-   Which customers have low purchasing value?
-   Which customers should receive retention or re-engagement campaigns?

------------------------------------------------------------------------

## 💡 Solution

The solution follows an end-to-end pipeline:

``` text
Raw Transaction Data
        │
        ▼
Data Cleaning
        │
        ▼
Customer-Level Feature Engineering
        │
        ▼
RFM Scoring & Segment Creation
        │
        ▼
Train/Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Random Forest + Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Save Model Artifacts
        │
        ▼
New Customer Transactions
        │
        ▼
Same Cleaning + Feature Engineering
        │
        ▼
Random Forest Prediction
        │
        ▼
RFM Segment + Prediction Confidence
```

------------------------------------------------------------------------

# 🔄 Project Pipeline

## 1. Raw Data

The project uses an **Online Retail II-style transaction dataset**.

The raw transaction data contains fields such as:

  Column          Description
  --------------- --------------------------------
  `Invoice`       Invoice/transaction identifier
  `StockCode`     Product identifier
  `Description`   Product description
  `Quantity`      Number of units purchased
  `InvoiceDate`   Transaction date
  `Price`         Unit price
  `Customer ID`   Customer identifier
  `Country`       Customer country

The project processes transaction-level data and converts it into a
single row per customer.

------------------------------------------------------------------------

## 2. Data Cleaning

The cleaning pipeline is implemented in:

``` text
rfm_pipeline/data_cleaning.py
```

The following steps are performed:

1.  Convert `InvoiceDate` to datetime.
2.  Fill missing product descriptions with `"Unknown"`.
3.  Remove transactions without a `Customer ID`.
4.  Remove exact duplicate rows.
5.  Identify cancelled transactions using invoice values beginning with
    `C`.
6.  Remove anomalous/non-product stock codes.
7.  Remove service-line items such as carriage/image fees.
8.  Normalize product descriptions to uppercase.
9.  Keep transactions with strictly positive prices.

This ensures that the same cleaning logic is used during both **training
and prediction**.

------------------------------------------------------------------------

# 3. Customer-Level Feature Engineering

The cleaned transaction data is aggregated from:

``` text
Transaction Level
        ↓
Customer Level
```

The feature engineering code is implemented in:

``` text
rfm_pipeline/feature_engineering.py
```

The model uses **16 engineered features**.

### Core RFM Features

#### Recency

Number of days since the customer's most recent purchase.

``` text
Recency = Snapshot Date - Last Purchase Date
```

Lower recency generally indicates a more recently active customer.

#### Frequency

Number of unique invoices made by the customer.

#### Monetary

Total customer spending:

``` text
Monetary = Quantity × Price
```

summed across the customer's transactions.

------------------------------------------------------------------------

## Additional Behavioral Features

The project goes beyond traditional RFM by creating additional customer
behavior features:

  -----------------------------------------------------------------------
  Feature                             Meaning
  ----------------------------------- -----------------------------------
  `Total_Products_Purchased`          Total quantity of products
                                      purchased

  `Unique_Products_Purchased`         Number of unique products purchased

  `Avg_Transaction_Value`             Average spending per invoice

  `Customer_Tenure_Days`              Days between first purchase and
                                      snapshot date

  `Revenue_Per_Product`               Revenue generated per unique
                                      product

  `Avg_Days_Between_Purchases`        Average gap between purchases

  `Std_Days_Between_Purchases`        Variation in purchase gaps

  `Purchase_Regularity`               Measure of consistency in
                                      purchasing

  `Spending_Trend`                    Month-over-month spending trend

  `Monthly_Spending_Mean`             Average monthly spending

  `Monthly_Spending_Std`              Monthly spending variability

  `Cancellation_Rate`                 Percentage of cancelled
                                      transactions

  `Is_UK`                             Whether the customer's primary
                                      country is the United Kingdom
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 4. RFM Scoring

After customer-level features are generated, the project creates
population-relative RFM scores.

The three main dimensions are:

``` text
R = Recency
F = Frequency
M = Monetary
```

Each dimension is converted into a **1--5 score using quintiles**.

### Recency

Recency is reversed because a more recent customer is more valuable:

``` text
Lower Recency → Higher R Score
```

### Frequency

Customers with more transactions receive a higher F Score:

``` text
Higher Frequency → Higher F Score
```

### Monetary

Customers who spend more receive a higher M Score:

``` text
Higher Monetary → Higher M Score
```

------------------------------------------------------------------------

# 5. Customer Segmentation

The RFM scores are converted into business-oriented customer segments.

The project includes segments such as:

-   **Champions**
-   **Loyal Customers**
-   **Best Customers**
-   **Good Customers**
-   **At Risk - Big Spenders**
-   **Cant Lose Them**
-   **Lost - Big Spenders**
-   **Lost Customers**
-   **Low-Value Customers**
-   **Hibernating**
-   **Potential**

These segments provide a more actionable interpretation of customer
behavior than raw numerical scores alone.

### Example

A customer with:

``` text
High Recency Score
High Frequency Score
High Monetary Score
```

can be classified as:

``` text
Champions
```

A customer with low recency performance but high spending can fall into:

``` text
Lost - Big Spenders
```

This distinction is useful because losing a high-value customer may
require a stronger retention strategy.

------------------------------------------------------------------------

# 6. Churn Identification

The project also creates a rule-based churn indicator during training:

``` text
Churn = 1 if Recency > 90 days
Churn = 0 otherwise
```

The 90-day threshold is configurable in:

``` text
rfm_pipeline/config.py
```

The churn flag is used for analysis/label creation and is **not directly
provided to the Random Forest as an input feature**.

------------------------------------------------------------------------

# 7. Machine Learning Model

The project uses:

``` text
RandomForestClassifier
```

The model learns the relationship between the engineered customer
behavior features and the RFM segment labels.

### Why Random Forest?

Random Forest is suitable for this project because it:

-   Handles nonlinear relationships
-   Works well with mixed customer behavior patterns
-   Is relatively robust to noise
-   Can model interactions between features
-   Provides class probabilities that can be used as prediction
    confidence
-   Performs well without requiring a complex neural network

------------------------------------------------------------------------

# 8. Train/Test Split

The labeled customer dataset is split into:

``` text
80% → Training
20% → Testing
```

The split uses:

``` text
random_state = 42
stratify = target
```

Stratification helps maintain the class distribution between training
and testing datasets.

------------------------------------------------------------------------

# 9. Feature Scaling

The 16 model features are transformed using:

``` text
StandardScaler
```

The scaler is:

``` text
fit only on training data
```

and then used to transform both training and test data.

The fitted scaler is saved so that new prediction data receives the same
transformation.

------------------------------------------------------------------------

# 10. Hyperparameter Tuning

For the normal training process, the project uses:

``` text
RandomizedSearchCV
```

with:

``` text
20 iterations
5-fold cross-validation
```

The search optimizes the Random Forest using:

``` text
weighted F1-score
```

Parameters searched include:

-   `n_estimators`
-   `max_depth`
-   `min_samples_split`
-   `min_samples_leaf`
-   `max_features`

The best model found in the current project uses:

``` text
n_estimators = 100
min_samples_split = 2
min_samples_leaf = 2
max_features = sqrt
max_depth = None
```

------------------------------------------------------------------------

# 11. Model Evaluation

The project evaluates the model using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Weighted F1-score
-   5-fold cross-validation

The saved evaluation results currently report approximately:

``` text
Test Accuracy       : 98.21%
Test Weighted F1    : 98.18%
CV Weighted F1      : 97.17%
```

These metrics indicate that the trained classifier performs strongly on
the available test data.

> Note: These results are dataset-dependent. They should not be
> interpreted as guaranteed performance on a different retail dataset or
> future production data.

------------------------------------------------------------------------

# 12. Model Artifacts

The trained model and preprocessing objects are stored in:

``` text
artifacts/
```

The directory contains:

``` text
artifacts/
├── random_forest.pkl
├── scaler.pkl
├── label_encoder.pkl
├── feature_columns.json
└── metrics.json
```

### Artifact purpose

  Artifact                 Purpose
  ------------------------ ------------------------------------------------
  `random_forest.pkl`      Trained Random Forest model
  `scaler.pkl`             Fitted StandardScaler
  `label_encoder.pkl`      Converts encoded classes back to segment names
  `feature_columns.json`   Stores the exact feature order
  `metrics.json`           Stores evaluation metrics and best parameters

Keeping these artifacts allows the prediction application to use the
trained model without retraining every time.

------------------------------------------------------------------------

# 13. Prediction Pipeline

For a new transaction dataset, the prediction flow is:

``` text
New Raw Transactions
        │
        ▼
Data Cleaning
        │
        ▼
Customer Feature Engineering
        │
        ▼
Load Saved Scaler
        │
        ▼
Feature Scaling
        │
        ▼
Load Random Forest
        │
        ▼
Predict Segment
        │
        ▼
Calculate Prediction Confidence
        │
        ▼
Customer ID + Segment + Confidence
```

An important design decision is that the **same cleaning and
feature-engineering functions are reused during training and
prediction**.

This reduces the risk of training-serving mismatch.

------------------------------------------------------------------------

# 14. Prediction Confidence

The Random Forest provides class probabilities using:

``` python
model.predict_proba()
```

The project takes the highest class probability as:

``` text
Prediction_Confidence
```

For example:

``` text
Predicted Segment: Champions
Prediction Confidence: 94.2%
```

This gives users an indication of how strongly the model favors the
predicted segment.

------------------------------------------------------------------------

# 15. Streamlit Application

The project includes an interactive web application:

``` text
streamlit_app.py
```

The application provides two prediction options.

### Option 1 --- Manual Input

Users can enter customer-level metrics such as:

-   Recency
-   Frequency
-   Monetary
-   Total products
-   Unique products
-   Average transaction value
-   Customer tenure
-   Spending trend
-   Cancellation rate
-   Country indicator

The application then displays:

``` text
Predicted RFM Segment
Prediction Confidence
```

### Option 2 --- Upload Transactions

Users can upload a CSV containing raw transactions.

The application automatically:

``` text
Upload CSV
   ↓
Clean Transactions
   ↓
Engineer Customer Features
   ↓
Predict Segments
   ↓
Display Results
   ↓
Download Predictions CSV
```

------------------------------------------------------------------------

# 16. Project Structure

``` text
customer-segmentation-retention-analysis/
│
├── artifacts/
│   ├── feature_columns.json
│   ├── label_encoder.pkl
│   ├── metrics.json
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.csv
│   │
│   └── processed/
│       └── clean_customer_data.csv
│
├── notebook/
│   ├── CSRA.ipynb
│   └── model.ipynb
│
├── rfm_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── predict.py
│   └── train.py
│
├── main.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# 🚀 Installation

## 1. Clone the repository

``` bash
git clone <your-repository-url>
cd customer-segmentation-retention-analysis
```

## 2. Create a virtual environment

Windows:

``` bash
python -m venv venv
```

Activate it:

``` bash
venv\Scripts\activate
```

## 3. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# ▶️ Train the Model

To run the complete training pipeline:

``` bash
python main.py train --input data/raw/online_retail_II.csv
```

This performs:

``` text
Raw Data
→ Cleaning
→ Feature Engineering
→ RFM Labeling
→ Train/Test Split
→ Scaling
→ Hyperparameter Tuning
→ Random Forest Training
→ Evaluation
→ Artifact Saving
```

------------------------------------------------------------------------

## ⚡ Quick Training

For a faster smoke test without hyperparameter search:

``` bash
python main.py train --input data/raw/online_retail_II.csv --quick
```

------------------------------------------------------------------------

# 🔮 Run Predictions

To predict segments from a new transaction CSV:

``` bash
python main.py predict \
    --input data/new_transactions.csv \
    --output predictions.csv
```

You can also specify an explicit snapshot date:

``` bash
python main.py predict \
    --input data/new_transactions.csv \
    --output predictions.csv \
    --snapshot-date 2026-09-14
```

The output contains:

``` text
Customer ID
Predicted_RFM_Segment
Prediction_Confidence
```

------------------------------------------------------------------------

# 🖥️ Run the Streamlit Application

Start the application using:

``` bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

------------------------------------------------------------------------

# 🧠 Important Design Principle

The project separates **segment generation** from **segment
prediction**.

### During training

``` text
Transactions
→ Features
→ RFM Quantile Scores
→ Rule-Based Segment
→ Random Forest learns the mapping
```

### During prediction

``` text
New Transactions
→ Features
→ Random Forest
→ Predicted Segment
```

The prediction pipeline does **not** calculate new population RFM
quantiles for every prediction batch. Instead, it uses the trained
classifier to approximate the learned segment definitions from customer
behavior features.

This makes the prediction process easier to operationalize for new
customer data.

------------------------------------------------------------------------

# 📊 Business Use Cases

The resulting customer segments can support:

### Customer Retention

Identify customers such as:

``` text
At Risk - Big Spenders
Lost - Big Spenders
Cant Lose Them
```

and prioritize retention efforts.

### Customer Loyalty

Identify:

``` text
Champions
Loyal Customers
Best Customers
```

for loyalty programs and personalized offers.

### Re-engagement

Identify:

``` text
Lost Customers
Hibernating
```

for win-back campaigns.

### Cross-Selling

Use purchase frequency and product diversity to identify customers who
may be suitable for cross-sell recommendations.

### Marketing Segmentation

Create targeted campaigns based on customer behavior instead of sending
the same campaign to every customer.

------------------------------------------------------------------------

# 🛠️ Technology Stack

  Category                    Technologies
  --------------------------- --------------------------
  Programming                 Python
  Data Processing             Pandas, NumPy
  Statistics                  SciPy
  Machine Learning            Scikit-learn
  Model                       Random Forest Classifier
  Hyperparameter Tuning       RandomizedSearchCV
  Visualization               Matplotlib, Seaborn
  Model Persistence           Joblib
  Experiment/Model Tracking   MLflow
  Web Application             Streamlit
  Notebook                    Jupyter

------------------------------------------------------------------------

# 📈 Key Project Outcomes

This project demonstrates an end-to-end machine learning workflow:

-   Raw transaction data processing
-   Data cleaning
-   Exploratory analysis
-   RFM analysis
-   Advanced feature engineering
-   Customer segmentation
-   Churn-oriented analysis
-   Supervised machine learning
-   Hyperparameter tuning
-   Cross-validation
-   Model evaluation
-   Model artifact management
-   Batch prediction
-   Interactive Streamlit deployment

------------------------------------------------------------------------

# 🔧 Future Improvements

Possible improvements for a production version include:

1.  Add automated data validation.
2.  Add scheduled model retraining.
3.  Track experiments and model versions consistently with MLflow.
4.  Add data drift and model drift monitoring.
5.  Add explainability using SHAP.
6.  Add a customer-level dashboard with segment trends.
7.  Add automated retention recommendations for each segment.
8.  Deploy the Streamlit application to a cloud platform.
9.  Add CI/CD testing and automated deployment.
10. Add a database/data warehouse integration for continuously updated
    customer features.

------------------------------------------------------------------------

# 👤 Author

**Sahil Laskar**

This project was developed as an end-to-end Data Science / Machine
Learning project focused on customer analytics, segmentation, and
retention.

------------------------------------------------------------------------

## ⭐ Summary

**Customer Segmentation & Retention Analysis** transforms raw retail
transactions into actionable customer intelligence.

The complete system:

``` text
Raw Transactions
      ↓
Data Cleaning
      ↓
Customer-Level Features
      ↓
RFM Scoring
      ↓
Customer Segments
      ↓
Random Forest Model
      ↓
Evaluation & Artifacts
      ↓
New Customer Prediction
      ↓
Segment + Confidence
      ↓
Business Retention Actions
```

The main objective is to help businesses understand **who their
customers are, how they behave, which customers are valuable or at risk,
and where retention efforts should be focused**.
