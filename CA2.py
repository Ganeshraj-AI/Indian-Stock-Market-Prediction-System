# ============================================================
# 📈 INDIAN STOCK MARKET PREDICTION SYSTEM
# Machine Learning CA2 Project
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("📈 Indian Stock Market Prediction System")

st.write(
    "Machine Learning based prediction and analysis "
    "of Indian stock market prices."
)


# ============================================================
# 3. STOCK SELECTION
# ============================================================

stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "NIFTY 50": "^NSEI"
}

selected_stock = st.selectbox(
    "📊 Select Stock",
    list(stocks.keys())
)

ticker = stocks[selected_stock]


# ============================================================
# 4. DOWNLOAD STOCK DATA
# ============================================================

st.info("Fetching latest stock market data...")

try:

    data = yf.download(
        ticker,
        period="2y",
        progress=False,
        auto_adjust=False
    )

except Exception as e:

    st.error(
        f"Unable to download stock data: {e}"
    )

    st.stop()


# ============================================================
# 5. CHECK DATA
# ============================================================

if data.empty:

    st.error(
        "No stock data was found. "
        "Please check your internet connection."
    )

    st.stop()


# ============================================================
# 6. FIX MULTI-INDEX COLUMNS
# ============================================================

if isinstance(data.columns, pd.MultiIndex):

    data.columns = data.columns.get_level_values(0)


# ============================================================
# 7. RESET INDEX
# ============================================================

data = data.reset_index()


# ============================================================
# 8. DATA CLEANING
# ============================================================

data = data.drop_duplicates()

data = data.dropna()


# Make sure Date is datetime
data["Date"] = pd.to_datetime(
    data["Date"]
)


# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

data["Previous_Close"] = (
    data["Close"].shift(1)
)

data["Day"] = (
    data["Date"].dt.day
)

data["Month"] = (
    data["Date"].dt.month
)

data["Weekday"] = (
    data["Date"].dt.weekday
)

data["Moving_Average"] = (
    data["Close"]
    .rolling(window=10)
    .mean()
)

data["Daily_Return"] = (
    data["Close"]
    .pct_change()
)


# Remove rows containing NaN
data = data.dropna()


# ============================================================
# 10. FEATURES AND TARGET
# ============================================================

features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Previous_Close",
    "Day",
    "Month",
    "Weekday",
    "Moving_Average",
    "Daily_Return"
]

X = data[features]

y = data["Close"]


# ============================================================
# 11. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

split = int(
    len(data) * 0.80
)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]


# ============================================================
# 12. MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
}


# ============================================================
# 13. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

predictions = {}


for model_name, model in models.items():

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predict
    pred = model.predict(
        X_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append([
        model_name,
        mae,
        rmse,
        r2
    ])

    trained_models[
        model_name
    ] = model

    predictions[
        model_name
    ] = pred


# ============================================================
# 14. MODEL EVALUATION TABLE
# ============================================================

result_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2 Score"
    ]
)


# ============================================================
# 15. DISPLAY MODEL RESULTS
# ============================================================

st.subheader(
    "🤖 Model Evaluation"
)

st.dataframe(
    result_df.round(3),
    use_container_width=True
)


# ============================================================
# 16. FIND BEST MODEL
# ============================================================

best_index = result_df[
    "R2 Score"
].idxmax()

best_model_name = result_df.loc[
    best_index,
    "Model"
]

best_model = trained_models[
    best_model_name
]


st.success(
    f"🏆 Best Model: {best_model_name}"
)


# ============================================================
# 17. MODEL COMPARISON GRAPH
# ============================================================

st.subheader(
    "📊 Model Performance Comparison"
)

fig_model = px.bar(
    result_df,
    x="Model",
    y="R2 Score",
    title="R² Score Comparison"
)

st.plotly_chart(
    fig_model,
    use_container_width=True
)


# ============================================================
# 18. CURRENT STOCK INFORMATION
# ============================================================

last = data.iloc[-1]


st.subheader(
    f"📌 {selected_stock} - Latest Information"
)


c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "Open",
    f"₹{float(last['Open']):.2f}"
)

c2.metric(
    "High",
    f"₹{float(last['High']):.2f}"
)

c3.metric(
    "Low",
    f"₹{float(last['Low']):.2f}"
)

c4.metric(
    "Close",
    f"₹{float(last['Close']):.2f}"
)

c5.metric(
    "Volume",
    f"{int(float(last['Volume'])):,}"
)


# ============================================================
# 19. PREDICT NEXT CLOSING PRICE
# ============================================================

latest_features = X.iloc[[-1]]

predicted_price = best_model.predict(
    latest_features
)[0]


st.subheader(
    "🔮 Predicted Closing Price"
)


st.metric(
    "Predicted Next Close",
    f"₹{predicted_price:.2f}"
)


# ============================================================
# 20. PRICE CHANGE
# ============================================================

current_price = float(
    last["Close"]
)

price_difference = (
    predicted_price -
    current_price
)

percentage_change = (
    price_difference /
    current_price
) * 100


if percentage_change > 0:

    st.success(
        f"📈 Expected Change: "
        f"+{percentage_change:.2f}%"
    )

elif percentage_change < 0:

    st.error(
        f"📉 Expected Change: "
        f"{percentage_change:.2f}%"
    )

else:

    st.info(
        "➡️ Expected Change: 0.00%"
    )


# ============================================================
# 21. CLOSING PRICE CHART
# ============================================================

st.subheader(
    "📈 Closing Price Trend"
)

fig1 = px.line(
    data,
    x="Date",
    y="Close",
    title=f"{selected_stock} Closing Price"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# ============================================================
# 22. MOVING AVERAGE CHART
# ============================================================

st.subheader(
    "📊 Closing Price vs Moving Average"
)

fig2 = px.line(
    data,
    x="Date",
    y=[
        "Close",
        "Moving_Average"
    ],
    title="Close Price vs 10-Day Moving Average"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# ============================================================
# 23. DAILY RETURN CHART
# ============================================================

st.subheader(
    "📉 Daily Return"
)

fig3 = px.line(
    data,
    x="Date",
    y="Daily_Return",
    title="Daily Stock Return"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)


# ============================================================
# 24. ACTUAL VS PREDICTED
# ============================================================

st.subheader(
    "🎯 Actual vs Predicted Closing Price"
)

comparison_df = pd.DataFrame({

    "Date": data.iloc[
        split:
    ]["Date"].values,

    "Actual": y_test.values,

    "Predicted": predictions[
        best_model_name
    ]

})


fig4 = px.line(
    comparison_df,
    x="Date",
    y=[
        "Actual",
        "Predicted"
    ],
    title="Actual vs Predicted Price"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)


# ============================================================
# 25. FEATURE IMPORTANCE
# ============================================================

if best_model_name == "Random Forest":

    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance":
            best_model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    st.subheader(
        "🔍 Feature Importance"
    )

    fig5 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )


# ============================================================
# 26. HISTORICAL DATA
# ============================================================

st.subheader(
    "📋 Historical Stock Data"
)

st.dataframe(
    data.tail(20),
    use_container_width=True
)


# ============================================================
# 27. PROJECT INFORMATION
# ============================================================

st.sidebar.title(
    "📚 Project Information"
)

st.sidebar.write(
    """
    **Project Title**

    Indian Stock Market Prediction
    System Using Machine Learning

    **ML Models**

    • Linear Regression

    • Decision Tree Regression

    • Random Forest Regression

    **Evaluation Metrics**

    • MAE

    • RMSE

    • R² Score

    **Data Source**

    Yahoo Finance

    **Stocks**

    • Reliance

    • TCS

    • Infosys

    • HDFC Bank

    • NIFTY 50
    """
)


# ============================================================
# 28. DISCLAIMER
# ============================================================

st.warning(
    "⚠️ This prediction is for educational/project "
    "purposes only and should not be considered financial advice."
)


st.caption(
    "Indian Stock Market Prediction System | ML CA2 Project"
)