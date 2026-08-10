import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Yes Bank Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

FEATURES = ["Open", "High", "Low", "Month", "Year"]

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_resource

def train_model(uploaded_file):
    data = pd.read_csv(uploaded_file).copy()

    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["Date"] = pd.to_datetime(
        data["Date"],
        format="%b-%y",
        errors="coerce"
    )

    if data["Date"].isna().any():
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

    data = data.dropna(
        subset=["Date", "Open", "High", "Low", "Close"]
    ).copy()

    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year

    FEATURES = ["Open", "High", "Low", "Month", "Year"]

    X = data[FEATURES]
    y = data["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    metrics = {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred) * 100
    }

    return (
        data,
        model,
        scaler,
        metrics,
        y_test.reset_index(drop=True),
        pd.Series(y_pred)
    )

st.title("📈 Yes Bank Stock Price Prediction")
st.caption("Machine Learning Regression App | Linear Regression")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader(
        "Upload data_YesBank_StockPrices.csv",
        type=["csv"]
    )
    st.info(
        "Expected columns: Date, Open, High, Low, Close. "
        "The app derives Month and Year from Date."
    )

if uploaded_file is None:
    st.warning("Upload the original Yes Bank CSV dataset to start the application.")
    st.markdown("""
    ### Project objective
    Predict the **closing price** of Yes Bank using:
    - Open
    - High
    - Low
    - Month
    - Year

    The notebook's final selected model is **Linear Regression**.
    """)
    st.stop()

try:
    df, model, scaler, metrics, y_test, y_pred = train_model(uploaded_file)
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🎯 Prediction", "📈 Model Performance", "📋 Data"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Features", 5)
    c3.metric("R² Score", f"{metrics['R2']:.2f}%")
    c4.metric("MAE", f"₹{metrics['MAE']:.2f}")

    st.subheader("Closing Price Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (₹)")
    ax.set_title("Yes Bank Closing Price Over Time")
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("Predict Closing Price")
    st.write("Enter the same input features used by the trained model.")

    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Month / Year")
        open_price = st.number_input("Open Price (₹)", min_value=0.0, value=100.0)
    with col2:
        high_price = st.number_input("High Price (₹)", min_value=0.0, value=110.0)
        low_price = st.number_input("Low Price (₹)", min_value=0.0, value=90.0)
    with col3:
        st.metric("Selected Month", date.month)
        st.metric("Selected Year", date.year)

    if st.button("Predict Closing Price", type="primary"):
        input_df = pd.DataFrame([{
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Month": date.month,
            "Year": date.year
        }])
        input_scaled = scaler.transform(input_df[FEATURES])
        prediction = model.predict(input_scaled)[0]

        st.success(f"Predicted Closing Price: ₹{prediction:,.2f}")
        st.caption(
            "This is a machine-learning estimate based on the historical dataset "
            "and selected input features; it is not investment advice."
        )

with tab3:
    st.subheader("Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{metrics['R2']:.2f}%")
    c2.metric("RMSE", f"₹{metrics['RMSE']:.2f}")
    c3.metric("MAE", f"₹{metrics['MAE']:.2f}")
    c4.metric("MSE", f"{metrics['MSE']:.2f}")

    comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(comparison["Actual"].values, label="Actual")
    ax.plot(comparison["Predicted"].values, label="Predicted")
    ax.set_title("Actual vs Predicted Closing Price")
    ax.set_ylabel("Close Price (₹)")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Linear Regression Coefficients")
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=np.abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.write("Shape:", df.shape)
