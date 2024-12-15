import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import openai

# Title
st.title("SCM Lead Time Prediction App")

# Sidebar: File Uploads
st.sidebar.title("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Initialize prediction as None
prediction = None

# Load Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Preprocess Data
    st.write("### Preprocessing Data...")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"])
    df["Lead_Time"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days

    X = df[["Order_Size", "Shipping_Distance", "Supplier_Reliability"]]
    y = df["Lead_Time"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"### Model Evaluation\nMean Squared Error: {mse:.2f}")

    # Prediction Inputs
    st.write("### Predict Lead Time")
    order_size = st.number_input("Order Size", min_value=1, value=100)
    shipping_distance = st.number_input("Shipping Distance (km)", min_value=1, value=500)
    supplier_reliability = st.number_input("Supplier Reliability (%)", min_value=50, max_value=100, value=90)

    if st.button("Predict"):
        input_data = np.array([[order_size, shipping_distance, supplier_reliability]])
        prediction = model.predict(input_data)[0]
        st.session_state["prediction"] = prediction  # Store prediction in session state
        st.success(f"Predicted Lead Time: {prediction:.2f} days")

    # Generative AI Insights
    if api_key and "prediction" in st.session_state:
        prediction = st.session_state["prediction"]
        if st.button("Ask AI for Insights"):
            try:
                openai.api_key = api_key
                prompt = (
                    f"Explain why the lead time prediction is {prediction:.2f} days "
                    f"given an order size of {order_size}, a shipping distance of {shipping_distance} km, "
                    f"and a supplier reliability of {supplier_reliability}."
                )
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                )
                st.write(response.choices[0].message.content.strip())
            except Exception as e:
                st.error(f"Error with Generative AI: {e}")
    else:
        st.warning("Please make a prediction before asking AI for insights.")
else:
    st.warning("Please upload a CSV file to proceed.")
