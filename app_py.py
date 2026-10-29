import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler (using relative paths)
@st.cache_resource
def load_artifacts():
    model = joblib.load('olist_kmeans_model.pkl')
    scaler = joblib.load('olist_scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# App Header
st.title("Fabulous Data School: E-Commerce Segmentation")
st.write("Input a customer's RFM metrics to predict their consumer segment.")

# Sidebar for User Inputs
st.sidebar.header("Customer RFM Inputs")
recency = st.sidebar.number_input("Recency (Days since last purchase)", min_value=0, value=45)
frequency = st.sidebar.number_input("Frequency (Number of orders)", min_value=1, value=2)
monetary = st.sidebar.number_input("Monetary Value (Total spend in BRL)", min_value=0.0, value=150.0)

# Prediction Logic
if st.sidebar.button("Predict Customer Segment"):
    # Format input as dataframe
    input_data = pd.DataFrame([[recency, frequency, monetary]],
                              columns=['Recency', 'Frequency', 'Monetary'])

    # Scale the input
    scaled_data = scaler.transform(input_data)

    # Predict the cluster
    cluster = model.predict(scaled_data)[0]

    # Map cluster to business logic
    segment_map = {
        0: ("Recent & Low Spend", "Newer customers who have placed small orders.", "info"),
        1: ("At Risk", "Haven't purchased in a long time with low engagement.", "error"),
        2: ("Loyal Spenders", "Frequent buyers with steady, average spending.", "warning"),
        3: ("Whales / Champions", "High frequency and massive total monetary value.", "success")
    }

    segment_name, description, color = segment_map.get(cluster, ("Unknown", "", "info"))

    # Display Results
    st.subheader(f"Predicted Segment: **{segment_name}**")
    st.info(description, icon="ℹ️")
