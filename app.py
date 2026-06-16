import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model  # Import load_model for Keras .h5 files

# Set page configuration
st.set_page_config(layout="wide", page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict if they will churn.")


# Load the model and scaler
@st.cache_resource
def load_artifacts():
    try:
        # Load the Keras model from .h5 file
        model = load_model('churn_prediction_model.h5')

        # Load the StandardScaler from .pkl file
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()


model, scaler = load_artifacts()

# Define the input features based on the x_train columns
# These are the columns after one-hot encoding and dropping original categoricals
feature_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                   'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                   'Geography_Germany', 'Geography_Spain', 'Gender_Male']

# Create input widgets
with st.form("churn_prediction_form"):
    st.header("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider("Credit Score", 350, 850, 600)
        age = st.slider("Age", 18, 92, 35)
        tenure = st.slider("Tenure (Years at Bank)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)

    with col2:
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 75000.0)

    with col3:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Prepare input data for prediction
        input_data = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary,
            'Geography_Germany': 1 if geography == 'Germany' else 0,
            'Geography_Spain': 1 if geography == 'Spain' else 0,
            'Gender_Male': 1 if gender == 'Male' else 0
        }

        # Create a DataFrame with the same column order as during training
        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Scale the input data
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction_prob = model.predict(scaled_input)[0][0]
        prediction_binary = (prediction_prob > 0.5).astype(int)

        st.subheader("Prediction Results")
        if prediction_binary == 1:
            st.error(f"This customer is likely to churn.")
        else:
            st.success(f"This customer is unlikely to churn.")

        #st.write("Raw input data:", input_df)
        #st.write("Scaled input data (first 5 values):")
        st.write(scaled_input[0, :5])  # Display first 5 values for brevity