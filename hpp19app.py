import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and data columns
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

def load_columns():
    with open('columns.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
columns = load_columns()

# App UI
st.title("🏡 Bengaluru House Price Predictor-19")
st.markdown("Enter the details below to get an estimated house price in Bengaluru 💰")

# User Inputs
location = st.selectbox("Select Location", sorted(columns['locations']))
sqft = st.number_input("Total Square Feet", min_value=300, step=10)
bath = st.slider("Number of Bathrooms", 1, 10)
bhk = st.slider("BHK (Bedrooms)", 1, 10)

# Prediction
if st.button("Predict Price"):
    input_data = np.zeros(len(columns['data_columns']))
    input_data[0] = sqft
    input_data[1] = bath
    input_data[2] = bhk

    loc_index = columns['data_columns'].index(location.lower()) if location.lower() in columns['data_columns'] else -1
    if loc_index >= 0:
        input_data[loc_index] = 1

    prediction = model.predict([input_data])[0]
    st.success(f"Estimated Price: ₹ {round(prediction, 2)} Lakhs")


import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save column info (you need location dummies and column list)
data_columns = {
    'data_columns': list(X.columns),  # your input features
    'locations': list(unique_locations)  # your cleaned list of locations
}
with open('columns.pkl', 'wb') as f:
    pickle.dump(data_columns, f)
