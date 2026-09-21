import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model and column structure
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# Extract dropdown options from model columns
titles = [col.replace('title_', '') for col in model_columns if col.startswith('title_')]
states = [col.replace('state_', '') for col in model_columns if col.startswith('state_')]
towns = [col.replace('town_grouped_', '') for col in model_columns if col.startswith('town_grouped_')]
towns = sorted([t for t in towns if t != 'Other']) + ['Other']

st.title("🏠 Nigeria House Price Predictor")
st.write("Enter property details to get a predicted price.")

# Input widgets
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=4)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=4)
toilets = st.number_input("Toilets", min_value=1, max_value=20, value=5)
parking_space = st.number_input("Parking Space", min_value=0, max_value=20, value=4)
title = st.selectbox("House Type", titles)
state = st.selectbox("State", states)
town = st.selectbox("Town", towns)

if st.button("Predict Price"):
    # Build input row, matching training structure
    input_data = pd.DataFrame([[bedrooms, bathrooms, toilets, parking_space]],
                                columns=['bedrooms', 'bathrooms', 'toilets', 'parking_space'])

    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    title_col = f'title_{title}'
    state_col = f'state_{state}'
    town_col = f'town_grouped_{town}'
    if title_col in input_data.columns:
        input_data[title_col] = 1
    if state_col in input_data.columns:
        input_data[state_col] = 1
    if town_col in input_data.columns:
        input_data[town_col] = 1

    input_data = input_data[model_columns]

    log_pred = model.predict(input_data)[0]
    prediction = round(np.exp(log_pred))

    st.success(f"Based on: {bedrooms} Bedrooms, {bathrooms} Bathrooms, {toilets} Toilets, "
               f"{parking_space} Parking Spaces, {title} in {town}, {state}")
    st.header(f"Predicted Price: ₦{prediction:,}")