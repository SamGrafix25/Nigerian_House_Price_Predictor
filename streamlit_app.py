import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Nigeria House Price Predictor", page_icon="🏠", layout="centered")

# CSS
st.markdown("""
    <style>
        .block-container {
            max-width: 800px;
            margin: auto;
            padding-top: 2rem;
        }

        h1 {
            text-align: center;
        }

        .stButton > button {
            background: linear-gradient(90deg, #6c63ff, #48cae4) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-size: 1.1rem !important;
            font-weight: bold !important;
        }
        .stButton > button:hover {
            opacity: 0.85 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load models
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

df = pd.read_csv('nigeria_houses_data.csv')
state_town_map = (
    df.groupby('state')['town']
    .apply(lambda x: sorted(x.dropna().unique().tolist()))
    .to_dict()
)

def get_price_label(price):
    if price < 10_000_000:
        return " Budget", "#27ae60"
    elif price < 50_000_000:
        return " Mid-Range", "#f39c12"
    elif price < 200_000_000:
        return " Premium", "#e67e22"
    else:
        return " Luxury", "#8e44ad"

# Extract dropdown options
titles = [col.replace('title_', '') for col in model_columns if col.startswith('title_')]
states = [col.replace('state_', '') for col in model_columns if col.startswith('state_')]
towns = [col.replace('town_grouped_', '') for col in model_columns if col.startswith('town_grouped_')]
towns = sorted([t for t in towns if t != 'Other']) + ['Other']

# Header
st.markdown("<h1>🏠 Nigeria House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter property details below to get an estimated market price.</p>", unsafe_allow_html=True)
st.divider()

# Numeric inputs
col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("🛏 Bedrooms", min_value=1, max_value=20, value=4)
    toilets = st.number_input("🚽 Toilets", min_value=1, max_value=20, value=5)
with col2:
    bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=20, value=4)
    parking_space = st.number_input("🚗 Parking Space", min_value=0, max_value=20, value=4)

st.divider()

# Dropdowns
col3, col4, col5 = st.columns(3)
with col3:
    title = st.selectbox("🏡 House Type", titles)
with col4:
    state = st.selectbox("📍 State", states)
with col5:
    available_towns = state_town_map.get(state, towns)
    town = st.selectbox("🏘 Town", available_towns)
st.divider()

# Predict button
if st.button("Predict Price", use_container_width=True, type="primary"):
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

    price_formatted = f"₦{prediction:,}"
    summary = f"{title} in {town}, {state}"
    details = f"{bedrooms} bed · {bathrooms} bath · {toilets} toilets · {parking_space} parking"
    label, label_color = get_price_label(prediction)

    st.markdown(f"""
        <div style="border: 2px solid #6c63ff; border-radius: 12px; padding: 24px 32px; margin-top: 16px;">
            <p style="color: grey; margin: 0 0 4px 0; font-size: 0.9rem;">Estimated Price — {summary}</p>
            <h2 style="color: #6c63ff; margin: 0 0 4px 0; font-size: 2.2rem;">{price_formatted}</h2>
            <span style="background: {label_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">{label}</span>
            <p style="color: grey; margin: 8px 0 0 0; font-size: 0.85rem;">{details}</p>
        </div>
    """, unsafe_allow_html=True)

    # Price breakdown chart
        # Price context
    st.markdown("#### 📊 Price Range Context")
    
    ranges = {
        "Budget\n(Under ₦10M)": 10_000_000,
        "Mid-Range\n(₦10M–₦50M)": 50_000_000,
        "Premium\n(₦50M–₦200M)": 200_000_000,
        "Luxury\n(Above ₦200M)": 500_000_000,
    }

    categories = list(ranges.keys())
    max_prices = list(ranges.values())
    bar_colors = ["#27ae60" if prediction <= v else "#e0e0e0" for v in max_prices]
    # Highlight the matching tier
    bar_colors = []
    for i, v in enumerate(max_prices):
        prev = max_prices[i-1] if i > 0 else 0
        if prev < prediction <= v:
            bar_colors.append("#6c63ff")
        elif prediction > v:
            bar_colors.append("#cccccc")
        else:
            bar_colors.append("#e0e0e0")

    fig = go.Figure(go.Bar(
        x=categories,
        y=[10, 40, 150, 300],
        marker_color=bar_colors,
        text=["<₦10M", "₦10M–50M", "₦50M–200M", ">₦200M"],
        textposition="inside",
        insidetextanchor="middle",
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(visible=False),
        xaxis=dict(tickfont=dict(size=11)),
        height=250,
        margin=dict(t=20, b=20),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Your property falls in the **{label}** tier.")