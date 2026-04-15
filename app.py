import streamlit as st
import pickle
import numpy as np
import warnings

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Predictor",
    layout="wide",
)

# ── Load model from Google Drive ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    import gdown
    import os

    model_path = "flight_price_model.pkl"

    # ✅ Correct Google Drive direct download link
    url = "https://drive.google.com/uc?export=download&id=1A1muCU1X_vYLOWnszvjrpNoeuDk8LZzR"

    # Download only if not exists
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pickle.load(open(model_path, "rb"))

    return model

model = load_model()

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Flight Price Predictor")

# ── Constants ─────────────────────────────────────────────────────────────────
CITIES = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Bangalore"]
AIRLINES = ["Air India", "GO FIRST", "Indigo", "SpiceJet", "Vistara", "AirAsia"]

# ── Feature Builder ───────────────────────────────────────────────────────────
def build_features(duration_hrs, duration_mins, days_left):
    total_mins = duration_hrs * 60 + duration_mins
    return np.array([[duration_hrs, days_left, total_mins]])

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    source = st.selectbox("Source", CITIES)
    destination = st.selectbox("Destination", [c for c in CITIES if c != source])
    airline = st.selectbox("Airline", AIRLINES)

with col2:
    duration_hrs = st.number_input("Duration Hours", 0, 24, 2)
    duration_mins = st.number_input("Duration Minutes", 0, 59, 30)
    days_left = st.number_input("Days Left", 1, 365, 30)

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("Predict Price"):
    try:
        features = build_features(duration_hrs, duration_mins, days_left)
        price = model.predict(features)[0]
        st.success(f"Estimated Price: ₹ {price:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")