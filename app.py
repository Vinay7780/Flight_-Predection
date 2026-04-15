import streamlit as st
import pickle
import numpy as np
import warnings

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model  ───────────────────────────
@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    import gdown
    import os

    model_path = "flight_price_model.pkl"

    # Download model if not present
    if not os.path.exists(model_path):
        url = "url = "https://drive.google.com/uc?export=download&id=1A1muCU1X_vYLOWnszvjrpNoeuDk8LZzR"
        gdown.download(url, model_path, quiet=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pickle.load(open(model_path, "rb"))

model = load_model()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}
.stApp {
    background-color: #F0F2F5;
}
.hero {
    text-align: center;
    padding: 3rem;
    background: white;
    border-radius: 10px;
    margin-bottom: 1rem;
}
div.stButton > button {
    width: 100%;
    background: #1e40af;
    color: white;
    border-radius: 8px;
}
.footer {
    text-align: center;
    font-size: 0.7rem;
    color: gray;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CITIES = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Bangalore"]
AIRLINES = ["Air India", "GO FIRST", "Indigo", "SpiceJet", "Vistara", "AirAsia"]

# ── Simple Feature Builder (keep same as training) ─────────────────────────────
def build_features(duration_hrs, duration_mins, days_left, source, destination):
    total_mins = duration_hrs * 60 + duration_mins
    return np.array([[duration_hrs, days_left, total_mins]])

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
<h1>Flight Price Predictor</h1>
<p>AI-powered fare estimation</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    source = st.selectbox("Source", CITIES)
    destination = st.selectbox("Destination", [c for c in CITIES if c != source])
    airline = st.selectbox("Airline", AIRLINES)

with col2:
    duration_hrs = st.number_input("Duration Hours", 0, 24, 2)
    duration_mins = st.number_input("Duration Minutes", 0, 59, 30)
    days_left = st.number_input("Days Left", 1, 365, 30)

predict = st.button("Predict Price")

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict:
    features = build_features(duration_hrs, duration_mins, days_left, source, destination)

    try:
        price = model.predict(features)[0]
        st.success(f"Estimated Price: ₹ {price:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Flight Price Prediction System</div>', unsafe_allow_html=True)