import streamlit as st
import pickle
import numpy as np
import warnings
import os
import gdown

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    /* ── Background: warm off-white with very subtle grid texture ── */
    .stApp {
        background-color: #F0F2F5;
        background-image:
            linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px);
        background-size: 32px 32px;
        min-height: 100vh;
    }

    /* ── Content container ── */
    .block-container {
        max-width: 980px;
        padding-top: 0.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* ── Hero header ── */
    .hero {
        text-align: center;
        padding: 4rem 1rem 3rem;
        background-image: linear-gradient(rgba(240, 242, 245, 0.85), rgba(240, 242, 245, 1)), url('https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=2000&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        border-radius: 0 0 24px 24px;
        margin-bottom: 2rem;
        margin-top: -3rem;
    }
    .hero h1 {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin-bottom: .25rem;
    }
    .hero p {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 500;
        color: #475569;
        margin-top: 0;
        letter-spacing: 0.01em;
    }

    /* ── Section captions ── */
    .stCaption, [data-testid="stCaptionContainer"] p {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #2563eb !important;
    }

    /* ── Horizontal divider ── */
    hr {
        border-color: #e2e8f0 !important;
        margin: 0.6rem 0 1rem !important;
    }

    /* ── Form labels ── */
    label, .stSelectbox label, .stNumberInput label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
        letter-spacing: 0.01em !important;
    }

    /* ── Dropdowns & inputs ── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 7px !important;
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
        transition: border-color .15s !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    }

    /* ── Predict button ── */
    div.stButton > button {
        width: 100%;
        padding: 0.85rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        background: #1e40af;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        letter-spacing: 0.04em;
        transition: background .2s, box-shadow .2s, transform .15s;
        box-shadow: 0 2px 10px rgba(30,64,175,0.3);
    }
    div.stButton > button:hover {
        background: #1d4ed8;
        box-shadow: 0 6px 20px rgba(30,64,175,0.35);
        transform: translateY(-2px);
    }
    div.stButton > button:active {
        background: #1e3a8a;
        transform: translateY(0);
        box-shadow: 0 1px 5px rgba(30,64,175,0.2);
    }

    /* ── Error messages ── */
    .stAlert {
        border-radius: 8px !important;
        font-size: 0.875rem !important;
    }

    /* ── Result panel ── */
    .result-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #1e40af;
        border-radius: 12px;
        padding: 2rem 2.4rem;
        margin-top: 2rem;
        box-shadow: 0 4px 24px rgba(15,23,42,0.08);
        animation: fadeIn .45s ease;
    }
    .result-panel .rp-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.4rem;
    }
    .result-panel .rp-route {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.01em;
    }
    .result-panel .rp-tag {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        padding: .3rem .85rem;
        border-radius: 20px;
        letter-spacing: .06em;
        text-transform: uppercase;
        border: 1px solid #bfdbfe;
    }
    .result-panel .rp-price-block {
        text-align: center;
        padding: 1.4rem 0;
        border-top: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 1.4rem;
    }
    .result-panel .rp-price-label {
        font-family: 'Inter', sans-serif;
        font-size: .72rem;
        font-weight: 500;
        color: #94a3b8;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .3rem;
    }
    .result-panel .rp-price {
        font-family: 'DM Sans', sans-serif;
        font-size: 3.6rem;
        font-weight: 700;
        color: #1e40af;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .result-panel .rp-price-note {
        font-family: 'Inter', sans-serif;
        font-size: .75rem;
        color: #94a3b8;
        margin-top: .35rem;
    }
    .result-panel .rp-stats {
        display: flex;
        gap: 0;
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #f1f5f9;
        overflow: hidden;
    }
    .result-panel .rp-stat {
        flex: 1;
        text-align: center;
        padding: .85rem .5rem;
        border-right: 1px solid #e2e8f0;
    }
    .result-panel .rp-stat:last-child {
        border-right: none;
    }
    .result-panel .rp-stat-label {
        font-family: 'Inter', sans-serif;
        font-size: .68rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: .3rem;
    }
    .result-panel .rp-stat-value {
        font-family: 'DM Sans', sans-serif;
        font-size: .95rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: .73rem;
        color: #94a3b8;
        letter-spacing: .03em;
        padding: 2rem 0 1rem;
    }

    /* ── Animation ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0);    }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading / Loading model (~1.7GB model)... Please wait!!")
def load_model():
    model_path = "flight_price_model.pkl"
    file_id = "1A1muCU1X_vYLOWnszvjrpNoeuDk8LZzR"
    
    # Check if we need to download it from GDrive
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pickle.load(open(model_path, "rb"))

model = load_model()

# ── Constants ──────────────────────────────────────────────────────────────────
CITIES       = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Bangalore"]
AIRLINES     = ["Air India", "GO FIRST", "Indigo", "SpiceJet", "Vistara", "AirAsia"]
TIME_SLOTS   = ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"]
STOPS        = ["Non-stop (0)", "One stop (1)", "Two or more stops (2+)"]
FLIGHT_CLASS = ["Economy", "Business"]

# ── Route encoding ─────────────────────────────────────────────────────────────
ROUTE_MAP = {
    ("Delhi",     "Mumbai"):     0,
    ("Delhi",     "Chennai"):    1,
    ("Delhi",     "Kolkata"):    2,
    ("Delhi",     "Hyderabad"):  3,
    ("Delhi",     "Bangalore"):  4,
    ("Mumbai",    "Delhi"):      5,
    ("Mumbai",    "Chennai"):    6,
    ("Mumbai",    "Kolkata"):    7,
    ("Mumbai",    "Hyderabad"):  8,
    ("Mumbai",    "Bangalore"):  9,
    ("Chennai",   "Delhi"):     10,
    ("Chennai",   "Mumbai"):    11,
    ("Chennai",   "Kolkata"):   12,
    ("Chennai",   "Hyderabad"): 13,
    ("Chennai",   "Bangalore"): 14,
    ("Kolkata",   "Delhi"):     15,
    ("Kolkata",   "Mumbai"):    16,
    ("Kolkata",   "Chennai"):   17,
    ("Kolkata",   "Hyderabad"): 18,
    ("Kolkata",   "Bangalore"): 19,
    ("Hyderabad", "Delhi"):     20,
    ("Hyderabad", "Mumbai"):    21,
    ("Hyderabad", "Chennai"):   22,
    ("Hyderabad", "Kolkata"):   23,
    ("Hyderabad", "Bangalore"): 24,
}

def encode_route(src, dst):
    return ROUTE_MAP.get((src, dst), 0)

# ── Feature builder ─────────────────────────────────────────────────────────────
def build_features(
    duration_hrs, duration_mins_extra, days_left,
    source, destination, airline,
    dep_time, arr_time, stops, flight_class,
):
    total_mins = duration_hrs * 60 + duration_mins_extra
    route      = encode_route(source, destination)

    a = airline.replace(" ", "_")
    airline_air_india = int(a == "Air_India")
    airline_go_first  = int(a == "GO_FIRST")
    airline_indigo    = int(a == "Indigo")
    airline_spicejet  = int(a == "SpiceJet")
    airline_vistara   = int(a == "Vistara")

    src_chennai   = int(source == "Chennai")
    src_delhi     = int(source == "Delhi")
    src_hyderabad = int(source == "Hyderabad")
    src_kolkata   = int(source == "Kolkata")
    src_mumbai    = int(source == "Mumbai")

    dst_chennai   = int(destination == "Chennai")
    dst_delhi     = int(destination == "Delhi")
    dst_hyderabad = int(destination == "Hyderabad")
    dst_kolkata   = int(destination == "Kolkata")
    dst_mumbai    = int(destination == "Mumbai")

    dt = dep_time.replace(" ", "_")
    dep_early_morning = int(dt == "Early_Morning")
    dep_evening       = int(dt == "Evening")
    dep_late_night    = int(dt == "Late_Night")
    dep_morning       = int(dt == "Morning")
    dep_night         = int(dt == "Night")

    at = arr_time.replace(" ", "_")
    arr_early_morning = int(at == "Early_Morning")
    arr_evening       = int(at == "Evening")
    arr_late_night    = int(at == "Late_Night")
    arr_morning       = int(at == "Morning")
    arr_night         = int(at == "Night")

    stops_two_or_more = int(stops == "Two or more stops (2+)")
    stops_zero        = int(stops == "Non-stop (0)")
    class_economy     = int(flight_class == "Economy")

    features = np.array([[
        duration_hrs,
        days_left,
        total_mins,
        route,
        airline_air_india, airline_go_first, airline_indigo, airline_spicejet, airline_vistara,
        src_chennai, src_delhi, src_hyderabad, src_kolkata, src_mumbai,
        dst_chennai, dst_delhi, dst_hyderabad, dst_kolkata, dst_mumbai,
        dep_early_morning, dep_evening, dep_late_night, dep_morning, dep_night,
        arr_early_morning, arr_evening, arr_late_night, arr_morning, arr_night,
        stops_two_or_more,
        stops_zero,
        class_economy,
    ]])
    return features

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero">
        <h1>Flight Price Predictor</h1>
        <p>AI-powered fare estimation for Indian domestic routes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Row 1 – Route ─────────────────────────────────────────────────────────────
st.caption("ROUTE")
col1, col2, col3, col4 = st.columns(4)
with col1:
    source = st.selectbox("Departure City", CITIES, index=0, key="source")
with col2:
    dest_options = [c for c in CITIES if c != source]
    destination  = st.selectbox("Arrival City", dest_options, key="destination")
with col3:
    dep_time = st.selectbox("Departure Time", TIME_SLOTS, key="dep_time")
with col4:
    arr_time = st.selectbox("Arrival Time", TIME_SLOTS, index=2, key="arr_time")

st.divider()

# ── Row 2 – Flight Details ─────────────────────────────────────────────────────
st.caption("FLIGHT DETAILS")
col5, col6, col7, col8 = st.columns(4)
with col5:
    airline = st.selectbox("Airline", AIRLINES, key="airline")
with col6:
    flight_class = st.selectbox("Class", FLIGHT_CLASS, key="flight_class")
with col7:
    stops = st.selectbox("Number of Stops", STOPS, key="stops")
with col8:
    days_left = st.number_input(
        "Days Until Departure",
        min_value=1, max_value=365, value=30, step=1,
        key="days_left",
    )

st.divider()

# ── Row 3 – Duration ───────────────────────────────────────────────────────────
st.caption("FLIGHT DURATION")
col9, col10, _, __ = st.columns(4)
with col9:
    duration_hrs = st.number_input(
        "Hours", min_value=0, max_value=24, value=2, step=1, key="dur_hrs"
    )
with col10:
    duration_mins_extra = st.number_input(
        "Minutes", min_value=0, max_value=59, value=30, step=5, key="dur_mins"
    )

# ── Validation ─────────────────────────────────────────────────────────────────
same_city = source == destination
zero_dur  = (duration_hrs == 0 and duration_mins_extra == 0)

if same_city:
    st.error("Departure and arrival cities cannot be the same.")
elif zero_dur:
    st.error("Flight duration must be greater than 0 minutes.")

# ── Predict button (centered, full-width inside middle column) ─────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("Predict Fare", disabled=(same_city or zero_dur))

# ── Result ─────────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running model..."):
        features = build_features(
            duration_hrs, duration_mins_extra, days_left,
            source, destination, airline,
            dep_time, arr_time, stops, flight_class,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                price = model.predict(features)[0]

            dur_total_hrs = duration_hrs + duration_mins_extra / 60
            price_per_hr  = price / max(dur_total_hrs, 0.5)
            stop_label    = stops.split("(")[0].strip()

            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="rp-header">
                        <div class="rp-route">{source} &nbsp;&rarr;&nbsp; {destination}</div>
                        <span class="rp-tag">{flight_class}</span>
                    </div>
                    <div class="rp-price-block">
                        <div class="rp-price-label">Estimated Fare</div>
                        <div class="rp-price">&#8377;&nbsp;{price:,.0f}</div>
                        <div class="rp-price-note">Indian Rupees &nbsp;&middot;&nbsp; AI prediction based on historical fares</div>
                    </div>
                    <div class="rp-stats">
                        <div class="rp-stat">
                            <div class="rp-stat-label">Airline</div>
                            <div class="rp-stat-value">{airline}</div>
                        </div>
                        <div class="rp-stat">
                            <div class="rp-stat-label">Stops</div>
                            <div class="rp-stat-value">{stop_label}</div>
                        </div>
                        <div class="rp-stat">
                            <div class="rp-stat-label">Duration</div>
                            <div class="rp-stat-value">{duration_hrs}h {duration_mins_extra}m</div>
                        </div>
                        <div class="rp-stat">
                            <div class="rp-stat-label">Days to Departure</div>
                            <div class="rp-stat-value">{days_left} days</div>
                        </div>
                        <div class="rp-stat">
                            <div class="rp-stat-label">Price / Hour</div>
                            <div class="rp-stat-value">&#8377;&nbsp;{price_per_hr:,.0f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Powered by Random Forest &nbsp;|&nbsp; Indian Domestic Flight Data</div>',
    unsafe_allow_html=True,
)