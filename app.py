import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Smart Fare Predictor", page_icon="🚖", layout="centered")

# ----------------------------
# Custom CSS - Black Background, Visible Labels
# ----------------------------
st.markdown("""
    <style>
    /* ── Global background ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    [data-testid="stHeader"] {
        background-color: #000000 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0d0d0d !important;
    }

    /* ── Main content area ── */
    .block-container {
        background-color: #000000 !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* ── All text white ── */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #ffffff !important;
    }

    /* ── Streamlit labels (slider, number, select) ── */
    .stSlider label, .stNumberInput label, .stSelectbox label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] {
        color: #ffffff !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }

    /* ── Slider track ── */
    .stSlider [data-baseweb="slider"] {
        padding: 0 4px;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #e53935 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 0 3px #e53935 !important;
    }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #aaaaaa !important;
        font-size: 12px !important;
    }

    /* ── Number input ── */
    .stNumberInput input {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 0.5px solid #333333 !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
    }
    .stNumberInput input:focus {
        border-color: #e53935 !important;
        box-shadow: 0 0 0 2px rgba(229,57,53,0.3) !important;
    }
    .stNumberInput button {
        background-color: #1a1a1a !important;
        color: #aaaaaa !important;
        border: 0.5px solid #333333 !important;
        border-radius: 8px !important;
    }
    .stNumberInput button:hover {
        color: #e53935 !important;
        border-color: #e53935 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        border: 0.5px solid #333333 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    .stSelectbox [data-baseweb="select"] span {
        color: #ffffff !important;
        font-size: 15px !important;
    }
    .stSelectbox svg {
        fill: #aaaaaa !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"] {
        background-color: #1a1a1a !important;
        border: 0.5px solid #333333 !important;
        border-radius: 10px !important;
    }
    [data-baseweb="menu"] {
        background-color: #1a1a1a !important;
    }
    [role="option"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    [role="option"]:hover {
        background-color: #2a2a2a !important;
    }

    /* ── Card style ── */
    .fare-card {
        background: #111111;
        border: 0.5px solid #2a2a2a;
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* ── Divider ── */
    hr {
        border-color: #222222 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Predict button ── */
    .stButton > button {
        background: #e53935 !important;
        color: #ffffff !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.3px !important;
        margin-top: 0.5rem !important;
    }
    .stButton > button:hover {
        background: #c62828 !important;
        transform: scale(1.015) !important;
        box-shadow: 0 4px 20px rgba(229,57,53,0.4) !important;
    }
    .stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* ── Result box ── */
    .result-box {
        background: linear-gradient(135deg, #0d47a1, #1565c0);
        border-radius: 14px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-top: 1.25rem;
    }
    .result-label {
        font-size: 12px !important;
        color: rgba(255,255,255,0.65) !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .result-amount {
        font-size: 56px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        line-height: 1;
        margin: 8px 0;
    }
    .result-sub {
        font-size: 12px !important;
        color: rgba(255,255,255,0.55) !important;
        margin-top: 10px;
    }

    /* ── Breakdown table ── */
    .breakdown {
        background: rgba(0,0,0,0.25);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        text-align: left;
    }
    .breakdown-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: rgba(255,255,255,0.75);
        padding: 5px 0;
        border-bottom: 0.5px solid rgba(255,255,255,0.08);
    }
    .breakdown-row:last-child {
        border-bottom: none;
        color: #ffffff !important;
        font-weight: 700;
        font-size: 15px;
        padding-top: 10px;
    }
    .breakdown-row span {
        color: inherit !important;
    }

    /* ── Badges ── */
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 14px;
    }
    .badge {
        background: rgba(255,255,255,0.13);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 12px !important;
        color: rgba(255,255,255,0.85) !important;
    }

    /* ── Hero subtitle ── */
    .hero-sub {
        text-align: center;
        color: #888888 !important;
        font-size: 13px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------------
# Load Model (comment out if testing without model)
# ----------------------------
# model       = joblib.load("models/xgboost_model.pkl")
# transformer = joblib.load("models/transformer.pkl")
# columns     = joblib.load("models/columns.pkl")


# ----------------------------
# Fare calculation (replace with model.predict when ready)
# ----------------------------
def calculate_fare(hour, surge, dist, pax, day, weather, vehicle):
    base_rate   = {'UberX': 30,  'UberXL': 50,  'Uber Black': 80}
    per_km      = {'UberX': 12,  'UberXL': 18,  'Uber Black': 28}
    weather_mul = {'Clear': 1.0, 'Foggy': 1.1,  'Rainy': 1.25, 'Snowy': 1.4}
    peak_hours  = [7, 8, 9, 17, 18, 19, 20]
    is_weekend  = day in ('Saturday', 'Sunday')

    base_fare   = base_rate[vehicle]
    dist_fare   = dist * per_km[vehicle]
    surge_adj   = (base_fare + dist_fare) * (surge - 1)
    weather_adj = (base_fare + dist_fare + surge_adj) * (weather_mul[weather] - 1)
    peak_adj    = (base_fare + dist_fare) * 0.2 if hour in peak_hours else 0
    weekend_adj = (base_fare + dist_fare) * 0.1 if is_weekend else 0
    pax_adj     = max(0, pax - 2) * 8

    total = base_fare + dist_fare + surge_adj + weather_adj + peak_adj + weekend_adj + pax_adj
    return round(total, 2), {
        'base_fare':    base_fare,
        'dist_fare':    dist_fare,
        'surge_adj':    surge_adj,
        'weather_adj':  weather_adj,
        'peak_adj':     peak_adj,
        'weekend_adj':  weekend_adj,
        'pax_adj':      pax_adj,
    }


# ----------------------------
# Title
# ----------------------------
st.markdown("<h1 style='text-align:center;font-size:30px;font-weight:700;'>🚖 Smart Fare Predictor</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>AI-powered ride price estimation ⚡</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------
# Row 1 — Hour | Surge
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='fare-card'>", unsafe_allow_html=True)
    hour = st.slider("🕒 Pickup Hour", min_value=0, max_value=23, value=8)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='fare-card'>", unsafe_allow_html=True)
    surge = st.number_input("⚡ Surge Multiplier", min_value=1.0, max_value=3.0,
                            value=1.0, step=0.25, format="%.2f")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Row 2 — Distance | Day
# ----------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown("<div class='fare-card'>", unsafe_allow_html=True)
    dist = st.number_input("📍 Distance (km)", min_value=0.5, max_value=50.0,
                           value=5.0, step=0.5, format="%.2f")
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='fare-card'>", unsafe_allow_html=True)
    day = st.selectbox("📅 Day", [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ])
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Row 3 — Passengers | Weather
# ----------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown("<div class='fare-card'>", unsafe_allow_html=True)
    pax = st.number_input("👥 Passengers", min_value=1, max_value=6, value=2, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

with col6:
    st.markdown("<div class='fare-card'>", unsafe_allow_html=True)
    weather = st.selectbox("🌦️ Weather", ["Clear", "Rainy", "Foggy", "Snowy"])
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Row 4 — Vehicle (full width)
# ----------------------------
st.markdown("<div class='fare-card'>", unsafe_allow_html=True)
vehicle = st.selectbox("🚗 Vehicle Type", ["UberX", "UberXL", "Uber Black"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🚀 Predict Fare"):

    # ── If using real model, replace calculate_fare with: ──
    # input_data = {
    #     "pickup_hour": hour, "day_of_week": day,
    #     "weather_condition": weather, "vehicle_type": vehicle,
    #     "distance_km": dist, "passenger_count": pax,
    #     "surge_multiplier": surge
    # }
    # df = pd.DataFrame([input_data])
    # df_transformed = pd.DataFrame(transformer.transform(df), columns=columns)
    # total = round(model.predict(df_transformed)[0], 2)
    # breakdown = {}

    total, breakdown = calculate_fare(hour, surge, dist, pax, day, weather, vehicle)

    # ── Badges ──
    badges = []
    if hour in [7, 8, 9, 17, 18, 19, 20]:
        badges.append("Peak hours")
    if surge > 1:
        badges.append(f"Surge ×{surge:.2f}")
    if weather != "Clear":
        badges.append(f"{weather} weather")
    if day in ("Saturday", "Sunday"):
        badges.append("Weekend")
    if pax > 2:
        badges.append(f"{pax} passengers")

    badge_html = "".join([f"<span class='badge'>{b}</span>" for b in badges])

    # ── Breakdown rows ──
    rows = [
        (f"Base fare ({vehicle})",          f"₹{breakdown['base_fare']:.2f}"),
        (f"Distance ({dist:.2f} km)",        f"₹{breakdown['dist_fare']:.2f}"),
    ]
    if breakdown['surge_adj']   > 0: rows.append(("Surge multiplier",   f"₹{breakdown['surge_adj']:.2f}"))
    if breakdown['weather_adj'] > 0: rows.append((f"{weather} weather", f"₹{breakdown['weather_adj']:.2f}"))
    if breakdown['peak_adj']    > 0: rows.append(("Peak hour premium",  f"₹{breakdown['peak_adj']:.2f}"))
    if breakdown['weekend_adj'] > 0: rows.append(("Weekend pricing",    f"₹{breakdown['weekend_adj']:.2f}"))
    if breakdown['pax_adj']     > 0: rows.append(("Extra passengers",   f"₹{breakdown['pax_adj']:.2f}"))
    rows.append(("Total", f"₹{total:.2f}"))

    breakdown_html = "".join([
        f"<div class='breakdown-row'><span>{r[0]}</span><span>{r[1]}</span></div>"
        for r in rows
    ])

    st.markdown(f"""
        <div class='result-box'>
            <p class='result-label'>Estimated Fare</p>
            <p class='result-amount'>₹{total:.2f}</p>
            <p class='result-sub'>Approximate price based on current conditions</p>
            <div class='badge-row'>{badge_html}</div>
            <div class='breakdown'>{breakdown_html}</div>
        </div>
    """, unsafe_allow_html=True)