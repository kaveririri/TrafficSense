
import json
import html as html_lib
from datetime import datetime, time

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_DATASET,
    MODEL_PATH,
    METRICS_PATH,
    METADATA_PATH,
)
from src.data_loader import load_dataset, inspect_dataset
from src.feature_engineering import clean_raw_data
from src.prediction import predict_traffic
from src.scenario_analysis import compare_scenarios
from src.model_training import train_and_evaluate
from src.visualization import (
    traffic_trend_chart,
    weather_traffic_chart,
    scenario_comparison_chart,
    actual_vs_predicted_chart,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TrafficSense 2.0",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080F1B;
        --surface: #111E30;
        --surface-hover: #17283D;
        --border: #20344B;
        --text: #F4F8FC;
        --muted: #A8B8CA;
        --teal: #32D6C7;
        --teal-hover: #55E6D6;
    }

    /* GLOBAL */
    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text);
        letter-spacing: -0.5px;
    }

    p, li, label, input, button {
        font-family: 'Inter', sans-serif;
    }

    /* HEADER */
    [data-testid="stHeader"] {
        background: rgba(8, 15, 27, 0.96);
    }

    /* MAIN CONTENT SPACING */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* METRIC CARDS */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 20px;
        border-radius: 16px;
        transition: border-color 0.2s ease,
                    transform 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(50, 214, 199, 0.5);
        transform: translateY(-2px);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--teal);
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }

    /* HERO */
    .hero {
        background: linear-gradient(
            120deg,
            #10243B 0%,
            #123B4A 55%,
            #10243B 100%
        );
        border: 1px solid #245267;
        border-radius: 20px;
        padding: 42px;
        margin: 10px 0 30px 0;
    }

    .hero-label, .section-label {
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #FFFFFF;
        font-family: 'Outfit', sans-serif;
        font-size: clamp(36px, 5vw, 58px);
        font-weight: 700;
        line-height: 1.08;
        letter-spacing: -1.5px;
        margin: 16px 0;
    }

    .hero-description {
        color: #B8C9DC;
        font-size: 15px;
        line-height: 1.8;
        max-width: 650px;
    }

    /* INFORMATION CARDS */
    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        min-height: 140px;
        transition: all 0.2s ease;
    }

    .info-card:hover {
        background: var(--surface-hover);
        border-color: rgba(50, 214, 199, 0.5);
        transform: translateY(-3px);
    }

    .info-card h3 {
        color: var(--teal);
        font-family: 'Outfit', sans-serif;
        margin-bottom: 10px;
    }

    .info-card p {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
    }

    /* BUTTONS */
    .stButton > button {
        background: var(--teal);
        color: #07131D;
        border: 1px solid var(--teal);
        border-radius: 10px;
        font-weight: 600;
        min-height: 44px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: var(--teal-hover);
        border-color: var(--teal-hover);
        color: #07131D;
        transform: translateY(-1px);
    }

    /* INPUTS AND SELECT BOXES */
    [data-baseweb="input"],
    [data-baseweb="select"] > div,
    [data-baseweb="textarea"] {
        border-radius: 10px;
    }

    /* SLIDERS */
    [data-baseweb="slider"] [role="slider"] {
        background: var(--teal);
        border-color: var(--teal);
    }

    [data-baseweb="slider"] [data-testid="stTickBar"] {
        background: var(--border);
    }

    /* TABS */
    div[data-baseweb="tab-list"] {
        gap: 12px;
    }

    /* DIVIDERS */
    hr {
        border-color: rgba(168, 184, 202, 0.14);
    }

    /* DATA TABLES */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* FOOTER */
    footer {
        visibility: hidden;
    }

    /* REDUCED MOTION */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
        }
    }
        /* --------------------------------
       PAGE LOAD ANIMATIONS
    -------------------------------- */

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes chartGrow {
        0% {
            opacity: 0;
            transform: scale(0.94);
        }
        70% {
            opacity: 1;
            transform: scale(1.015);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }

    /* Main page content */
    [data-testid="stMainBlockContainer"] {
        animation: fadeUp 0.65s ease-out both;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        animation: fadeUp 0.7s ease-out both;
    }

    /* Charts grow into view when first rendered */
    [data-testid="stPlotlyChart"] {
        width: 100%;
        border-radius: 14px;
        overflow: hidden;
        transform-origin: center;
        animation: chartGrow 0.9s cubic-bezier(.2,.8,.2,1) both;
    }

    /* Stagger chart entrance */
    [data-testid="stPlotlyChart"]:nth-of-type(2) {
        animation-delay: 0.15s;
    }

    /* Smooth navigation appearance */
    [data-testid="stNavigation"] {
        background: #080F1B !important;
        animation: fadeUp 0.5s ease-out both;
    }

    /* Keep navigation above page content */
    [data-testid="stHeader"] {
        background: #080F1B;
    }

    /* Respect reduced-motion preferences */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
        /* --------------------------------
       PAGE LOAD ANIMATIONS
    -------------------------------- */

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes chartGrow {
        0% {
            opacity: 0;
            transform: scale(0.94);
        }
        70% {
            opacity: 1;
            transform: scale(1.015);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }

    /* Main page content */
    [data-testid="stMainBlockContainer"] {
        animation: fadeUp 0.65s ease-out both;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        animation: fadeUp 0.7s ease-out both;
    }

    /* Charts grow into view when first rendered */
    [data-testid="stPlotlyChart"] {
        width: 100%;
        border-radius: 14px;
        overflow: hidden;
        transform-origin: center;
        animation: chartGrow 0.9s cubic-bezier(.2,.8,.2,1) both;
    }

    /* Stagger chart entrance */
    [data-testid="stPlotlyChart"]:nth-of-type(2) {
        animation-delay: 0.15s;
    }

    /* Smooth navigation appearance */
    [data-testid="stNavigation"] {
        background: #080F1B !important;
        animation: fadeUp 0.5s ease-out both;
    }

    /* Keep navigation above page content */
    [data-testid="stHeader"] {
        background: #080F1B;
    }

    /* Respect reduced-motion preferences */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }

    /* =====================================
    TRAFFICSENSE — CINEMATIC MOTION
    ===================================== */

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes chartReveal {
        from {
            opacity: 0;
            clip-path: inset(0 100% 0 0);
        }
        to {
            opacity: 1;
            clip-path: inset(0 0 0 0);
        }
    }

    @keyframes barReveal {
        from {
            opacity: 0;
            transform: scaleY(0.05);
            transform-origin: bottom;
        }
        to {
            opacity: 1;
            transform: scaleY(1);
            transform-origin: bottom;
        }
    }

    /* Page entrance */
    [data-testid="stMainBlockContainer"] {
        animation: fadeUp 0.65s ease-out both;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        animation: fadeUp 0.7s ease-out both;
        transition: border-color 0.3s ease,
                    box-shadow 0.3s ease,
                    transform 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(50, 214, 199, 0.65);
        box-shadow: 0 8px 28px rgba(50, 214, 199, 0.08);
        transform: translateY(-3px);
    }

    /* Actual Plotly line paths */
    .js-plotly-plot .scatterlayer .js-line {
        animation: chartReveal 1.2s ease-out both;
    }

    /* Actual Plotly bars */
    .js-plotly-plot .barlayer .point path {
        animation: barReveal 0.85s ease-out both;
    }

    /* Chart container */
    [data-testid="stPlotlyChart"] {
        width: 100%;
        border-radius: 14px;
        overflow: hidden;
    }

    /* Section entrance */
    [data-testid="stSubheader"],
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        animation: fadeUp 0.6s ease-out both;
    }

    /* Navigation */
    [data-testid="stNavigation"] {
        background: #080F1B !important;
    }

    /* Reduced motion accessibility */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }



</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA AND MODEL LOADING
# --------------------------------------------------

@st.cache_data
def get_raw_data():
    return load_dataset(DEFAULT_DATASET)


@st.cache_data
def get_clean_data():
    return clean_raw_data(get_raw_data())


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def get_metrics():
    if not METRICS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_json(METRICS_PATH)
    except Exception:
        return pd.DataFrame()


def get_metadata():
    if not METADATA_PATH.exists():
        return {}

    try:
        return json.loads(METADATA_PATH.read_text())
    except Exception:
        return {}


def require_data():
    try:
        return get_raw_data(), get_clean_data()
    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        st.info(
            "Check that Metro_Interstate_Traffic_Volume.csv "
            "is inside the data folder."
        )
        return None, None


def require_model():
    try:
        model = get_model()
        if model is None:
            st.warning(
                "The trained model was not found. "
                "Open Model Lab and train the models first."
            )
        return model
    except Exception as error:
        st.error(f"Unable to load the trained model: {error}")
        return None



def get_holiday_for_date(cleaned, selected_date):
    """Return the holiday label recorded for this exact dataset date."""
    if "date_time" not in cleaned.columns or "holiday" not in cleaned.columns:
        return "None"

    dates = pd.to_datetime(cleaned["date_time"], errors="coerce").dt.date
    day_rows = cleaned.loc[dates == selected_date, "holiday"].dropna()
    valid = day_rows.astype(str).str.strip()
    valid = valid[~valid.str.lower().isin(["", "none", "nan", "null"])]

    if valid.empty:
        return "None"

    return valid.value_counts().index[0]


def page_heading(label, heading, description):
    st.markdown(f'<div class="section-label">{label}</div>',
                unsafe_allow_html=True)
    st.title(heading)
    st.caption(description)


# --------------------------------------------------
# HOME
# --------------------------------------------------



def home_page():
    raw, cleaned = require_data()
    if raw is None:
        return

    metadata = get_metadata()
    metrics = get_metrics()

    best_r2 = None
    if not metrics.empty and "R2" in metrics.columns:
        best_r2 = metrics["R2"].max()

    # ------------------------------------------
    # HERO SECTION: TEXT LEFT + ANIMATION RIGHT
    # ------------------------------------------

    left, right = st.columns([1, 1.15], gap="large")


    with left:
        st.title("TRAFFICSENSE")
        st.subheader("See beyond the traffic.")
        st.write("Predict patterns. Explore possibilities.")


    with right:
        st.components.v1.html(
            """
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                body {
                    margin:0;
                    background:transparent;
                    overflow:hidden;
                }

                .scene {
                width: 100%;
                height: 380px;
                animation: cityEntrance 1.4s ease-out both;
                }

                @keyframes cityEntrance {
                    0% {
                        opacity: 0;
                        transform: translateY(35px) scale(0.90);
                        filter: blur(8px);
                    }

                    60% {
                        opacity: 1;
                        filter: blur(0);
                    }

                    100% {
                        opacity: 1;
                        transform: translateY(0) scale(1);
                        filter: blur(0);
                    }
                }

                .windows {
                    animation: windowGlow 2s ease-in-out infinite alternate;
                }

                @keyframes windowGlow {
                    from { opacity: 0.45; }
                    to { opacity: 1; }
                }

            .lights {
                animation: move 1.5s linear infinite;
            }

            .nodes {
                animation: pulse 1.5s ease-in-out infinite alternate;
            }
                .lights {
                    animation: move 2s linear infinite;
                }

                .windows {
                    animation: glow 2s ease-in-out infinite alternate;
                }

                .nodes {
                    animation: pulse 1.8s ease-in-out infinite alternate;
                }

                @keyframes move {
                    to { stroke-dashoffset:-100; }
                }

                @keyframes glow {
                    to { opacity:.35; }
                }

                @keyframes pulse {
                    to { opacity:.25; }
                }

                @media(prefers-reduced-motion:reduce) {
                    * { animation:none !important; }
                }
            </style>
            </head>

            <body>

            <svg class="scene" viewBox="0 0 560 330"
                 xmlns="http://www.w3.org/2000/svg">

                <defs>
                    <linearGradient id="road"
                        x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#153C50"/>
                        <stop offset="100%" stop-color="#071522"/>
                    </linearGradient>

                    <filter id="glow">
                        <feGaussianBlur stdDeviation="4" result="b"/>
                        <feMerge>
                            <feMergeNode in="b"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>

                <!-- City buildings -->
                <g fill="#183B50" stroke="#286078">
                    <rect x="20" y="100" width="65" height="135"/>
                    <rect x="95" y="65" width="75" height="170"/>
                    <rect x="180" y="120" width="60" height="115"/>
                    <rect x="320" y="105" width="70" height="130"/>
                    <rect x="400" y="55" width="75" height="180"/>
                    <rect x="485" y="115" width="55" height="120"/>
                </g>

                <!-- Illuminated windows -->
                <g fill="#43DCCB" class="windows">
                    <path d="
                        M35 120h8v8h-8z
                        M55 120h8v8h-8z
                        M35 145h8v8h-8z
                        M55 145h8v8h-8z
                        M110 85h8v9h-8z
                        M135 85h8v9h-8z
                        M420 80h8v9h-8z
                        M445 80h8v9h-8z
                    "/>
                </g>

                <!-- Futuristic highway -->
                <path
                    d="M170 330 L250 220 L310 220 L420 330 Z"
                    fill="url(#road)"
                    stroke="#2A6475"
                    stroke-width="2"
                />

                <path
                    d="M280 235 L295 330"
                    stroke="#43DCCB"
                    stroke-width="2"
                    stroke-dasharray="9 12"
                />

                <!-- Moving traffic light trails -->
                <g
                    fill="none"
                    stroke="#42E6D4"
                    stroke-width="4"
                    stroke-dasharray="15 16"
                    filter="url(#glow)"
                >
                    <path class="lights" d="M190 320 L258 235"/>
                    <path class="lights" d="M390 320 L305 235"/>
                </g>

                <!-- Connected city network -->
                <g fill="none" stroke="#40DCCB" stroke-width="1.5">
                    <path d="
                        M55 90
                        L130 45
                        L210 90
                        L280 35
                        L400 75
                        L490 35
                    "/>
                </g>

                <!-- Glowing network nodes -->
                <g fill="#67F5E5" class="nodes" filter="url(#glow)">
                    <circle cx="55" cy="90" r="4"/>
                    <circle cx="130" cy="45" r="4"/>
                    <circle cx="210" cy="90" r="4"/>
                    <circle cx="280" cy="35" r="5"/>
                    <circle cx="400" cy="75" r="4"/>
                    <circle cx="490" cy="35" r="4"/>
                </g>

                <!-- Traffic intelligence panel -->
                <g transform="translate(355 155)">
                    <rect
                        width="155"
                        height="58"
                        rx="9"
                        fill="#102A3B"
                        stroke="#35CFC0"
                    />

                    <text
                        x="12"
                        y="19"
                        fill="#AFC9D9"
                        font-size="9"
                    >
                        TRAFFIC INTELLIGENCE
                    </text>

                    <path
                        d="M12 43 L35 33 L55 38 L78 25 L100 31 L135 18"
                        fill="none"
                        stroke="#43E6D4"
                        stroke-width="2.5"
                    />
                </g>

            </svg>
            </body>
            </html>
            """,
            height=400,
            scrolling=False
        )

    # ------------------------------------------
    # PLATFORM OVERVIEW
    # ------------------------------------------

    st.markdown("### Platform Overview")

    row1 = st.columns(2, gap="medium")
    row2 = st.columns(2, gap="medium")

    row1[0].metric(
        "Historical Records",
        f"{len(cleaned):,}"
    )

    row1[1].metric(
        "Average Traffic",
        f"{cleaned['traffic_volume'].mean():,.0f}",
        help="Average vehicles per hour in the historical dataset."
    )

    row2[0].metric(
        "Peak Traffic",
        f"{cleaned['traffic_volume'].max():,}",
        help="Maximum recorded hourly traffic volume."
    )

    row2[1].metric(
        "Model R²",
        f"{best_r2:.3f}" if best_r2 is not None else "Not trained"
    )

    st.divider()

    # ------------------------------------------
    # TRAFFIC CHARTS
    # ------------------------------------------

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Traffic Patterns")
        st.plotly_chart(
            traffic_trend_chart(cleaned),
            use_container_width=True
        )

    with right:
        st.subheader("Weather Insights")
        st.plotly_chart(
            weather_traffic_chart(cleaned),
            use_container_width=True
        )

    # ------------------------------------------
    # I-94 TRAFFIC CORRIDOR — CINEMATIC MAP
    # ------------------------------------------

    st.divider()
    st.subheader("The I-94 Traffic Corridor")

    st.caption(
        "Illustrative corridor visualization | Minneapolis ↔ St. Paul | "
        "Historical traffic context, not live traffic data."
    )
        # Historical traffic intensity classification
    traffic = cleaned["traffic_volume"].dropna()

    low_limit = traffic.quantile(1 / 3)
    high_limit = traffic.quantile(2 / 3)

    low_count = int((traffic < low_limit).sum())

    moderate_count = int(
        ((traffic >= low_limit) & (traffic < high_limit)).sum()
    )

    high_count = int((traffic >= high_limit).sum())

    low_percent = low_count / len(traffic) * 100
    moderate_percent = moderate_count / len(traffic) * 100
    high_percent = high_count / len(traffic) * 100

    average_traffic = traffic.mean()
    peak_traffic = traffic.max()
    median_traffic = traffic.median()

    st.components.v1.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            * { box-sizing: border-box; }

            body {
                margin: 0;
                background: transparent;
                font-family: Inter, Arial, sans-serif;
                color: #F4F8FC;
            }

            .map {
                position: relative;
                height: 490px;
                overflow: hidden;
                border: 1px solid #176477;
                border-radius: 18px;
                background:
                    radial-gradient(ellipse at 50% 50%,
                    #0c2939 0%, #071522 75%);
            }

            .map-grid {
                position: absolute;
                inset: 0;
                opacity: .4;
                background-image:
                    linear-gradient(#16445b 1px, transparent 1px),
                    linear-gradient(90deg, #16445b 1px, transparent 1px);
                background-size: 28px 28px;
            }

            .roads {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
            }

            .main-road {
                fill: none;
                stroke-width: 12;
                stroke: #123d4d;
                stroke-linecap: round;
            }

            .traffic-road {
                fill: none;
                stroke-width: 5;
                stroke-linecap: round;
                stroke-dasharray: 13 8;
                animation: flow 2s linear infinite;
                filter: drop-shadow(0 0 7px currentColor);
            }

            .low { stroke: #20E89A; color: #20E89A; }
            .medium { stroke: #FFD34E; color: #FFD34E; }
            .high { stroke: #FF5268; color: #FF5268; }

            @keyframes flow {
                to { stroke-dashoffset: -42; }
            }

            .marker {
                position: absolute;
                padding: 10px 14px;
                border: 1px solid #24758A;
                border-radius: 10px;
                background: rgba(5, 23, 37, .95);
                font-weight: 600;
                box-shadow: 0 0 20px rgba(35, 211, 204, .13);
            }

            .minneapolis { left: 5%; top: 38%; }
            .stpaul { right: 5%; top: 62%; }

            .pin {
                display: inline-block;
                width: 9px;
                height: 9px;
                margin-right: 8px;
                border-radius: 50%;
                background: #32D6C7;
                box-shadow: 0 0 12px #32D6C7;
                animation: pulse 1.8s infinite alternate;
            }

            @keyframes pulse {
                to { box-shadow: 0 0 22px 6px rgba(50,214,199,.35); }
            }

            .legend {
                position: absolute;
                top: 18px;
                right: 18px;
                padding: 14px 17px;
                border: 1px solid #22536A;
                border-radius: 12px;
                background: rgba(5, 20, 33, .94);
                font-size: 12px;
                line-height: 2;
            }

            .legend-title {
                font-weight: 700;
                margin-bottom: 5px;
            }

            .dot {
                display: inline-block;
                width: 22px;
                height: 4px;
                margin-right: 8px;
                vertical-align: middle;
                border-radius: 5px;
            }

            .low-dot { background: #20E89A; }
            .medium-dot { background: #FFD34E; }
            .high-dot { background: #FF5268; }

            .landmark {
                position: absolute;
                color: #75C9E4;
                font-size: 12px;
                text-align: center;
                opacity: .9;
            }

            .como { left: 36%; top: 25%; }
            .university { left: 30%; top: 52%; }
            .falcon { right: 32%; top: 34%; }

            .river {
                position: absolute;
                left: 40%;
                bottom: 18%;
                color: #58A8CB;
                font-size: 12px;
                font-style: italic;
            }

            .route-label {
                position: absolute;
                left: 51%;
                top: 57%;
                transform: translate(-50%, -50%);
                padding: 7px 13px;
                border-radius: 8px;
                background: #174E9A;
                border: 1px solid #80B9FF;
                font-weight: 700;
            }

            .map-footer {
                position: absolute;
                left: 18px;
                bottom: 18px;
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                padding: 12px 15px;
                background: rgba(5, 20, 33, .94);
                border: 1px solid #22536A;
                border-radius: 10px;
                font-size: 12px;
            }

            @media(max-width: 600px) {
                .map { height: 390px; }
                .legend { top: 10px; right: 10px; }
                .landmark { font-size: 10px; }
                .map-footer { gap: 8px; }
            }

            @media(prefers-reduced-motion: reduce) {
                * { animation: none !important; }
            }

            .road-label {
                position: absolute;
                padding: 5px 9px;
                border: 1px solid rgba(50, 214, 199, .35);
                border-radius: 6px;
                background: rgba(8, 25, 39, .92);
                color: #A8C8D8;
                font-size: 10px;
                letter-spacing: .4px;
            }

            .road-one {
                left: 19%;
                top: 61%;
            }

            .road-two {
                left: 46%;
                top: 39%;
            }

            .road-three {
                right: 15%;
                top: 72%;
            }

            .road-four {
                left: 34%;
                top: 72%;
            }
        </style>
        </head>

        <body>
        <div class="map">
            <div class="map-grid"></div>

            <svg class="roads" viewBox="0 0 1000 490"
                 preserveAspectRatio="none">

                <!-- Secondary road network -->
                <g fill="none" stroke="#14516A"
                   stroke-width="2" opacity=".7">
                    <path d="M0 80 L1000 110"/>
                    <path d="M0 160 L1000 180"/>
                    <path d="M0 290 L1000 250"/>
                    <path d="M0 400 L1000 370"/>
                    <path d="M130 0 L150 490"/>
                    <path d="M300 0 L320 490"/>
                    <path d="M510 0 L490 490"/>
                    <path d="M720 0 L700 490"/>
                    <path d="M900 0 L870 490"/>
                </g>

                <!-- Mississippi River -->
                <path d="M360 0 C300 100 450 130 380 230
                         S480 360 410 490"
                      fill="none" stroke="#14537A"
                      stroke-width="34" opacity=".8"/>

                <!-- I-94 corridor -->
                <path d="M80 220 C180 260 220 160 330 245
                         S470 215 560 260
                         S700 320 790 265 S900 340 950 335"
                      class="main-road"/>

                <!-- Traffic intensity segments -->
                <path d="M80 220 C180 260 220 160 330 245"
                      class="traffic-road low"/>

                <path d="M330 245 C400 270 470 215 560 260"
                      class="traffic-road medium"/>

                <path d="M560 260 C640 290 700 320 790 265"
                      class="traffic-road high"/>

                <path d="M790 265 C870 250 900 340 950 335"
                      class="traffic-road low"/>

                <!-- Direction arrows -->
                <g fill="#43E6D4" font-size="23">
                    <text x="220" y="205">››</text>
                    <text x="460" y="220">››</text>
                    <text x="680" y="292">››</text>
                    <text x="865" y="300">››</text>
                </g>
            </svg>

            <div class="marker minneapolis">
                <span class="pin"></span>Minneapolis
            </div>

            <div class="marker stpaul">
                <span class="pin"></span>St. Paul
            </div>

            <div class="landmark como">Como Park</div>
            <div class="landmark university">University of Minnesota</div>
            <div class="landmark falcon">Falcon Heights</div>
            <div class="river">Mississippi River</div>

            <!-- Secondary roads and junctions -->
            <div class="road-label road-one">I-35W / I-94</div>
            <div class="road-label road-two">MN-280</div>
            <div class="road-label road-three">I-35E / I-94</div>
            <div class="road-label road-four">University Avenue</div>

            <div class="route-label">I-94</div>

            <div class="legend">
                <div class="legend-title">TRAFFIC INTENSITY</div>
                <div><span class="dot low-dot"></span>Low</div>
                <div><span class="dot medium-dot"></span>Moderate</div>
                <div><span class="dot high-dot"></span>High</div>
            </div>

            <div class="map-footer">
                <span>↔ Minneapolis → St. Paul</span>
                <span>◎ I-94 Corridor</span>
                <span>Historical context</span>
            </div>
        </div>
        </body>
        </html>
        """,
        height=510,
        scrolling=False
    )

        # ------------------------------------------
    # HISTORICAL TRAFFIC INTELLIGENCE
    # ------------------------------------------

    st.markdown("#### Historical Traffic Intelligence")

    st.caption(
        f"Traffic observations classified using historical "
        f"volume thresholds: Low below {low_limit:,.0f}, "
        f"Moderate from {low_limit:,.0f} to {high_limit:,.0f}, "
        f"High above {high_limit:,.0f} vehicles/hour."
    )

    # Traffic intensity cards
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Low Traffic Records",
        f"{low_count:,}",
        f"{low_percent:.1f}% of observations"
    )

    c2.metric(
        "Moderate Traffic Records",
        f"{moderate_count:,}",
        f"{moderate_percent:.1f}% of observations"
    )

    c3.metric(
        "High Traffic Records",
        f"{high_count:,}",
        f"{high_percent:.1f}% of observations"
    )


    st.divider()

    st.markdown("#### Corridor Statistics")

    info1, info2, info3, info4 = st.columns(4)

    info1.metric(
        "Average Traffic",
        f"{average_traffic:,.0f}",
        "Vehicles/hour"
    )

    info2.metric(
        "Peak Traffic",
        f"{peak_traffic:,.0f}",
        "Vehicles/hour"
    )

    info3.metric(
        "Median Traffic",
        f"{median_traffic:,.0f}",
        "Vehicles/hour"
    )

    info4.metric(
        "Historical Records",
        f"{len(traffic):,}"
    )

    st.caption(
        "These statistics are calculated from the cleaned "
        "Metro Interstate Traffic Volume dataset. "
        "They describe historical observations, not live "
        "traffic conditions or individual road segments."
    )
    st.markdown("#### Historical Traffic Distribution")

    traffic_distribution = pd.DataFrame({
        "Traffic Intensity": [
            "Low",
            "Moderate",
            "High"
        ],
        "Records": [
            low_count,
            moderate_count,
            high_count
        ]
    })

    intensity_fig = px.bar(
        traffic_distribution,
        x="Traffic Intensity",
        y="Records",
        color="Traffic Intensity",
        color_discrete_map={
            "Low": "#20E89A",
            "Moderate": "#FFD34E",
            "High": "#FF5268"
        },
        text="Records"
    )

    intensity_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080F1B",
        plot_bgcolor="#111E30",
        font=dict(
            family="Inter",
            color="#A8B8CA"
        ),
        showlegend=False,
        height=320,
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis_title="Traffic Intensity",
        yaxis_title="Number of Historical Records"
    )

    intensity_fig.update_traces(
        textposition="outside",
        marker_line_width=0
    )

    st.plotly_chart(
        intensity_fig,
        use_container_width=True
    )
    st.caption(
        "The map is a conceptual corridor illustration. "
        "Its traffic colors and landmark positions are not "
        "live measurements or geographically verified traffic conditions."
    )


    # ------------------------------------------
    # EXPLORE FEATURES
    # ------------------------------------------

    st.divider()
    st.subheader("Explore TrafficSense")

    cards = st.columns(3)

    features = [
        ("Analytics", "Explore historical traffic patterns."),
        ("Predictor", "Estimate hourly traffic volume."),
        ("Simulator", "Explore different traffic scenarios."),
    ]

    for col, (title, description) in zip(cards, features):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        f"Best model: {metadata.get('best_model', 'Not available')}"
    )

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

def analytics_page():
    raw, cleaned = require_data()
    if raw is None:
        return

    page_heading(
        "DATA INTELLIGENCE",
        "Traffic Analytics",
        "Explore historical traffic behaviour and environmental patterns.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(cleaned):,}")
    c2.metric(
        "Average Volume",
        f"{cleaned['traffic_volume'].mean():,.0f}",
    )
    c3.metric(
        "Maximum Volume",
        f"{cleaned['traffic_volume'].max():,}",
    )

    st.divider()

    st.subheader("Traffic Volume Over Time")

    daily = (
        cleaned.set_index("date_time")["traffic_volume"]
        .resample("D")
        .mean()
        .dropna()
        .reset_index()
    )

    fig = px.line(
        daily,
        x="date_time",
        y="traffic_volume",
        title="Daily Average Traffic Volume",
        labels={
            "date_time": "Date",
            "traffic_volume": "Vehicles per Hour",
        },
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Hourly Traffic")
        st.plotly_chart(
            traffic_trend_chart(cleaned),
            use_container_width=True,
        )

    with right:
        st.subheader("Weather and Traffic")
        st.plotly_chart(
            weather_traffic_chart(cleaned),
            use_container_width=True,
        )

    st.subheader("Traffic by Day of the Week")

    weekday = (
        cleaned.groupby("day_of_week", as_index=False)
        ["traffic_volume"].mean()
    )

    weekday_names = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday",
        5: "Saturday", 6: "Sunday",
    }

    weekday["Day"] = weekday["day_of_week"].map(weekday_names)

    fig = px.bar(
        weekday,
        x="Day",
        y="traffic_volume",
        title="Average Traffic by Weekday",
        labels={"traffic_volume": "Vehicles per Hour"},
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View dataset sample"):
        st.dataframe(cleaned.head(100), use_container_width=True)




# --------------------------------------------------
# REUSABLE WEATHER WALK — CUSTOM SVG, NO IMAGE ASSETS
# --------------------------------------------------

def render_weather_scene(weather, rain=0.0, snow=0.0, clouds=0,
                         selected_time=None, label="WEATHER EXPERIENCE"):
    """Render a theme-matched animated weather scene from the current inputs."""
    condition = str(weather or "Clear")
    condition_lower = condition.lower()
    rain_value = float(rain or 0)
    snow_value = float(snow or 0)
    cloud_value = int(clouds or 0)

    if snow_value > 0 or "snow" in condition_lower:
        mode = "snow"
        description = "Snowfall • winter walking conditions"
    elif rain_value > 0 or any(word in condition_lower for word in ("rain", "drizzle", "thunder")):
        mode = "rain"
        description = "Rainfall • wet road surface"
    elif any(word in condition_lower for word in ("mist", "fog", "haze")):
        mode = "fog"
        description = "Reduced visibility • mist and fog"
    elif cloud_value >= 65 or "cloud" in condition_lower or "overcast" in condition_lower:
        mode = "clouds"
        description = "Cloud cover • diffused daylight"
    else:
        mode = "clear"
        description = "Clear conditions • bright visibility"

    hour = selected_time.hour if hasattr(selected_time, "hour") else 12
    if hour < 6 or hour >= 19:
        time_label = "NIGHT"
        sky_top, sky_bottom = "#081522", "#102B3A"
        ambient = "#173B50"
    elif hour < 8 or hour >= 17:
        time_label = "DUSK / DAWN"
        sky_top, sky_bottom = "#14283B", "#24485A"
        ambient = "#24485A"
    else:
        time_label = "DAYLIGHT"
        sky_top, sky_bottom = "#102B40", "#16384A"
        ambient = "#1C5262"

    condition_label = html_lib.escape(condition.upper())
    label_text = html_lib.escape(str(label).upper())
    desc_text = html_lib.escape(description)
    rain_text = f"{rain_value:g} mm/h"
    snow_text = f"{snow_value:g} mm/h"

    weather_effects = {
        "rain": """
          <g class="rain-layer" stroke="#65CFE8" stroke-width="2" opacity=".8">
            <path d="M70 12l-9 22 M125 48l-9 22 M195 8l-9 22 M265 42l-9 22 M345 15l-9 22 M425 48l-9 22 M500 10l-9 22 M580 42l-9 22 M650 12l-9 22 M720 48l-9 22 M790 15l-9 22"/>
            <path d="M95 92l-9 22 M170 115l-9 22 M245 88l-9 22 M325 112l-9 22 M405 90l-9 22 M485 118l-9 22 M565 88l-9 22 M645 115l-9 22 M725 92l-9 22"/>
          </g>""",
        "snow": """
          <g class="snow-layer" fill="#C7F1F5" opacity=".9">
            <circle cx="75" cy="18" r="3"/><circle cx="150" cy="55" r="2.5"/><circle cx="230" cy="20" r="3"/>
            <circle cx="310" cy="65" r="2.5"/><circle cx="390" cy="18" r="3"/><circle cx="470" cy="52" r="2.5"/>
            <circle cx="550" cy="20" r="3"/><circle cx="630" cy="62" r="2.5"/><circle cx="710" cy="18" r="3"/>
            <circle cx="790" cy="55" r="2.5"/><circle cx="115" cy="110" r="2.5"/><circle cx="275" cy="125" r="3"/>
            <circle cx="445" cy="105" r="2.5"/><circle cx="610" cy="130" r="3"/><circle cx="760" cy="112" r="2.5"/>
          </g>""",
        "fog": """
          <g class="fog-layer" fill="none" stroke="#9BBAC5" stroke-width="13" stroke-linecap="round" opacity=".20">
            <path d="M20 65H300 M390 85H790 M80 115H520 M470 45H760"/>
          </g>""",
        "clouds": """
          <g class="cloud-layer" fill="#7298AA" opacity=".45">
            <path d="M95 66c-1-13 10-23 24-23 5-17 31-20 42-5 17-5 31 8 30 24z"/>
            <path d="M545 55c0-12 10-21 23-21 7-16 31-17 41-2 17-4 30 8 29 23z"/>
          </g>""",
        "clear": """
          <g class="sun-layer">
            <circle cx="690" cy="48" r="20" fill="#F4D98B" opacity=".9"/>
            <circle cx="690" cy="48" r="31" fill="#F4D98B" opacity=".10"/>
          </g>"""
    }[mode]

    weather_filter = "filter: drop-shadow(0 0 5px rgba(50,214,199,.2));"
    if mode == "rain":
        ground = "#102C3B"
    elif mode == "snow":
        ground = "#294454"
    elif mode == "fog":
        ground = "#243B49"
    else:
        ground = "#102B3A"

    scene_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; }}
      html, body {{ margin:0; padding:0; background:transparent; font-family:Inter,Arial,sans-serif; }}
      .weather-panel {{
        width:100%; overflow:hidden; border:1px solid #245267; border-radius:16px;
        background:linear-gradient(110deg,{sky_top},{sky_bottom}); color:#F4F8FC;
        box-shadow:0 12px 35px rgba(0,0,0,.18);
      }}
      .panel-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; padding:14px 18px 8px; }}
      .eyebrow {{ color:#32D6C7; font-size:10px; font-weight:700; letter-spacing:2px; }}
      .condition {{ color:#DCEAF2; font-size:12px; font-weight:700; }}
      .scene {{ display:block; width:100%; height:auto; }}
      .person {{ animation:walkAcross 11s linear infinite alternate; transform-box:fill-box; transform-origin:center; }}
      .legs {{ animation:step .42s ease-in-out infinite alternate; transform-box:fill-box; transform-origin:top center; }}
      .arm-a {{ animation:armSwing .42s ease-in-out infinite alternate; transform-box:fill-box; transform-origin:top center; }}
      .arm-b {{ animation:armSwing .42s ease-in-out infinite alternate-reverse; transform-box:fill-box; transform-origin:top center; }}
      .rain-layer {{ animation:rainFall .75s linear infinite; }}
      .snow-layer {{ animation:snowFall 5s linear infinite; }}
      .cloud-layer {{ animation:cloudDrift 18s ease-in-out infinite alternate; }}
      .fog-layer {{ animation:fogDrift 9s ease-in-out infinite alternate; }}
      .sun-layer {{ animation:sunPulse 4s ease-in-out infinite alternate; }}
      .panel-foot {{ display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; padding:0 18px 14px; color:#A8B8CA; font-size:11px; }}
      .weather-note {{ color:#32D6C7; }}
      @keyframes walkAcross {{ from {{ transform:translateX(-245px); }} to {{ transform:translateX(245px); }} }}
      @keyframes step {{ to {{ transform:rotate(22deg); }} }}
      @keyframes armSwing {{ to {{ transform:rotate(-20deg); }} }}
      @keyframes rainFall {{ to {{ transform:translate(18px,45px); }} }}
      @keyframes snowFall {{ to {{ transform:translateY(35px); }} }}
      @keyframes cloudDrift {{ to {{ transform:translateX(35px); }} }}
      @keyframes fogDrift {{ to {{ transform:translateX(30px); opacity:.35; }} }}
      @keyframes sunPulse {{ to {{ opacity:.65; }} }}
      @media(prefers-reduced-motion:reduce) {{ *,*::before,*::after {{ animation:none!important; }} }}
      @media(max-width:600px) {{ .panel-head {{ padding:11px 12px 5px; }} .panel-foot {{ padding:0 12px 11px; font-size:10px; }} }}
    </style>
    </head>
    <body>
      <div class="weather-panel">
        <div class="panel-head">
          <span class="eyebrow">{label_text}</span>
          <span class="condition">{condition_label} &nbsp;•&nbsp; {time_label}</span>
        </div>
        <svg class="scene" viewBox="0 0 840 190" role="img" aria-label="Custom animated person walking in {condition_label.lower()} weather">
          <defs>
            <linearGradient id="skybg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stop-color="{sky_top}"/><stop offset="1" stop-color="{sky_bottom}"/>
            </linearGradient>
            <linearGradient id="roadbg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stop-color="{ground}"/><stop offset="1" stop-color="#071521"/>
            </linearGradient>
          </defs>
          <rect width="840" height="190" fill="url(#skybg)"/>
          {weather_effects}
          <g fill="#183B50" stroke="#286078" stroke-width="1">
            <path d="M35 112V62h48v50z M91 112V39h58v73z M158 112V77h43v35z"/>
            <path d="M640 112V68h48v44z M695 112V48h56v64z M758 112V80h45v32z"/>
          </g>
          <g fill="#32D6C7" opacity=".55">
            <rect x="48" y="76" width="6" height="7"/><rect x="67" y="76" width="6" height="7"/>
            <rect x="108" y="53" width="7" height="7"/><rect x="128" y="53" width="7" height="7"/>
            <rect x="657" y="82" width="6" height="7"/><rect x="676" y="82" width="6" height="7"/>
            <rect x="713" y="63" width="7" height="7"/><rect x="734" y="63" width="7" height="7"/>
          </g>
          <path d="M0 126 Q210 119 420 126 T840 125 L840 190H0Z" fill="url(#roadbg)"/>
          <path d="M0 126 Q210 119 420 126 T840 125" fill="none" stroke="#2A6475" stroke-width="2"/>
          <path d="M0 159H840" stroke="#1B4658" stroke-width="1" stroke-dasharray="4 9"/>
          <path d="M0 181H840" stroke="#1B4658" stroke-width="1"/>
          <g fill="none" stroke="#32D6C7" stroke-width="2" opacity=".5">
            <path d="M20 145H180 M660 145H820"/>
          </g>
          <g class="person" transform="translate(420 0)" stroke="#43E6D4" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" fill="none">
            <circle cx="0" cy="83" r="7" fill="#43E6D4" stroke="none"/>
            <path d="M0 92L-2 119" stroke-width="5"/>
            <g class="arm-a"><path d="M-1 98L-15 111"/></g>
            <g class="arm-b"><path d="M-1 98L13 110"/></g>
            <g class="legs"><path d="M-2 119L-13 139 M-2 119L10 138"/></g>
            {('<path d="M-14 83 Q0 61 14 83" stroke="#32D6C7" stroke-width="3"/><path d="M0 83V91" stroke="#32D6C7"/>' if mode == "rain" else '')}
            {('<path d="M-7 91L-7 119 M7 91L7 119" stroke="#B7DDE8" stroke-width="2" opacity=".8"/>' if mode == "snow" else '')}
          </g>
          <text x="22" y="24" fill="#A8B8CA" font-size="10" letter-spacing="1.4">{time_label} SCENE</text>
          <text x="818" y="24" text-anchor="end" fill="#32D6C7" font-size="10">{condition_label}</text>
        </svg>
        <div class="panel-foot">
          <span>{desc_text}</span>
          <span>Rain: {rain_text} &nbsp; • &nbsp; Snow: {snow_text} &nbsp; • &nbsp; Clouds: {cloud_value}%</span>
        </div>
      </div>
    </body>
    </html>
    """
    st.components.v1.html(scene_html, height=255, scrolling=False)


# --------------------------------------------------
# PREDICTOR
# --------------------------------------------------

def predictor_page():
    raw, cleaned = require_data()
    if raw is None:
        return

    model = require_model()
    if model is None:
        return

    page_heading(
        "PREDICTIVE INTELLIGENCE",
        "Traffic Predictor",
        "Enter weather and time conditions to estimate hourly traffic volume.",
    )

    st.info(
        "This model predicts historical-pattern-based traffic volume. "
        "It does not provide live traffic measurements."
    )

    left, right = st.columns(2)

    with left:
        selected_date = st.date_input(
            "Select Date",
            value=datetime(2018, 9, 30).date(),
        )

        selected_time = st.time_input(
            "Select Time",
            value=time(17, 0),
        )

        temperature_c = st.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=60.0,
            value=15.0,
            step=0.5,
        )

        clouds = st.slider(
            "Cloud Coverage (%)",
            0, 100, 40,
        )

    with right:
        rain = st.number_input(
            "Rainfall (mm per hour)",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=0.1,
        )

        snow = st.number_input(
            "Snowfall (mm per hour)",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=0.1,
        )

        weather_options = sorted(
            cleaned["weather_main"].dropna().unique().tolist()
        )
        weather = st.selectbox(
            "Weather Condition",
            weather_options,
            index=(
                weather_options.index("Clear")
                if "Clear" in weather_options else 0
            ),
        )

    holiday = get_holiday_for_date(cleaned, selected_date)
    st.caption(
        f"Date-based holiday detection: **{holiday}**"
        if holiday != "None"
        else "Date-based holiday detection: No recorded holiday for this date."
    )

    descriptions = sorted(
        cleaned.loc[
            cleaned["weather_main"] == weather,
            "weather_description",
        ].dropna().unique().tolist()
    )

    description = st.selectbox(
        "Weather Description",
        descriptions if descriptions else ["Unknown"],
    )

    submitted = st.button(
        "Predict Traffic Volume",
        use_container_width=True,
    )

    render_weather_scene(
        weather, rain, snow, clouds, selected_time,
        label="PREDICTOR WEATHER PREVIEW"
    )

    if submitted:
        prediction_input = {
            "date_time": datetime.combine(
                selected_date, selected_time
            ),
            "temp": temperature_c + 273.15,
            "rain_1h": rain,
            "snow_1h": snow,
            "clouds_all": clouds,
            "weather_main": weather,
            "weather_description": description,
            "holiday": holiday,
        }

        try:
            result = predict_traffic(model, prediction_input)

            st.divider()
            st.success("Prediction generated successfully.")

            st.metric(
                "Estimated Hourly Traffic",
                f"{result:,} vehicles/hour",
            )

            st.caption(
                "Estimated traffic volume based on the trained "
                "historical dataset and selected input conditions."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# SCENARIO SIMULATOR
# --------------------------------------------------

def simulator_page():
    raw, cleaned = require_data()
    if raw is None:
        return

    model = require_model()
    if model is None:
        return

    page_heading(
        "SCENARIO ENGINE",
        "Traffic Simulator",
        "Explore how changing weather conditions may affect predicted traffic.",
    )

    st.info(
        "All scenarios use the same selected date and time. "
        "Only the specified conditions change."
    )

    selected_date = st.date_input(
        "Date",
        value=datetime(2018, 9, 30).date(),
        key="scenario_date",
    )

    selected_time = st.time_input(
        "Time",
        value=time(17, 0),
        key="scenario_time",
    )

    temperature_c = st.number_input(
        "Temperature (°C)",
        value=15.0,
        key="scenario_temp",
    )

    base_weather = st.selectbox(
        "Baseline Weather",
        sorted(cleaned["weather_main"].dropna().unique()),
        key="scenario_weather",
    )

    clouds = st.slider(
        "Cloud Coverage (%)",
        0, 100, 40,
        key="scenario_clouds",
    )

    run = st.button(
        "Run Scenario Simulation",
        use_container_width=True,
    )

    render_weather_scene(
        base_weather, 0.0, 0.0, clouds, selected_time,
        label="SIMULATOR BASELINE WEATHER"
    )

    if run:
        descriptions = cleaned.loc[
            cleaned["weather_main"] == base_weather,
            "weather_description",
        ].dropna()

        holiday = get_holiday_for_date(cleaned, selected_date)
        st.caption(
            f"Date-based holiday detection: **{holiday}**"
            if holiday != "None"
            else "Date-based holiday detection: No recorded holiday for this date."
        )

        base_input = {
            "date_time": datetime.combine(
                selected_date, selected_time
            ),
            "temp": temperature_c + 273.15,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": clouds,
            "weather_main": base_weather,
            "weather_description": (
                descriptions.iloc[0] if not descriptions.empty
                else "Unknown"
            ),
            "holiday": holiday,
        }

        scenarios = {
            "Baseline": {},
            "Heavy Rain": {
                "rain_1h": 5.0,
                "weather_main": "Rain",
                "weather_description": "moderate rain",
            },
            "Cloudy Weather": {
                "clouds_all": 95,
                "weather_main": "Clouds",
                "weather_description": "overcast clouds",
            },
            "Snowfall": {
                "snow_1h": 2.0,
                "weather_main": "Snow",
                "weather_description": "light snow",
            },
        }

        try:
            baseline, results = compare_scenarios(
                model, base_input, scenarios
            )

            st.divider()
            st.metric(
                "Baseline Prediction",
                f"{baseline:,} vehicles/hour",
            )

            st.plotly_chart(
                scenario_comparison_chart(results),
                use_container_width=True,
            )

            st.subheader("Scenario Weather Experiences")
            scenario_scene_specs = [
                ("Baseline", base_weather, 0.0, 0.0, clouds),
                ("Heavy Rain", "Rain", 5.0, 0.0, clouds),
                ("Cloudy Weather", "Clouds", 0.0, 0.0, 95),
                ("Snowfall", "Snow", 0.0, 2.0, clouds),
            ]
            for scene_name, scene_weather, scene_rain, scene_snow, scene_clouds in scenario_scene_specs:
                render_weather_scene(
                    scene_weather, scene_rain, scene_snow,
                    scene_clouds, selected_time,
                    label=f"SIMULATOR • {scene_name}"
                )

            st.subheader("Scenario Results")
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as error:
            st.error(f"Simulation failed: {error}")


# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

def comparison_page():
    raw, cleaned = require_data()
    if raw is None:
        return

    model = require_model()
    if model is None:
        return

    page_heading(
        "DECISION SUPPORT",
        "Scenario Comparison",
        "Compare four predicted traffic outcomes under identical base conditions.",
    )

    selected_date = st.date_input(
        "Comparison Date",
        value=datetime(2018, 9, 30).date(),
        key="comparison_date",
    )

    selected_time = st.time_input(
        "Comparison Time",
        value=time(17, 0),
        key="comparison_time",
    )

    holiday = get_holiday_for_date(cleaned, selected_date)
    st.caption(
        f"Date-based holiday detection: **{holiday}**"
        if holiday != "None"
        else "Date-based holiday detection: No recorded holiday for this date."
    )

    base_input = {
        "date_time": datetime.combine(
            selected_date, selected_time
        ),
        "temp": 288.15,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 40,
        "weather_main": "Clear",
        "weather_description": "Sky is clear",
        "holiday": holiday,
    }

    scenarios = {
        "Baseline": {},
        "Heavy Rain": {
            "rain_1h": 5.0,
            "weather_main": "Rain",
            "weather_description": "moderate rain",
        },
        "Cloudy Weather": {
            "clouds_all": 95,
            "weather_main": "Clouds",
            "weather_description": "overcast clouds",
        },
        "Snowfall": {
            "snow_1h": 2.0,
            "weather_main": "Snow",
            "weather_description": "light snow",
        },
    }

    if st.button("Generate Comparison", use_container_width=True):
        try:
            baseline, results = compare_scenarios(
                model, base_input, scenarios
            )

            st.metric(
                "Baseline Traffic",
                f"{baseline:,} vehicles/hour",
            )

            st.subheader("Weather Scenario Experiences")
            comparison_scene_specs = [
                ("Baseline", "Clear", 0.0, 0.0, 40),
                ("Heavy Rain", "Rain", 5.0, 0.0, 40),
                ("Cloudy Weather", "Clouds", 0.0, 0.0, 95),
                ("Snowfall", "Snow", 0.0, 2.0, 40),
            ]
            for scene_name, scene_weather, scene_rain, scene_snow, scene_clouds in comparison_scene_specs:
                render_weather_scene(
                    scene_weather, scene_rain, scene_snow,
                    scene_clouds, selected_time,
                    label=f"COMPARE • {scene_name}"
                )

            st.plotly_chart(
                scenario_comparison_chart(results),
                use_container_width=True,
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as error:
            st.error(f"Comparison failed: {error}")


# --------------------------------------------------
# MODEL LAB
# --------------------------------------------------

def model_lab_page():
    raw, cleaned = require_data()
    if raw is None:
        return

    page_heading(
        "MODEL INTELLIGENCE",
        "Model Lab",
        "Review model performance, evaluation metrics and training information.",
    )

    metrics = get_metrics()
    metadata = get_metadata()

    if not metrics.empty:
        st.subheader("Model Performance")

        display_metrics = metrics.copy()

        for col in ["MAE", "RMSE"]:
            if col in display_metrics:
                display_metrics[col] = display_metrics[col].round(2)

        if "R2" in display_metrics:
            display_metrics["R2"] = display_metrics["R2"].round(4)

        st.dataframe(
            display_metrics,
            use_container_width=True,
            hide_index=True,
        )

        best_model = metadata.get("best_model", "Not available")

        st.success(f"Selected model: {best_model}")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Training Rows",
            f"{metadata.get('train_rows', 0):,}",
        )
        c2.metric(
            "Testing Rows",
            f"{metadata.get('test_rows', 0):,}",
        )
        c3.metric(
            "Cleaned Rows",
            f"{metadata.get('cleaned_rows', 0):,}",
        )

        st.caption(
            f"Chronological split cutoff: "
            f"{metadata.get('split_cutoff', 'Not available')}"
        )

    else:
        st.warning("No model evaluation metrics found yet.")

    st.divider()
    st.subheader("Train Models")

    st.write(
        "Train and compare Random Forest, Extra Trees, "
        "and HistGradientBoosting using the historical dataset."
    )

    if st.button("Train and Evaluate Models", use_container_width=True):
        with st.spinner("Training models. Please wait..."):
            try:
                result = train_and_evaluate(raw)

                get_model.clear()
                get_metrics.clear()
                get_metadata.clear()

                st.success(
                    f"Training complete. Best model: "
                    f"{result['best_model']}"
                )

                st.dataframe(
                    result["metrics"].round(4),
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as error:
                st.error(f"Model training failed: {error}")

    st.divider()
    st.subheader("Model Methodology")

    st.markdown("""
    - **Problem type:** Supervised regression
    - **Target:** Hourly traffic volume
    - **Feature processing:** Missing-value imputation and one-hot encoding
    - **Feature engineering:** Time-based and cyclical features
    - **Evaluation:** MAE, RMSE and R²
    - **Validation:** Chronological train-test split
    """)

    st.caption(
        "Evaluation metrics describe performance on the held-out "
        "historical test period. They do not guarantee future accuracy."
    )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

def about_page():
    page_heading(
        "ABOUT THE PLATFORM",
        "About TrafficSense 2.0",
        "A historical traffic analysis and prediction platform.",
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-label">SMART MOBILITY • MACHINE LEARNING</div>
        <div class="hero-title">Turning Traffic Data<br>Into Intelligence.</div>
        <div class="hero-description">
            TrafficSense uses historical traffic and weather observations
            to analyse traffic patterns, estimate traffic volume and
            compare hypothetical conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Technology Stack")

    cols = st.columns(4)

    technologies = [
        ("Python", "Core programming"),
        ("Streamlit", "Web application"),
        ("Scikit-learn", "Machine learning"),
        ("Plotly", "Interactive visualizations"),
    ]

    for col, (name, description) in zip(cols, technologies):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{name}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Dataset Information")

    st.write(
        "The application uses the Metro Interstate Traffic Volume "
        "historical dataset, containing traffic volume, weather "
        "conditions and timestamps."
    )

    st.warning(
        "TrafficSense 2.0 is a historical-data-based demonstration. "
        "It is not connected to live traffic sensors, GPS feeds or "
        "real-time traffic services."
    )


def render_brand_logo():
    st.markdown("""
    <div class="brand-header">
        <div class="brand-lockup">
            <svg class="brand-symbol" viewBox="0 0 100 80"
                 xmlns="http://www.w3.org/2000/svg">
                <path d="M10 66 Q48 4 90 66"
                      fill="none" stroke="#32D6C7"
                      stroke-width="3" opacity=".7"/>
                <path d="M17 69 Q48 30 83 69"
                      fill="none" stroke="#32D6C7"
                      stroke-width="3" stroke-dasharray="100"
                      class="road-line"/>
                <path d="M50 67 L50 52"
                      stroke="#FFFFFF" stroke-width="2"
                      stroke-dasharray="5 5"
                      class="road-center"/>
                <g fill="none" stroke="#32D6C7"
                   stroke-width="2" class="city-lights">
                    <path d="M27 51 V36 H37 V51"/>
                    <path d="M41 48 V22 H52 V48"/>
                    <path d="M57 51 V31 H68 V51"/>
                </g>
            </svg>
            <div class="brand-name">
                <span>Traffic</span><span>Sense</span>
                <small>2.0</small>
            </div>
        </div>
        <div class="brand-tagline">
            INTELLIGENT MOBILITY PLATFORM
        </div>
    </div>
    <style>
        .brand-header {
            padding: 12px 0 16px;
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-symbol {
            width: 48px;
            height: 42px;
            overflow: visible;
        }
        .brand-name {
            font-family: 'Outfit', sans-serif;
            font-size: 25px;
            font-weight: 700;
            letter-spacing: -1px;
            color: #F4F8FC;
        }
        .brand-name span:nth-child(2) {
            color: #32D6C7;
        }
        .brand-name small {
            font-size: 11px;
            background: #32D6C7;
            color: #07131D;
            padding: 3px 6px;
            border-radius: 5px;
            margin-left: 7px;
            vertical-align: middle;
        }
        .brand-tagline {
            font-size: 9px;
            letter-spacing: 3px;
            color: #8298AD;
            margin-left: 58px;
            margin-top: -4px;
        }
        .road-line {
            animation: roadDraw 2.5s ease-in-out infinite alternate;
        }
        .road-center {
            animation: roadPulse 1.8s ease-in-out infinite;
        }
        .city-lights {
            animation: cityGlow 2s ease-in-out infinite alternate;
        }
        @keyframes roadDraw {
            from { stroke-dashoffset: 100; opacity: .4; }
            to { stroke-dashoffset: 0; opacity: 1; }
        }
        @keyframes roadPulse {
            50% { opacity: .25; }
        }
        @keyframes cityGlow {
            from { opacity: .45; }
            to { opacity: 1; }
        }
        @media (prefers-reduced-motion: reduce) {
            .road-line, .road-center, .city-lights {
                animation: none;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# TOP NAVIGATION
# --------------------------------------------------
render_brand_logo()
pages = [
    st.Page(home_page, title="Home", default=True),
    st.Page(analytics_page, title="Analytics"),
    st.Page(predictor_page, title="Predictor"),
    st.Page(simulator_page, title="Simulator"),
    st.Page(comparison_page, title="Compare"),
    st.Page(model_lab_page, title="Model Lab"),
    st.Page(about_page, title="About"),
]

navigation = st.navigation(pages, position="top")
navigation.run()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div style="
        text-align:center;
        color:#71849A;
        padding:30px 0 5px 0;
        font-size:12px;
    ">
        TRAFFICSENSE 2.0 • INTELLIGENT MOBILITY PLATFORM
    </div>
    """,
    unsafe_allow_html=True,
)
