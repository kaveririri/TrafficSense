from __future__ import annotations
import streamlit as st
from config import DEFAULT_DATASET, MODEL_PATH, METRICS_PATH, METADATA_PATH
from src.data_loader import load_dataset, inspect_dataset
from src.model_training import train_and_evaluate

st.set_page_config(page_title="TrafficSense | Traffic Intelligence", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ts-ink:#263238; --ts-muted:#748078; --ts-green:#638c78; --ts-green-dark:#456b59; --ts-amber:#d9a441; --ts-paper:#f5f1e8; --ts-card:#fffdf8; --ts-border:#e5ded0; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(145deg,#f7f3ea 0%,#f1f2eb 58%,#fbf8f0 100%); color:var(--ts-ink); }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; color:#172b2a !important; letter-spacing:-.025em; }
p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color:var(--ts-ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#263238 0%,#1f292d 100%); border-right:1px solid #3b484b; }
[data-testid="stSidebar"] * { color:#f3f0e7; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { color:#f3f0e7 !important; }
.block-container { padding-top:2rem; max-width:1480px; padding-bottom:3rem; }
.ts-hero { padding:2rem 2.2rem; border:1px solid #e5ded0; border-radius:22px; background:linear-gradient(115deg,#f3ead9 0%,#fbf8f0 62%,#f4ead7 100%); box-shadow:0 12px 34px rgba(49,55,48,.07); margin-bottom:1.25rem; }
.ts-home-title {font-family:Manrope,sans-serif; font-size:clamp(2.7rem,5vw,4.4rem); line-height:1; letter-spacing:-.055em; color:#263238; margin:.3rem 0 .8rem;}
.ts-home-title span {color:#b96f3e;}
.ts-home-subtitle {font-family:Manrope,sans-serif; font-size:1.25rem; font-weight:700; color:#455c50; margin:.4rem 0;}
.ts-home-copy {color:#748078; font-size:1rem; line-height:1.65; max-width:34rem;}
.ts-road-chip {display:inline-block; margin-top:.7rem; padding:.48rem .8rem; border-radius:999px; background:#f0e5d1; color:#896b3d; font-size:.7rem; font-weight:800; letter-spacing:.08em; border:1px solid #e5d6bb;}
.ts-model-road {margin-left:auto; font-size:2rem; background:#f3ead9; border-radius:14px; padding:.65rem .8rem;}
.ts-kicker { color:#147d64; font-size:.76rem; letter-spacing:.18em; font-weight:800; text-transform:uppercase; }
.ts-muted { color:#60756d !important; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #dce6df; padding:18px 20px; border-radius:16px; box-shadow:0 5px 18px rgba(29,65,48,.045); }
div[data-testid="stMetric"] label { color:#60756d !important; font-weight:600; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#172b2a; font-family:'Manrope',sans-serif; }
.stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"] button { border-radius:11px; border:1px solid #b97842; background:#b97842; color:#fff; font-weight:700; min-height:2.65rem; }
.stButton>button:hover, .stDownloadButton>button:hover { background:#986039; border-color:#986039; color:#fff; }
.stTabs [data-baseweb="tab"] { color:#5d7068; font-weight:600; }
.stTabs [aria-selected="true"] { color:#147d64 !important; }
[data-baseweb="input"]>div, [data-baseweb="select"]>div, [data-testid="stDateInput"] input, [data-testid="stNumberInput"] input { background:#fff; border-color:#d3e0d6; border-radius:10px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #dce6df; border-radius:12px; overflow:hidden; }
hr { border-color:#dce6df; }
.ts-note { padding:.9rem 1rem; border-radius:12px; background:#e7f3eb; border:1px solid #c9dfd2; color:#315b49; }
.ts-section-label { color:#789889; font-size:.70rem; letter-spacing:.19em; font-weight:800; margin:1.8rem 0 .55rem; }
.ts-model-banner { display:flex; align-items:center; gap:1rem; padding:1.1rem 1.25rem; border:1px solid #d5e3d9; border-radius:15px; background:linear-gradient(110deg,#e8f1e9,#f8f8f1); margin:.4rem 0 1rem; }
.ts-model-dot { width:11px; height:11px; border-radius:50%; background:#789889; box-shadow:0 0 0 5px #dce9df; flex:0 0 auto; }
.ts-model-eyebrow,.ts-detail-label { color:#789889; font-size:.68rem; font-weight:800; letter-spacing:.14em; }
.ts-model-name { color:#263c33; font-family:'Manrope',sans-serif; font-size:1.2rem; font-weight:800; margin-top:.1rem; }
.ts-model-description,.ts-detail-arrow { color:#78877f; font-size:.82rem; margin-top:.15rem; }
.ts-detail-card { background:rgba(255,255,255,.78); border:1px solid #dce6df; border-radius:15px; padding:1.1rem 1.2rem; min-height:116px; }
.ts-detail-value { color:#263c33; font-family:'Manrope',sans-serif; font-weight:700; font-size:1rem; margin-top:.45rem; overflow-wrap:anywhere; }


/* TrafficSense custom sidebar navigation */
[data-testid="stSidebar"] { background:linear-gradient(180deg,#263238 0%,#202a2e 100%) !important; border-right:1px solid #3b484b; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.25rem; }
.ts-side-brand { padding:.35rem .35rem 1.15rem; border-bottom:1px solid rgba(233,230,222,.16); margin-bottom:1rem; }
.ts-side-brand-title { color:#F5F1E8; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.35rem; letter-spacing:-.04em; }
.ts-side-brand-sub { color:#B9C2BE; font-size:.62rem; letter-spacing:.15em; font-weight:700; margin-top:.25rem; }
.ts-nav-label { color:#9EAAA4; font-size:.64rem; font-weight:800; letter-spacing:.17em; margin:.9rem .35rem .35rem; }
[data-testid="stSidebar"] .stButton > button { width:100%; text-align:left; justify-content:flex-start; background:transparent !important; color:#E9E6DE !important; border:1px solid transparent !important; border-radius:10px !important; padding:.62rem .8rem !important; min-height:2.65rem; font-weight:600; box-shadow:none !important; }
[data-testid="stSidebar"] .stButton > button:hover { background:rgba(233,230,222,.09) !important; border-color:rgba(233,230,222,.12) !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:#D9A441 !important; color:#263238 !important; border-color:#D9A441 !important; font-weight:800 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] { background:rgba(255,255,255,.06); border:1px dashed rgba(233,230,222,.25); border-radius:12px; padding:.55rem; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color:#E9E6DE !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#B9C2BE !important; }
.ts-data-card { background:rgba(255,255,255,.055); border:1px solid rgba(233,230,222,.13); border-radius:13px; padding:.85rem; margin-top:.6rem; }



/* GLOBAL DESIGN SYSTEM — shared by every TrafficSense page */
:root {
  --ts-asphalt:#263238; --ts-asphalt-2:#202a2e; --ts-paper:#f5f1e8;
  --ts-card:#fffdf8; --ts-amber:#d9a441; --ts-green:#638c78;
  --ts-red:#b96f69; --ts-blue:#8299a5; --ts-border:#e5ded0;
  --ts-muted:#748078; --ts-text:#263238;
}
.stApp { background:linear-gradient(145deg,#f7f3ea 0%,#f1f2eb 58%,#fbf8f0 100%) !important; }
.block-container { max-width:1480px !important; padding-top:2rem !important; padding-bottom:3rem !important; }
/* Consistent page titles and hierarchy */
.main h1 { font-family:'Manrope',sans-serif !important; color:var(--ts-asphalt) !important;
  font-size:clamp(1.9rem,3vw,2.55rem) !important; font-weight:800 !important;
  letter-spacing:-.045em !important; margin-bottom:.35rem !important; }
.main h2 { font-family:'Manrope',sans-serif !important; color:#34443e !important;
  font-size:1.35rem !important; font-weight:750 !important; letter-spacing:-.025em !important; }
.main h3 { font-family:'Manrope',sans-serif !important; color:#455c50 !important; font-weight:700 !important; }
.main p, .main label, .main [data-testid='stCaptionContainer'] { color:var(--ts-text); }
/* Common cards for metrics, forms, charts and data */
div[data-testid='stMetric'] { background:var(--ts-card) !important; border:1px solid var(--ts-border) !important;
  border-radius:16px !important; padding:18px 20px !important; box-shadow:0 6px 20px rgba(38,50,56,.055) !important; }
div[data-testid='stMetric'] label { color:#748078 !important; font-weight:650 !important; }
div[data-testid='stMetric'] [data-testid='stMetricValue'] { color:var(--ts-asphalt) !important;
  font-family:'Manrope',sans-serif !important; font-weight:800 !important; }
div[data-testid='stForm'], div[data-testid='stExpander'] { background:rgba(255,253,248,.88) !important;
  border:1px solid var(--ts-border) !important; border-radius:17px !important; padding: .65rem !important; }
div[data-testid='stPlotlyChart'], div[data-testid='stDataFrame'], div[data-testid='stTable'] {
  background:var(--ts-card) !important; border:1px solid var(--ts-border) !important;
  border-radius:16px !important; padding:.35rem !important; overflow:hidden !important;
  box-shadow:0 5px 18px rgba(38,50,56,.035) !important; }
/* Unified controls */
.stButton>button, .stDownloadButton>button, [data-testid='stFormSubmitButton'] button {
  background:#b97842 !important; color:#fff !important; border:1px solid #b97842 !important;
  border-radius:11px !important; min-height:2.65rem !important; font-weight:700 !important; }
.stButton>button:hover, .stDownloadButton>button:hover, [data-testid='stFormSubmitButton'] button:hover {
  background:#986039 !important; border-color:#986039 !important; color:#fff !important; }
[data-baseweb='input']>div, [data-baseweb='select']>div, [data-testid='stDateInput'] input,
[data-testid='stNumberInput'] input { background:#fffdf8 !important; border-color:#d9d2c5 !important; border-radius:10px !important; }
.stTabs [data-baseweb='tab'] { color:#68766e !important; font-weight:650 !important; }
.stTabs [aria-selected='true'] { color:#b97842 !important; }
hr { border-color:#e5ded0 !important; }
/* Shared section labels and informational panels */
.ts-page-kicker { color:#b97842; font-size:.68rem; letter-spacing:.18em; font-weight:850; text-transform:uppercase; margin-bottom:.35rem; }
.ts-page-intro { color:#748078; font-size:.96rem; line-height:1.6; margin:.2rem 0 1.2rem; }
.ts-panel { background:rgba(255,253,248,.92); border:1px solid var(--ts-border); border-radius:16px; padding:1rem 1.15rem; }
.ts-note { background:#e9f0e8 !important; border:1px solid #d4e1d4 !important; color:#405d4d !important; border-radius:13px !important; }
/* Sidebar remains visually consistent across all pages */
[data-testid='stSidebar'] { background:linear-gradient(180deg,#263238 0%,#202a2e 100%) !important; }
[data-testid='stSidebar'] * { font-family:'DM Sans',sans-serif; }
[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] p { color:#e9e6de !important; }
/* Weather and traffic result panels use the same rounded visual language */
.ts-weather-scene, .ts-traffic-result { border-radius:16px !important; overflow:hidden; }

</style>
""", unsafe_allow_html=True)

if "dataset" not in st.session_state:
    try:
        st.session_state.dataset = load_dataset(DEFAULT_DATASET)
        st.session_state.dataset_name = DEFAULT_DATASET.name
    except Exception as exc:
        st.session_state.dataset = None
        st.session_state.dataset_error = str(exc)

# Persistent direct-click navigation: no radio list, dropdown, or extra selection screen.
NAV_ITEMS = {
    "Dashboard": "Home Dashboard",
    "Traffic Analytics": "Traffic Analytics",
    "Single Simulator": "Scenario Simulator",
    "Scenario Comparison": "Scenario Comparison",
    "How It Works": "About & Methodology",
}
if "page" not in st.session_state:
    st.session_state.page = "Home Dashboard"

with st.sidebar:
    st.markdown("""<div class="ts-side-brand"><div class="ts-side-brand-title">🚦 TrafficSense</div><div class="ts-side-brand-sub">URBAN MOBILITY INTELLIGENCE</div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="ts-nav-label">OVERVIEW</div>', unsafe_allow_html=True)
    if st.button("⌂   Dashboard", key="nav_dashboard", use_container_width=True, type="primary" if st.session_state.page == "Home Dashboard" else "secondary"):
        st.session_state.page = "Home Dashboard"
        st.rerun()
    st.markdown('<div class="ts-nav-label">ANALYZE</div>', unsafe_allow_html=True)
    if st.button("▥   Traffic Analytics", key="nav_analytics", use_container_width=True, type="primary" if st.session_state.page == "Traffic Analytics" else "secondary"):
        st.session_state.page = "Traffic Analytics"
        st.rerun()
    st.markdown('<div class="ts-nav-label">PREDICT</div>', unsafe_allow_html=True)
    if st.button("◉   Single Simulator", key="nav_simulator", use_container_width=True, type="primary" if st.session_state.page == "Scenario Simulator" else "secondary"):
        st.session_state.page = "Scenario Simulator"
        st.rerun()
    if st.button("⇄   Scenario Comparison", key="nav_comparison", use_container_width=True, type="primary" if st.session_state.page == "Scenario Comparison" else "secondary"):
        st.session_state.page = "Scenario Comparison"
        st.rerun()
    st.markdown('<div class="ts-nav-label">PROJECT</div>', unsafe_allow_html=True)
    if st.button("ⓘ   How It Works", key="nav_about", use_container_width=True, type="primary" if st.session_state.page == "About & Methodology" else "secondary"):
        st.session_state.page = "About & Methodology"
        st.rerun()
    st.divider()
    st.markdown('<div class="ts-nav-label">DATA SOURCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="ts-data-card">Upload a compatible Metro Interstate Traffic Volume CSV to use a different dataset.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Traffic dataset (.csv)", type=["csv"], help="Upload a compatible Metro Interstate Traffic Volume CSV.", label_visibility="collapsed")
    if uploaded is not None:
        if st.button("Load dataset", key="load_dataset", use_container_width=True):
            try:
                st.session_state.dataset = load_dataset(uploaded)
                st.session_state.dataset_name = uploaded.name
                st.session_state.pop("training_result", None)
                st.session_state.pop("training_error", None)
                st.session_state.force_retrain = True
                st.success("Dataset loaded. Train the models from the dashboard.")
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.get("dataset") is not None:
        st.caption(f"Loaded: {st.session_state.get('dataset_name','CSV')}")

page = st.session_state.page

raw = st.session_state.get("dataset")
if raw is None:
    st.title("Welcome to TrafficSense")
    st.warning(st.session_state.get("dataset_error", "Upload your Metro traffic CSV to begin."))
    st.stop()

if ("training_result" not in st.session_state and not st.session_state.get("force_retrain", False)
        and MODEL_PATH.exists() and METRICS_PATH.exists() and METADATA_PATH.exists()):
    try:
        import json, joblib, pandas as pd
        from src.feature_engineering import clean_raw_data
        cleaned = clean_raw_data(raw)
        st.session_state.training_result = {"pipeline": joblib.load(MODEL_PATH), "metrics": pd.read_json(METRICS_PATH),
            "metadata": json.loads(METADATA_PATH.read_text()), "cleaned": cleaned,
            "best_model": json.loads(METADATA_PATH.read_text())["best_model"]}
    except Exception:
        pass

if page == "Home Dashboard":
    from pages.home import render
elif page == "Traffic Analytics":
    from pages.analytics import render
elif page == "Scenario Simulator":
    from pages.simulator import render
elif page == "Scenario Comparison":
    from pages.comparison import render
else:
    from pages.about import render
render(raw)
