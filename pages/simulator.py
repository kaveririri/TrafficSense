import streamlit as st
import pandas as pd
from datetime import date
from src.feature_engineering import clean_raw_data
from src.scenario_analysis import holiday_for_date, date_coverage_status
from src.prediction import predict_one, classify_traffic
from utils.validators import validate_scenario
from src.weather_visuals import weather_scene, traffic_result

def render(raw):
    st.markdown('<div class="ts-page-kicker">WHAT-IF PREDICTION</div><div class="ts-page-intro">Set the weather and time conditions to estimate traffic volume and congestion.</div>', unsafe_allow_html=True)
    st.title("Scenario Simulator")
    result=st.session_state.get("training_result")
    if result is None:
        st.warning("Train the models on the Home Dashboard before running predictions."); return
    cleaned=result["cleaned"]
    st.caption("Set a future or historical traffic scenario. Holiday status is inferred from the dataset's date-level holiday records, with a US–Minnesota holiday calendar fallback.")
    coverage = date_coverage_status(date.today(), cleaned)
    st.caption(f"Observed model history: {coverage['start']} to {coverage['end']}.")
    with st.form("scenario_form"):
        c1,c2,c3=st.columns(3)
        selected_date=c1.date_input("Scenario date",value=date.today())
        hour=c2.slider("Hour of day",0,23,8)
        temp_c=c3.number_input("Temperature (°C)",min_value=-90.0,max_value=60.0,value=15.0,step=0.5)
        c4,c5,c6=st.columns(3)
        rain=c4.number_input("Rainfall (mm/h)",min_value=0.0,value=0.0,step=0.1)
        snow=c5.number_input("Snowfall (mm/h)",min_value=0.0,value=0.0,step=0.1)
        clouds=c6.slider("Cloud coverage (%)",0,100,40)
        weather=st.selectbox("Main weather condition",sorted(cleaned.weather_main.dropna().astype(str).unique()))
        submitted=st.form_submit_button("Predict traffic volume",type="primary",use_container_width=True)
    if submitted:
        try:
            coverage = date_coverage_status(selected_date, cleaned)
            if coverage["outside"]:
                st.warning(f"This date is outside the model's observed history ({coverage['start']} to {coverage['end']}). The prediction is an extrapolation and may be less reliable than predictions within that period.")
            dt=validate_scenario(selected_date,hour,temp_c,rain,snow,clouds)
            holiday=holiday_for_date(dt,cleaned)
            descriptions=cleaned.loc[cleaned.weather_main.astype(str)==weather,"weather_description"].dropna().astype(str)
            description=descriptions.mode().iloc[0] if not descriptions.empty else weather.lower()
            scenario={"date_time":dt+pd.Timedelta(hours=int(hour)),"holiday":holiday or "","temp":float(temp_c)+273.15,
                      "rain_1h":rain,"snow_1h":snow,"clouds_all":clouds,"weather_main":weather,"weather_description":description}
            prediction=predict_one(result["pipeline"],scenario)
            thresholds=result["metadata"]["target_thresholds"]
            category=classify_traffic(prediction,thresholds["low_max"],thresholds["moderate_max"])
            st.divider(); st.subheader("Predicted conditions")
            st.markdown(weather_scene(weather), unsafe_allow_html=True)
            thresholds=result["metadata"]["target_thresholds"]
            st.markdown(traffic_result(category, prediction, thresholds["low_max"], thresholds["moderate_max"]), unsafe_allow_html=True)
            a,b=st.columns(2)
            a.metric("Detected holiday",holiday or "No holiday detected")
            b.metric("Temperature supplied to model",f"{temp_c:.1f} °C")
            st.caption(f"Weather description supplied to model: {description}. Temperature converted to {scenario['temp']:.2f} K.")
            st.session_state.last_scenario={"date":str(selected_date),"hour":int(hour),"temperature_c":temp_c,"rainfall":rain,"snowfall":snow,"clouds":clouds,"weather":weather,"holiday":holiday,"predicted_volume":prediction,"category":category}
        except Exception as exc:
            st.error(f"Could not generate prediction: {exc}")
