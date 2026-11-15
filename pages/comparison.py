import streamlit as st
import pandas as pd
import plotly.express as px
px.defaults.color_discrete_sequence = [
    "#789889", "#83A9A5", "#D2B58A", "#A69BB8", "#C99B96", "#87949A"
]
from datetime import date
from src.prediction import predict_one
from src.scenario_analysis import holiday_for_date, historical_weekday_hour_average, date_coverage_status
from src.visualization import DARK_LAYOUT
from utils.validators import validate_scenario
from src.prediction import classify_traffic
from src.weather_visuals import weather_scene, traffic_result

def render(raw):
    st.markdown('<div class="ts-page-kicker">SCENARIO LAB</div><div class="ts-page-intro">Compare multiple traffic scenarios against each other and historical conditions.</div>', unsafe_allow_html=True)
    st.title("Scenario Comparison")
    result=st.session_state.get("training_result")
    if result is None:
        st.warning("Train the models on the Home Dashboard before comparing scenarios."); return
    cleaned=result["cleaned"]
    st.caption("Build several what-if scenarios and compare predictions against one another and matched historical weekday/hour averages.")
    if "scenarios" not in st.session_state: st.session_state.scenarios=[]
    with st.form("comparison_add"):
        c1,c2,c3=st.columns(3)
        name=c1.text_input("Scenario name",value=f"Scenario {len(st.session_state.scenarios)+1}")
        day=c2.date_input("Date",value=date.today(),key="compare_date")
        hour=c3.slider("Hour",0,23,8,key="compare_hour")
        c4,c5,c6=st.columns(3)
        temp=c4.number_input("Temperature (°C)",min_value=-90.0,max_value=60.0,value=15.0,key="compare_temp")
        rain=c5.number_input("Rainfall (mm/h)",min_value=0.0,value=0.0,key="compare_rain")
        snow=c6.number_input("Snowfall (mm/h)",min_value=0.0,value=0.0,key="compare_snow")
        clouds=st.slider("Cloud coverage (%)",0,100,40,key="compare_clouds")
        weather=st.selectbox("Weather",sorted(cleaned.weather_main.dropna().astype(str).unique()),key="compare_weather")
        add=st.form_submit_button("Add scenario",type="primary")
    if add:
        try:
            validate_scenario(day,hour,temp,rain,snow,clouds)
            coverage = date_coverage_status(day, cleaned)
            if coverage["outside"]:
                st.warning(f"{day} is outside the observed model history ({coverage['start']} to {coverage['end']}); treat this prediction as an extrapolation.")
            holiday=holiday_for_date(day,cleaned)
            descriptions=cleaned.loc[cleaned.weather_main.astype(str)==weather,"weather_description"].dropna().astype(str)
            description=descriptions.mode().iloc[0] if not descriptions.empty else weather.lower()
            scenario={"date_time":pd.Timestamp(day)+pd.Timedelta(hours=hour),"holiday":holiday,"temp":temp+273.15,"rain_1h":rain,"snow_1h":snow,"clouds_all":clouds,"weather_main":weather,"weather_description":description}
            prediction=predict_one(result["pipeline"],scenario)
            thresholds=result["metadata"]["target_thresholds"]
            category=classify_traffic(prediction,thresholds["low_max"],thresholds["moderate_max"])
            hist=historical_weekday_hour_average(cleaned,day,hour)
            st.session_state.scenarios.append({"scenario":name,"date":str(day),"weekday":hist["weekday"],"hour":hour,"weather":weather,"predicted_volume":prediction,"category":category,"historical_average":hist["mean"],"historical_median":hist["median"],"historical_matches":hist["count"],"holiday":holiday or "None"})
        except Exception as exc: st.error(str(exc))
    if st.session_state.scenarios:
        frame=pd.DataFrame(st.session_state.scenarios)
        baseline=frame.predicted_volume.iloc[0]
        frame["difference_from_first_pct"]=((frame.predicted_volume-baseline)/baseline*100) if baseline else 0
        frame["difference_from_historical"] = frame["predicted_volume"] - frame["historical_average"]
        frame["difference_from_historical_pct"] = (frame["difference_from_historical"] / frame["historical_average"] * 100).where(frame["historical_average"].notna() & (frame["historical_average"] != 0))
        st.subheader("Scenario conditions and predictions")
        for start in range(0, len(frame), 2):
            cols=st.columns(2)
            for col, (_, row) in zip(cols, frame.iloc[start:start+2].iterrows()):
                with col:
                    st.markdown(f"**{row['scenario']} · {row['weather']}**")
                    st.markdown(weather_scene(row["weather"], show_person=True), unsafe_allow_html=True)
                    thresholds=result["metadata"]["target_thresholds"]
                    st.markdown(traffic_result(row["category"], row["predicted_volume"], thresholds["low_max"], thresholds["moderate_max"]), unsafe_allow_html=True)
        st.subheader("Scenario results")
        chart=px.bar(frame,x="scenario",y="predicted_volume",color="scenario",title="Predicted traffic volume by scenario")
        chart.update_layout(**DARK_LAYOUT); st.plotly_chart(chart,use_container_width=True)
        st.markdown('<div class="ts-note"><strong>How to read this comparison:</strong> The historical reference is matched by weekday and hour, so scenarios on the same weekday/hour share the same historical average. Changes in predicted volume reflect scenario inputs, including weather and holiday features.</div>', unsafe_allow_html=True)
        st.caption("Difference from historical = predicted volume − matched historical average. Percentage difference is relative to that historical average.")
        st.dataframe(frame,use_container_width=True,hide_index=True)
        if st.button("Clear all scenarios"):
            st.session_state.scenarios=[]; st.rerun()
        st.download_button("Download comparison CSV",frame.to_csv(index=False),file_name="trafficsense_scenario_comparison.csv",mime="text/csv")
    else:
        st.info("Add your first scenario using the form above.")
