import streamlit as st
import pandas as pd
import plotly.express as px
px.defaults.color_discrete_sequence = [
    "#789889", "#83A9A5", "#D2B58A", "#A69BB8", "#C99B96", "#87949A"
]
from src.feature_engineering import clean_raw_data
from src.visualization import DARK_LAYOUT

def render(raw):
    st.markdown('<div class="ts-page-kicker">TRAFFIC INTELLIGENCE</div><div class="ts-page-intro">Explore historical traffic patterns, weather relationships and time-based trends.</div>', unsafe_allow_html=True)
    st.title("Traffic Analytics")
    df = clean_raw_data(raw)
    df["hour"] = df.date_time.dt.hour
    df["weekday"] = df.date_time.dt.day_name()
    df["month"] = df.date_time.dt.month_name()
    df["year"] = df.date_time.dt.year
    df["date"] = df.date_time.dt.date
    st.caption("Explore observed traffic volume and its relationship with time, weather and holidays.")
    tab1,tab2,tab3 = st.tabs(["Time patterns", "Weather impact", "Holiday & distribution"])
    with tab1:
        c1,c2 = st.columns(2)
        hourly = df.groupby("hour", as_index=False).traffic_volume.mean()
        fig=px.line(hourly,x="hour",y="traffic_volume",markers=True,title="Average traffic by hour")
        fig.update_layout(**DARK_LAYOUT); c1.plotly_chart(fig,use_container_width=True)
        weekday_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        daily=df.groupby("weekday",as_index=False).traffic_volume.mean(); daily["weekday"]=pd.Categorical(daily.weekday,categories=weekday_order,ordered=True); daily=daily.sort_values("weekday")
        fig=px.bar(daily,x="weekday",y="traffic_volume",title="Average traffic by weekday"); fig.update_layout(**DARK_LAYOUT); c2.plotly_chart(fig,use_container_width=True)
        monthly=df.assign(month_num=df.date_time.dt.month).groupby("month_num",as_index=False).traffic_volume.mean()
        monthly["month_name"]=pd.to_datetime(monthly.month_num,format="%m").dt.month_name()
        monthly=monthly.sort_values("month_num")
        fig=px.line(monthly,x="month_name",y="traffic_volume",markers=True,title="Average traffic by month of year"); fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
        yearly=df.groupby("year",as_index=False).traffic_volume.mean()
        fig=px.bar(yearly,x="year",y="traffic_volume",title="Average traffic by year")
        fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
        first_year, last_year = df.date_time.min().year, df.date_time.max().year
        st.caption(f"Coverage note: {first_year} and {last_year} are partial calendar years in this dataset; compare their averages with that limitation in mind.")
    with tab2:
        weather=df.groupby("weather_main",as_index=False).traffic_volume.agg(["mean","median","count"]).reset_index()
        fig=px.bar(weather,x="weather_main",y="mean",hover_data=["count","median"],title="Average traffic by main weather condition"); fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
        fig=px.scatter(df.sample(min(5000,len(df)),random_state=42),x="temp",y="traffic_volume",color="weather_main",opacity=.5,title="Temperature and traffic volume")
        fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
        rain=df.assign(rain_bucket=pd.cut(df.rain_1h,[-.001,0,.1,1,5,10000],labels=["None","Trace","Light","Moderate","Heavy"]))
        rain_avg=rain.groupby("rain_bucket",observed=False,as_index=False).traffic_volume.mean()
        fig=px.bar(rain_avg,x="rain_bucket",y="traffic_volume",title="Average traffic by rainfall bucket"); fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
    with tab3:
        df["holiday_label"]=df.holiday.fillna("Non-holiday")
        holiday=df.groupby("holiday_label",as_index=False).traffic_volume.mean().sort_values("traffic_volume")
        fig=px.bar(holiday,x="holiday_label",y="traffic_volume",title="Average traffic by recorded holiday label"); fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
        fig=px.histogram(df,x="traffic_volume",nbins=45,title="Traffic volume distribution")
        fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
