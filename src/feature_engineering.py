from __future__ import annotations
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "temp", "rain_1h_log", "snow_1h_log", "clouds_all",
    "hour", "day_of_week", "day_of_month", "month", "day_of_year",
    "is_weekend", "is_holiday", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "month_sin", "month_cos"
]
CATEGORICAL_FEATURES = ["holiday_name", "weather_main", "weather_description"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def clean_raw_data(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df = df.drop_duplicates().copy()
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date_time", "traffic_volume"])
    df = df[df["traffic_volume"] >= 0].copy()
    # Negative precipitation/cloud values are invalid; zero Kelvin is treated as missing.
    df.loc[df["temp"] <= 0, "temp"] = np.nan
    for col in ["rain_1h", "snow_1h", "clouds_all"]:
        df.loc[df[col] < 0, col] = np.nan
    df["clouds_all"] = df["clouds_all"].clip(0, 100)
    return df.sort_values("date_time", kind="stable").reset_index(drop=True)

def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    dt = pd.to_datetime(df["date_time"], errors="coerce")
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    month = dt.dt.month
    holiday = df.get("holiday", pd.Series(index=df.index, dtype="object"))
    holiday = holiday.fillna("").astype(str).str.strip()
    df["holiday_name"] = holiday.where(holiday.ne(""), "None")
    df["is_holiday"] = holiday.ne("").astype(int)
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["rain_1h_log"] = np.log1p(pd.to_numeric(df["rain_1h"], errors="coerce").clip(lower=0))
    df["snow_1h_log"] = np.log1p(pd.to_numeric(df["snow_1h"], errors="coerce").clip(lower=0))
    df["clouds_all"] = pd.to_numeric(df["clouds_all"], errors="coerce")
    df["hour"] = hour
    df["day_of_week"] = dow
    df["day_of_month"] = dt.dt.day
    df["month"] = month
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = (dow >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    for col in ["weather_main", "weather_description"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    return df[FEATURE_COLUMNS]

def prepare_training_frame(raw: pd.DataFrame):
    cleaned = clean_raw_data(raw)
    X = build_features(cleaned)
    y = cleaned["traffic_volume"].astype(float)
    timestamps = cleaned["date_time"]
    return X, y, timestamps, cleaned
