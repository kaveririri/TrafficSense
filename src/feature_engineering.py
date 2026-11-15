
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------
# FEATURE DEFINITIONS
# --------------------------------------------------

NUMERIC_FEATURES = [
    "temp",
    "rain_1h_log",
    "snow_1h_log",
    "clouds_all",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

CATEGORICAL_FEATURES = [
    "holiday_name",
    "weather_main",
    "weather_description",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def clean_raw_data(frame: pd.DataFrame) -> pd.DataFrame:

    df = frame.copy()

    required_columns = [
        "date_time",
        "traffic_volume",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Remove exact duplicate records
    df = df.drop_duplicates().copy()

    # Convert date and target columns
    df["date_time"] = pd.to_datetime(
        df["date_time"], errors="coerce"
    )

    df["traffic_volume"] = pd.to_numeric(
        df["traffic_volume"], errors="coerce"
    )

    # Convert numerical weather columns
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Remove rows without valid timestamps or targets
    df = df.dropna(
        subset=["date_time", "traffic_volume"]
    )

    # Traffic volume cannot be negative
    df = df[df["traffic_volume"] >= 0].copy()

    # Invalid weather values
    # Zero Kelvin is treated as missing
    df.loc[df["temp"] <= 0, "temp"] = np.nan

    for col in ["rain_1h", "snow_1h", "clouds_all"]:
        df.loc[df[col] < 0, col] = np.nan

    # Cloud coverage ranges from 0 to 100
    df["clouds_all"] = df["clouds_all"].clip(
        lower=0, upper=100
    )

    # Sort chronologically
    df = df.sort_values(
        "date_time", kind="stable"
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def build_features(frame: pd.DataFrame) -> pd.DataFrame:

    df = frame.copy()

    # Extract date and time
    dt = pd.to_datetime(
        df["date_time"], errors="coerce"
    )

    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    month = dt.dt.month

    # --------------------------------------------------
    # HOLIDAY FEATURES
    # --------------------------------------------------

    holiday = df["holiday"].copy() if "holiday" in df.columns else pd.Series(
        "None", index=df.index, dtype="object"
    )

    holiday = holiday.fillna("").astype(str).str.strip()

    # Correctly identify ordinary days
    no_holiday = holiday.str.lower().isin(
        ["", "none", "nan", "null"]
    )

    df["holiday_name"] = holiday.mask(
        no_holiday, "None"
    )

    df["is_holiday"] = (
        ~no_holiday
    ).astype(int)

    # --------------------------------------------------
    # NUMERICAL WEATHER FEATURES
    # --------------------------------------------------

    df["temp"] = pd.to_numeric(
        df["temp"], errors="coerce"
    )

    rain = pd.to_numeric(
        df["rain_1h"], errors="coerce"
    ).clip(lower=0)

    snow = pd.to_numeric(
        df["snow_1h"], errors="coerce"
    ).clip(lower=0)

    df["rain_1h_log"] = np.log1p(rain)
    df["snow_1h_log"] = np.log1p(snow)

    df["clouds_all"] = pd.to_numeric(
        df["clouds_all"], errors="coerce"
    )

    # --------------------------------------------------
    # CALENDAR FEATURES
    # --------------------------------------------------

    df["hour"] = hour
    df["day_of_week"] = dow
    df["day_of_month"] = dt.dt.day
    df["month"] = month
    df["day_of_year"] = dt.dt.dayofyear

    # Weekend indicator
    df["is_weekend"] = (
        dow >= 5
    ).astype(int)

    # --------------------------------------------------
    # CYCLICAL TIME FEATURES
    # --------------------------------------------------

    # Hour cycle
    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    # Day-of-week cycle
    df["dow_sin"] = np.sin(
        2 * np.pi * dow / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * dow / 7
    )

    # Month cycle
    df["month_sin"] = np.sin(
        2 * np.pi * (month - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (month - 1) / 12
    )

    # --------------------------------------------------
    # CATEGORICAL WEATHER FEATURES
    # --------------------------------------------------

    for col in ["weather_main", "weather_description"]:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    # Return features in a consistent order
    return df[FEATURE_COLUMNS].copy()


# --------------------------------------------------
# TRAINING DATA PREPARATION
# --------------------------------------------------

def prepare_training_frame(raw: pd.DataFrame):

    # Clean original dataset
    cleaned = clean_raw_data(raw)

    # Generate model input features
    X = build_features(cleaned)

    # Target variable
    y = cleaned["traffic_volume"].astype(float)

    # Timestamps for chronological splitting
    timestamps = cleaned["date_time"]

    return X, y, timestamps, cleaned
