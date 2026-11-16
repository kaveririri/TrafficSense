
import numpy as np
import pandas as pd


# Features used by the machine learning models
NUMERIC_FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]

CATEGORICAL_FEATURES = [
    "holiday",
    "weather_main",
    "weather_description",
]


def clean_raw_data(raw):
    """Clean the original traffic dataset and create time features."""

    df = raw.copy()

    # Parse date and target columns
    df["date_time"] = pd.to_datetime(
        df["date_time"], errors="coerce"
    )

    df["traffic_volume"] = pd.to_numeric(
        df["traffic_volume"], errors="coerce"
    )

    # Remove invalid timestamps and target values
    df = df.dropna(
        subset=["date_time", "traffic_volume"]
    ).copy()

    # Remove exact duplicate records only
    df = df.drop_duplicates().copy()

    # Fill missing holiday values
    df["holiday"] = (
        df["holiday"].fillna("None").astype(str)
    )

    # Convert numerical columns
    numeric_columns = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Fill missing weather categories
    for col in ["weather_main", "weather_description"]:
        df[col] = (
            df[col].fillna("Unknown").astype(str)
        )

    # Extract date and time features
    dt = df["date_time"]

    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclical encoding for time features
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day"] / 31
    )
    df["day_cos"] = np.cos(
        2 * np.pi * df["day"] / 31
    )

    return df.reset_index(drop=True)


def prepare_training_frame(raw):
    """Prepare model features, target and timestamps."""

    cleaned = clean_raw_data(raw)

    X = cleaned[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    y = cleaned["traffic_volume"].copy()

    timestamps = cleaned["date_time"].copy()

    return X, y, timestamps, cleaned
