
import numpy as np
import pandas as pd

from .feature_engineering import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def prepare_prediction_input(input_data):
    """Convert user inputs into the model's expected features."""

    dt = pd.to_datetime(input_data["date_time"])

    hour = dt.hour
    day = dt.day
    month = dt.month
    day_of_week = dt.dayofweek

    row = {
        "temp": float(input_data.get("temp", 288.0)),
        "rain_1h": float(input_data.get("rain_1h", 0.0)),
        "snow_1h": float(input_data.get("snow_1h", 0.0)),
        "clouds_all": float(input_data.get("clouds_all", 40)),
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * day / 31),
        "day_cos": np.cos(2 * np.pi * day / 31),
        "holiday": str(input_data.get("holiday", "None")),
        "weather_main": str(
            input_data.get("weather_main", "Clear")
        ),
        "weather_description": str(
            input_data.get("weather_description", "Sky is clear")
        ),
    }

    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    return pd.DataFrame([row], columns=columns)


def predict_traffic(pipeline, input_data):
    """Predict hourly traffic volume."""

    features = prepare_prediction_input(input_data)

    prediction = pipeline.predict(features)[0]

    return max(0, int(round(float(prediction))))
