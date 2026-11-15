from __future__ import annotations
import joblib
import pandas as pd
from .feature_engineering import build_features

def predict_one(pipeline, scenario: dict) -> float:
    frame = pd.DataFrame([scenario])
    features = build_features(frame)
    return float(max(0, pipeline.predict(features)[0]))

def load_pipeline(path):
    return joblib.load(path)

def classify_traffic(value: float, low_max: float, moderate_max: float) -> str:
    if value <= low_max: return "Low"
    if value <= moderate_max: return "Moderate"
    return "High"
