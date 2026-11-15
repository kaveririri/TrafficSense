from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = {
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume"
}

class DatasetError(ValueError):
    pass

def load_dataset(source) -> pd.DataFrame:
    try:
        frame = pd.read_csv(source)
    except Exception as exc:
        raise DatasetError(f"Could not read the CSV file: {exc}") from exc
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DatasetError("Dataset is missing required columns: " + ", ".join(sorted(missing)))
    return frame

def inspect_dataset(frame: pd.DataFrame) -> dict:
    dates = pd.to_datetime(frame["date_time"], errors="coerce")
    target = pd.to_numeric(frame["traffic_volume"], errors="coerce")
    return {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "exact_duplicates": int(frame.duplicated().sum()),
        "invalid_timestamps": int(dates.isna().sum()),
        "timestamp_start": None if dates.notna().sum() == 0 else str(dates.min()),
        "timestamp_end": None if dates.notna().sum() == 0 else str(dates.max()),
        "unique_timestamps": int(dates.nunique()),
        "target_missing": int(target.isna().sum()),
        "target_min": None if target.notna().sum() == 0 else float(target.min()),
        "target_max": None if target.notna().sum() == 0 else float(target.max()),
    }
