
from pathlib import Path
import pandas as pd


def load_dataset(file_path):
    """Load the traffic dataset safely."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    required_columns = [
        "date_time",
        "traffic_volume",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
        "holiday",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["date_time"] = pd.to_datetime(
        df["date_time"], errors="coerce"
    )

    df["traffic_volume"] = pd.to_numeric(
        df["traffic_volume"], errors="coerce"
    )

    return df


def inspect_dataset(df):
    """Return basic dataset information."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_timestamps": int(
            df["date_time"].nunique()
        ),
        "date_start": str(df["date_time"].min()),
        "date_end": str(df["date_time"].max()),
        "average_traffic": round(
            float(df["traffic_volume"].mean()), 2
        ),
        "maximum_traffic": int(
            df["traffic_volume"].max()
        ),
    }
