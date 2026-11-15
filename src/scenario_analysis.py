from __future__ import annotations
import pandas as pd


def holiday_for_date(selected_date, cleaned_history: pd.DataFrame) -> str:
    day = pd.Timestamp(selected_date).date()
    matched = cleaned_history.loc[cleaned_history["date_time"].dt.date == day, "holiday"]
    labels = [str(x).strip() for x in matched.dropna().unique() if str(x).strip()]
    if labels:
        return labels[0]
    try:
        import holidays
        calendar = holidays.US(years=[day.year], state="MN")
        return str(calendar.get(day, ""))
    except ImportError:
        return ""


def historical_weekday_hour_average(cleaned_history: pd.DataFrame, selected_date, hour: int):
    day = pd.Timestamp(selected_date)
    mask = ((cleaned_history.date_time.dt.dayofweek == day.dayofweek) &
            (cleaned_history.date_time.dt.hour == int(hour)))
    values = cleaned_history.loc[mask, "traffic_volume"]
    return {"mean": None if values.empty else float(values.mean()),
            "median": None if values.empty else float(values.median()),
            "count": int(len(values)), "weekday": day.day_name(), "hour": int(hour)}


def compare_scenarios(rows: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if not result.empty and "predicted_volume" in result:
        baseline = result["predicted_volume"].iloc[0]
        result["difference_from_first_pct"] = result["predicted_volume"].apply(
            lambda x: None if baseline == 0 else (x-baseline)/baseline*100)
    return result


def date_coverage_status(selected_date, cleaned_history: pd.DataFrame) -> dict:
    """Describe whether a scenario date falls inside the observed dataset history."""
    day = pd.Timestamp(selected_date).date()
    start = cleaned_history["date_time"].min().date()
    end = cleaned_history["date_time"].max().date()
    return {"start": start, "end": end, "outside": day < start or day > end,
            "before": day < start, "after": day > end}
