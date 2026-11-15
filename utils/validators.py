import pandas as pd

def validate_scenario(date_value, hour, temp_c, rainfall, snowfall, clouds):
    if not 0 <= int(hour) <= 23: raise ValueError("Hour must be between 0 and 23.")
    if not -90 <= float(temp_c) <= 60: raise ValueError("Temperature must be between -90°C and 60°C.")
    if float(rainfall) < 0 or float(snowfall) < 0: raise ValueError("Precipitation cannot be negative.")
    if not 0 <= float(clouds) <= 100: raise ValueError("Cloud coverage must be between 0 and 100%.")
    return pd.Timestamp(date_value)
