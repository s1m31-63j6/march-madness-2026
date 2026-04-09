import re
import pandas as pd
import numpy as np
from dateutil import parser as dtparser


def _parse_messy_datetime(series: pd.Series) -> pd.Series:
    """
    Parse a series of messy datetime strings to datetime64.

    Handles:
      - DD-Mon-YYYY HH:MM          e.g. '08-Mar-2025 12:50'
      - MM/DD/YYYY HH:MM AM/PM     e.g. '07/08/2025 06:00 PM'
      - YYYY-MM-DD HH:MM           e.g. '2025-04-02 09:30'
      - YYYY/MM/DD HH:MM:SS        e.g. '2025/05/13 08:20:00'
      - All of the above with trailing MDT / MST tokens and extra whitespace
    """
    def _parse_one(val):
        if pd.isna(val):
            return pd.NaT
        # Strip whitespace and drop timezone tokens (all data is Mountain time)
        clean = re.sub(r"\s+(MDT|MST)\s*$", "", val.strip())
        try:
            # dayfirst=False → MM/DD/YYYY for ambiguous slash formats (US locale)
            return dtparser.parse(clean, dayfirst=False)
        except (ValueError, OverflowError, TypeError):
            return pd.NaT

    return series.map(_parse_one)


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert messy datetime strings into usable datetime objects and engineer
    time-based analytical features.

    Parsed columns created:
      - stop_datetime             parsed from stop_datetime_raw
      - scheduled_window_start    parsed from scheduled_window_start_raw

    Engineered features:
      - day_of_week   int  0=Monday … 6=Sunday (derived from stop_datetime)
      - is_weekend    int  1 if Saturday/Sunday, 0 otherwise
      - lateness_min  float  minutes past the scheduled window start at actual
                             arrival; capped at 0 (never negative — early / on-time = 0)
    """
    df = df.copy()

    # --- Parse raw datetime columns ---
    df["stop_datetime"] = _parse_messy_datetime(df["stop_datetime_raw"])
    df["scheduled_window_start"] = _parse_messy_datetime(df["scheduled_window_start_raw"])

    # --- Day-of-week and weekend indicators ---
    df["day_of_week"] = df["stop_datetime"].dt.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)  # 1=Sat/Sun

    # --- Lateness metric (minutes, never negative) ---
    # actual_arrival_min = minutes elapsed since 7 AM shift start (verified against raw data)
    # scheduled_window_start = parsed scheduled window start datetime
    # scheduled_window_min = duration of the delivery window in minutes
    # Window end (from 7 AM) = (sched_start_from_midnight - 420) + scheduled_window_min
    # Positive → arrived after window closed; 0 → arrived within or before window
    sched_mins_from_midnight = (
        df["scheduled_window_start"].dt.hour * 60
        + df["scheduled_window_start"].dt.minute
    )
    window_end_from_7am = (sched_mins_from_midnight - 420) + df["scheduled_window_min"]
    df["lateness_min"] = (
        df["actual_arrival_min"] - window_end_from_7am
    ).clip(lower=0)

    return df
