import pandas as pd
import numpy as np


def impute_missing(
    df: pd.DataFrame,
    features=None,
    group_cols=None,
) -> pd.DataFrame:
    """
    Fill missing values without dropping any rows.

    Strategy per column:
      - Numeric    → group median; fall back to global median when a group is all-null
      - Categorical → group mode;  fall back to global mode when a group is all-null

    Group-based imputation is applied first. Any values still missing after the
    group pass (e.g. a group that was entirely null) are filled with the global
    statistic. Results are deterministic: mode ties are broken alphabetically
    (pandas .mode() returns sorted values).

    Parameters
    ----------
    df : DataFrame
    features : None | str | list
        Columns to impute.
        None  → all columns that contain at least one missing value
        str   → that single column
        list  → those specific columns
    group_cols : None | list
        Columns used to form groups for group-based imputation.
        None  → auto-detect: all '_clean' string columns in the DataFrame,
                which represent standardised categorical groups (hub, priority, etc.)
    """
    df = df.copy()

    # --- Resolve features to impute ---
    if features is None:
        target = [c for c in df.columns if df[c].isnull().any()]
    elif isinstance(features, str):
        target = [features]
    else:
        target = list(features)

    # --- Resolve grouping columns ---
    if group_cols is None:
        group_cols = [
            c for c in df.columns
            if c.endswith("_clean")
            and str(df[c].dtype) in ("object", "str", "string")
        ]

    str_dtypes = ("object", "str", "string")

    for col in target:
        is_categorical = str(df[col].dtype) in str_dtypes

        if is_categorical:
            # Global fallback: first (alphabetically) mode across all non-null values
            global_vals = df[col].dropna()
            global_fill = global_vals.mode().iloc[0] if len(global_vals) > 0 else None

            if group_cols:
                def _fill_cat(x, _gf=global_fill):
                    local_mode = x.dropna().mode()
                    fill = local_mode.iloc[0] if len(local_mode) > 0 else _gf
                    return x.fillna(fill)

                df[col] = df.groupby(group_cols)[col].transform(_fill_cat)

            # Global fallback for any remaining nulls
            if df[col].isnull().any() and global_fill is not None:
                df[col] = df[col].fillna(global_fill)

        else:
            # Global fallback: median across all non-null values
            global_fill = df[col].median()

            if group_cols:
                def _fill_num(x, _gf=global_fill):
                    local_median = x.median()  # NaN when entire group is null
                    fill = local_median if pd.notna(local_median) else _gf
                    return x.fillna(fill)

                df[col] = df.groupby(group_cols)[col].transform(_fill_num)

            # Global fallback for any remaining nulls
            if df[col].isnull().any():
                df[col] = df[col].fillna(global_fill)

    return df
