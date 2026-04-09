import pandas as pd
import numpy as np


def cap_outliers_iqr(
    df: pd.DataFrame,
    cols=None,
) -> pd.DataFrame:
    """
    Cap extreme values using Tukey's fence method (IQR * 1.5).

    Values below Q1 - 1.5*IQR are clipped to the lower fence.
    Values above Q3 + 1.5*IQR are clipped to the upper fence.
    All rows and columns are preserved; no values are dropped.

    Parameters
    ----------
    df : DataFrame
    cols : None | str | list
        Columns to process.
        None  → all numeric non-boolean columns
        str   → that single column
        list  → those specific columns
    """
    df = df.copy()

    # --- Identify boolean columns to exclude ---
    def _is_bool(col):
        s = df[col].dropna()
        return s.isin([0, 1]).all() and s.nunique() <= 2

    # --- Resolve target columns ---
    if cols is None:
        num_cols = df.select_dtypes(include="number").columns
        target = [c for c in num_cols if not _is_bool(c)]
    elif isinstance(cols, str):
        target = [cols]
    else:
        target = list(cols)

    for col in target:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower=lower, upper=upper)

    return df
