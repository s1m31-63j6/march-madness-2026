import pandas as pd
import numpy as np


def bin_rare_categories(
    df: pd.DataFrame,
    cols=None,
    min_prop: float = 0.05,
    suffix: str = "_binned",
) -> pd.DataFrame:
    """
    Consolidate infrequent categories to reduce cardinality.

    Categories appearing in fewer than `min_prop` of rows are replaced with
    "Other". NaN values are preserved as NaN (never mapped to "Other"). A new
    column is created for each processed column; originals are never overwritten.
    All rows are preserved.

    Parameters
    ----------
    df : DataFrame
    cols : None | str | list
        Columns to process.
        None  → all categorical (low-cardinality string) columns in the DataFrame.
                High-cardinality columns such as IDs are automatically skipped.
        str   → that single column
        list  → those specific columns
    min_prop : float
        Minimum proportion (0–1) a category must appear to be kept as-is.
        Categories below this threshold are grouped into "Other".
        Default 0.05 (5%).
    suffix : str
        Appended to each processed column name for the new binned column.
        Default '_binned'.
    """
    df = df.copy()

    # --- Resolve which columns to process ---
    if cols is None:
        # Only auto-select low-cardinality string columns (true categoricals).
        # Columns where unique values exceed 5% of rows are likely IDs — skip them.
        str_cols = [c for c in df.columns if str(df[c].dtype) in ("object", "str", "string")]
        target_cols = [c for c in str_cols if df[c].nunique() / len(df) <= 0.02]
    elif isinstance(cols, str):
        target_cols = [cols]
    else:
        target_cols = list(cols)

    for col in target_cols:
        # Proportions calculated over all rows (NaN excluded by value_counts default)
        proportions = df[col].value_counts(normalize=True)
        keep = proportions[proportions >= min_prop].index

        # Replace rare categories with "Other"; NaN stays NaN
        df[col + suffix] = df[col].where(
            df[col].isin(keep) | df[col].isna(),
            other="Other"
        )

    return df
