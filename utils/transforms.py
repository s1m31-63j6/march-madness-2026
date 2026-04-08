import pandas as pd
import numpy as np
from scipy import stats


def transform_skew(
    df: pd.DataFrame,
    features=None,
    suffix: str = "_skewfix",
) -> pd.DataFrame:
    """
    Reduce skew in numeric columns by selecting the best monotonic transformation.

    Transformations evaluated (lowest |skew| wins):
      'none'        — baseline, no change
      'log1p'       — log1p(x + shift); shift makes minimum = 0 for negatives/zeros
      'sqrt'        — sqrt(x + shift); same shift strategy
      'cbrt'        — cube root; handles negatives natively, no shift needed
      'yeo-johnson' — scipy.stats.yeojohnson; handles negatives natively, often best

    Tie-breaking: when two transforms produce equal |skew|, preference follows the
    order above (simpler / more interpretable transforms win ties).

    NaN values are preserved as NaN. Boolean columns are always skipped.

    Parameters
    ----------
    df : DataFrame
    features : None | str | list
        Columns to process.
        None  → all numeric non-boolean columns
        str   → that single column
        list  → those specific columns
    suffix : str
        Appended to each processed column name for the transformed column.
        Default '_skewfix'.
    """
    df = df.copy()

    # --- Identify boolean columns to exclude ---
    def _is_bool(col):
        s = df[col].dropna()
        return s.isin([0, 1]).all() and s.nunique() <= 2

    # --- Resolve feature list ---
    if features is None:
        num_cols = df.select_dtypes(include="number").columns
        target = [c for c in num_cols if not _is_bool(c)]
    elif isinstance(features, str):
        target = [features]
    else:
        target = list(features)

    # Tie-breaking order (lower index = higher priority when |skew| is equal)
    TRANSFORM_ORDER = ["none", "log1p", "sqrt", "cbrt", "yeo-johnson"]

    for col in target:
        series = df[col]
        notnull = series.dropna()

        if notnull.empty:
            df[col + suffix] = series
            continue

        # Shift so minimum = 0 (required by log1p and sqrt for non-negative inputs)
        shift = max(0.0, -notnull.min())

        # --- Evaluate each transformation on non-null values ---
        candidates = {}  # name → |skew|
        lam = None       # Yeo-Johnson lambda (fitted below)

        # Baseline
        candidates["none"] = abs(notnull.skew(skipna=True))

        # log1p (shift first so all values >= 0)
        try:
            candidates["log1p"] = abs(np.log1p(notnull + shift).skew(skipna=True))
        except Exception:
            pass

        # sqrt (shift first)
        try:
            candidates["sqrt"] = abs(np.sqrt(notnull + shift).skew(skipna=True))
        except Exception:
            pass

        # cbrt — handles negatives natively
        try:
            candidates["cbrt"] = abs(pd.Series(np.cbrt(notnull.values)).skew(skipna=True))
        except Exception:
            pass

        # Yeo-Johnson — fit lambda on non-null values, capture for later apply
        try:
            yj_vals, lam = stats.yeojohnson(notnull.values)
            candidates["yeo-johnson"] = abs(pd.Series(yj_vals).skew(skipna=True))
        except Exception:
            pass

        # --- Pick best: lowest |skew|, tie-break by TRANSFORM_ORDER priority ---
        best = min(
            candidates,
            key=lambda name: (round(candidates[name], 10), TRANSFORM_ORDER.index(name))
        )

        # --- Apply best transformation to full column (NaN stays NaN) ---
        if best == "none":
            df[col + suffix] = series

        elif best == "log1p":
            df[col + suffix] = np.log1p(series + shift)

        elif best == "sqrt":
            df[col + suffix] = np.sqrt(series + shift)

        elif best == "cbrt":
            df[col + suffix] = series.apply(lambda x: np.cbrt(x) if pd.notna(x) else np.nan)

        elif best == "yeo-johnson" and lam is not None:
            result = series.copy().astype(float)
            result[series.notna()] = stats.yeojohnson(notnull.values, lmbda=lam)
            df[col + suffix] = result

    return df
