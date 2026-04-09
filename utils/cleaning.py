import pandas as pd
import numpy as np
from difflib import get_close_matches


def wrangle_basic(df: pd.DataFrame, corrections: dict = None, min_prop: float = 0.05, fuzzy_cutoff: float = 0.75, max_cat_ratio: float = 0.02) -> pd.DataFrame:
    """
    Clean categorical text fields to eliminate data quality issues caused
    by inconsistent data entry.

    Works on any dataset — no column names are hardcoded.

    Strategy (applied per string column):
      1. Strip whitespace + lowercase
      2. Unify separators (space / hyphen / underscore → hyphen) ONLY when a
         column actually contains mixed separator types — avoids corrupting
         consistently snake_cased columns like failure_reason or delivery_note
      3. Auto-correct typos via fuzzy matching against the most frequent values
         — skipped on high-cardinality columns (e.g. IDs) to prevent false merges
      4. Apply optional user-provided corrections, which override everything
         — needed for abbreviations fuzzy matching can't infer (e.g. 'biz' → 'business')
         — also needed for errors that appear frequently enough to seem canonical

    Creates a '_clean' column for:
      - Low-cardinality columns where any value changed after normalization
        (catches consistently-formatted columns like delivery_zone that just need lowercase)
      - High-cardinality columns only when unique count decreases
        (limits _clean creation to genuine quality issues in ID-like columns)
    Rows are always preserved.

    Parameters
    ----------
    df : DataFrame
    corrections : dict, optional
        Explicit {normalized_bad: normalized_good} overrides applied last.
        Keys should be in normalized form (lowercase, hyphen-separated).
        Example: {'biz': 'business', 'res': 'residential', 'dnvr-east': 'denver-east'}
    min_prop : float
        A value must appear in at least this proportion of rows to be treated
        as a canonical reference for fuzzy matching. Default 0.05 (5%).
    fuzzy_cutoff : float
        Minimum string similarity (0–1) to accept a fuzzy match. Default 0.75.
    max_cat_ratio : float
        Columns where (unique values / total rows) exceeds this threshold are
        treated as ID-like and skip fuzzy matching. Default 0.05 (5%).
    """
    df = df.copy()
    corrections = corrections or {}

    str_cols = [c for c in df.columns if str(df[c].dtype) in ("object", "str", "string")]

    for col in str_cols:
        original = df[col]
        is_categorical = original.nunique() / len(df) <= max_cat_ratio

        # --- Step 1: strip + lowercase ---
        cleaned = original.str.strip().str.lower()

        # --- Step 2: separator normalization — only when mixed separators exist ---
        # Detects whether the column mixes spaces, hyphens, and underscores, which
        # indicates inconsistency rather than intentional naming convention.
        has_spaces     = cleaned.str.contains(r'\w \w', na=False).any()
        has_hyphens    = cleaned.str.contains(r'\w-\w', na=False).any()
        has_underscores = cleaned.str.contains(r'\w_\w', na=False).any()
        mixed_seps = sum([has_spaces, has_hyphens, has_underscores]) > 1

        if mixed_seps:
            cleaned = cleaned.str.replace(r"[\s_\-]+", "-", regex=True)

        # --- Step 3: fuzzy matching (low-cardinality columns only) ---
        if is_categorical:
            total = cleaned.notna().sum()
            counts = cleaned.value_counts()
            canonical = counts[counts / total >= min_prop].index.tolist()

            def resolve(val, _canonical=canonical, _corrections=corrections):
                if pd.isna(val):
                    return val
                if val in _corrections:
                    return _corrections[val]
                if val in _canonical:
                    return val
                matches = get_close_matches(val, _canonical, n=1, cutoff=fuzzy_cutoff)
                return matches[0] if matches else val

            cleaned = cleaned.map(resolve)
        else:
            if corrections:
                cleaned = cleaned.map(lambda x: corrections.get(x, x) if pd.notna(x) else x)

        # --- Decide whether to emit a _clean column ---
        # Low-cardinality: create _clean if any value changed (catches delivery_zone etc.)
        # High-cardinality: only create _clean if genuine deduplication occurred
        values_changed = not cleaned.fillna("__NA__").equals(original.fillna("__NA__"))
        unique_reduced = cleaned.nunique() < original.nunique()

        if (is_categorical and values_changed) or (not is_categorical and unique_reduced):
            df[f"{col}_clean"] = cleaned

    return df


def parse_seed(seed_str):
    """
    Extract numeric seed from tournament-style strings.

    Examples:
    - 'W01' -> 1
    - 'X16a' -> 16
    - 'Z11' -> 11

    Returns `np.nan` if the input is missing or no digits are found.
    """
    if pd.isna(seed_str):
        return np.nan

    import re

    m = re.search(r"(\d+)", str(seed_str))
    return int(m.group(1)) if m else np.nan
