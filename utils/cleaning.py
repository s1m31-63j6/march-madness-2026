"""
Data cleaning and standardization utilities.

Functions for text normalization, column renaming, team name crosswalks,
seed parsing, and categorical standardization. These operate on raw data
before any feature engineering or modeling.
"""

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


# ---------------------------------------------------------------------------
# Conference classification
# ---------------------------------------------------------------------------

# The "power 6" conferences that dominate NCAA tournament success.
# Adjust this set if conference realignment changes the landscape.
POWER6_CONFS = {"ACC", "B10", "B12", "P12", "SEC", "BE"}


def is_power6_conf(conf):
    """Return 1 if conference is in the power-6 set, else 0.

    Handles NaN and non-string values gracefully.

    Parameters
    ----------
    conf : str or any
        Conference abbreviation from Barttorvik data.

    Returns
    -------
    int
        1 if power-6 conference, 0 otherwise.
    """
    if isinstance(conf, str):
        return 1 if conf in POWER6_CONFS else 0
    return 0


# ---------------------------------------------------------------------------
# Barttorvik column normalization
# ---------------------------------------------------------------------------

def normalize_bart_columns(df):
    """Standardize Barttorvik CSV column names across seasons.

    Barttorvik changed header names over the years (e.g., "AdjOE" vs "adjoe").
    This maps all known variants to one consistent schema so the rest of the
    pipeline doesn't break across seasons.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Barttorvik season DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """
    rename_map = {
        # Team identity
        'team':       'team',
        'Team':       'team',
        'TEAM':       'team',
        'conf':       'conf',
        'Conf':       'conf',
        'Conf.':      'conf',
        # Core efficiency — title-case variants (older URLs / trank.php)
        'AdjOE':      'adj_o',
        'Adj OE':     'adj_o',
        'AdjDE':      'adj_d',
        'Adj DE':     'adj_d',
        'Barthag':    'barthag',
        'BARTHAG':    'barthag',
        'Adj T.':     'adj_t',
        'AdjTempo':   'adj_t',
        'Tempo':      'adj_t',
        # Core efficiency — lowercase variants (team_results.csv format)
        'adjoe':      'adj_o',
        'adjde':      'adj_d',
        'adjt':       'adj_t',
        # Four Factors – offense
        'EFG%':       'off_efg',
        'eFG%':       'off_efg',
        'TOR':        'off_to',
        'TO%':        'off_to',
        'ORB':        'off_or',
        'OR%':        'off_or',
        'FTR':        'off_ftr',
        # Four Factors – defense
        'EFGD%':      'def_efg',
        'eFGD%':      'def_efg',
        'TORD':       'def_to',
        'TOD%':       'def_to',
        'DRB':        'def_or',
        'DR%':        'def_or',
        'FTRD':       'def_ftr',
        # Shooting
        '2P%':        'fg2_pct',
        '2P%D':       'fg2d_pct',
        '3P%':        'fg3_pct',
        '3P%D':       'fg3d_pct',
        # Misc
        'WAB':        'wab',
        'Rk':         'bart_rank',
        'rank':       'bart_rank',
        'G':          'games',
        'Rec':        'record',
        'W-L':        'record',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


# ---------------------------------------------------------------------------
# Team name crosswalk (Barttorvik ↔ Kaggle)
# ---------------------------------------------------------------------------

# Known manual overrides: Barttorvik name → Kaggle TeamName.
# Values must match MTeams.csv TeamName exactly so kaggle_team_id resolves.
MANUAL_OVERRIDES = {
    # Major programs (exact Kaggle abbreviations)
    "UConn":              "Connecticut",
    "Ole Miss":           "Mississippi",
    "LSU":                "LSU",
    "USC":                "USC",
    "UCF":                "UCF",
    "UNLV":               "UNLV",
    "VCU":                "VCU",
    "SMU":                "SMU",
    "TCU":                "TCU",
    "BYU":                "BYU",
    "UAB":                "UAB",
    "UTSA":               "UT San Antonio",
    "UTEP":               "UTEP",
    "UNC":                "North Carolina",
    "NC State":           "NC State",
    "Miami FL":           "Miami FL",
    "Pitt":               "Pittsburgh",
    "Cal":                "California",
    "Saint Mary's":       "St Mary's CA",
    "St. John's":         "St John's",
    "St. Bonaventure":    "St Bonaventure",
    "Mt. St. Mary's":     "Mt St Mary's",
    "LIU":                "LIU Brooklyn",
    "SIU Edwardsville":   "SIUE",
    "SIUE":               "SIUE",
    "Loyola Chicago":     "Loyola-Chicago",
    "Loyola MD":          "Loyola MD",
    "CSUN":               "CS Northridge",
    "UCSB":               "UC Santa Barbara",
    "UC Santa Barbara":   "UC Santa Barbara",
    "Grambling St":       "Grambling",
    "Grambling St.":      "Grambling",
    "Morehead St":        "Morehead St",
    "Milwaukee":          "WI Milwaukee",
    "UW-Milwaukee":       "WI Milwaukee",
    "Abilene Christian":  "Abilene Chr",
    "Gardner-Webb":       "Gardner Webb",
    "Farleigh Dickinson": "F Dickinson",
    # Previously unmatched (Bart → Kaggle)
    "Albany":             "SUNY Albany",
    "American":           "American Univ",
    "Arkansas Pine Bluff": "Ark Pine Bluff",
    "Boston University":  "Boston Univ",
    "Cal St. Bakersfield": "CS Bakersfield",
    "Cal St. Fullerton":  "CS Fullerton",
    "Cal St. Northridge": "CS Northridge",
    "Central Connecticut": "Central Conn",
    "Central Michigan":   "C Michigan",
    "Charleston Southern": "Charleston So",
    "Coastal Carolina":   "Coastal Car",
    "Detroit Mercy":      "Detroit",
    "East Tennessee St.": "ETSU",
    "Eastern Illinois":   "E Illinois",
    "Eastern Kentucky":   "E Kentucky",
    "Eastern Michigan":   "E Michigan",
    "Eastern Washington": "E Washington",
    "FIU":                "Florida Intl",
    "Fairleigh Dickinson": "F Dickinson",
    "Florida Atlantic":   "FL Atlantic",
    "Florida Gulf Coast": "FGCU",
    "George Washington":  "G Washington",
    "Georgia Southern":   "Ga Southern",
    "Houston Christian":  "Houston Chr",
    "IU Indy":            "IUPUI",
    "Illinois Chicago":   "IL Chicago",
    "Kennesaw St.":       "Kennesaw",
    "Kent St.":           "Kent",
    "Little Rock":        "Ark Little Rock",
    "Louisiana Monroe":   "ULM",
    "Maryland Eastern Shore": "MD E Shore",
    "Middle Tennessee":   "MTSU",
    "Mississippi Valley St.": "MS Valley St",
    "Monmouth":           "Monmouth NJ",
    "Nebraska Omaha":     "NE Omaha",
    "North Carolina Central": "NC Central",
    "Northern Colorado":  "N Colorado",
    "Northern Illinois":  "N Illinois",
    "Northern Kentucky":  "N Kentucky",
    "Purdue Fort Wayne":  "PFW",
    "Queens":             "Queens NC",
    "Sacramento St.":     "CS Sacramento",
    "Saint Francis":      "St Francis PA",
    "Saint Joseph's":     "St Joseph's PA",
    "Saint Louis":        "St Louis",
    "Southeast Missouri St.": "SE Missouri St",
    "Southeastern Louisiana": "SE Louisiana",
    "Southern":           "Southern Univ",
    "Southern Illinois":  "S Illinois",
    "St. Thomas":         "St Thomas MN",
    "Stephen F. Austin":  "SF Austin",
    "Tennessee Martin":   "TN Martin",
    "Texas A&M Corpus Chris": "TAM C. Christi",
    "Texas Southern":     "TX Southern",
    "The Citadel":        "Citadel",
    "UMKC":               "Missouri KC",
    "UMass Lowell":       "MA Lowell",
    "UT Rio Grande Valley": "UTRGV",
    "Western Carolina":   "W Carolina",
    "Western Illinois":   "W Illinois",
    "Western Kentucky":   "WKU",
    "Western Michigan":   "W Michigan",
    # Bart sometimes uses same spelling as Kaggle
    "Kennesaw":           "Kennesaw",
    "LIU Brooklyn":       "LIU Brooklyn",
    "N Dakota St":        "N Dakota St",
    "North Dakota St":    "N Dakota St",
    "North Dakota St.":   "N Dakota St",
    "North Dakota State": "N Dakota St",
    "NDSU":               "N Dakota St",
    "St Louis":           "St Louis",
    "St. Louis":          "St Louis",
    "St Mary's CA":       "St Mary's CA",
    "Queens NC":          "Queens NC",
}


def build_crosswalk(bart_teams, kg_teams, manual_overrides=None, threshold=85):
    """Build a team name crosswalk mapping Barttorvik names to Kaggle TeamIDs.

    Uses manual overrides first (for known mismatches), then fuzzy matching
    as fallback. The crosswalk is essential because the two data sources use
    different naming conventions (e.g., "St. John's" vs "St John's").

    Parameters
    ----------
    bart_teams : pd.DataFrame
        Barttorvik data with a ``'team'`` column.
    kg_teams : pd.DataFrame
        Kaggle ``MTeams.csv`` with ``TeamName`` and ``TeamID``.
    manual_overrides : dict, optional
        ``{barttorvik_name: kaggle_name}`` overrides. Defaults to
        ``MANUAL_OVERRIDES`` if not provided.
    threshold : int
        Minimum fuzzy match score (0–100) to accept a match.

    Returns
    -------
    pd.DataFrame or None
    """
    from rapidfuzz import fuzz, process as rfprocess

    if kg_teams is None:
        print("[SKIP] Kaggle teams data not available — crosswalk skipped.")
        return None

    if manual_overrides is None:
        manual_overrides = MANUAL_OVERRIDES

    bart_names = bart_teams['team'].dropna().unique().tolist()
    kaggle_map = dict(zip(kg_teams['TeamName'], kg_teams['TeamID']))
    kaggle_names = list(kaggle_map.keys())

    rows = []
    unmatched = []

    for name in bart_names:
        # Manual override first
        if manual_overrides and name in manual_overrides:
            kaggle_name = manual_overrides[name]
            tid = kaggle_map.get(kaggle_name)
            rows.append({'bart_name': name, 'kaggle_name': kaggle_name,
                         'kaggle_team_id': tid, 'match_method': 'manual',
                         'match_score': 100})
            continue

        # Fuzzy match: token_sort_ratio ignores word order and small typos,
        # returns 0–100. We only accept matches at or above threshold so we
        # don't accidentally link wrong teams.
        result = rfprocess.extractOne(name, kaggle_names, scorer=fuzz.token_sort_ratio)
        if result is None:
            unmatched.append(name)
            continue
        match_name, score, _ = result
        if score >= threshold:
            tid = kaggle_map.get(match_name)
            rows.append({'bart_name': name, 'kaggle_name': match_name,
                         'kaggle_team_id': tid, 'match_method': 'fuzzy',
                         'match_score': score})
        else:
            unmatched.append(name)
            rows.append({'bart_name': name, 'kaggle_name': None,
                         'kaggle_team_id': None, 'match_method': 'unmatched',
                         'match_score': score})

    crosswalk = pd.DataFrame(rows)
    if unmatched:
        print(f"\n[WARN] {len(unmatched)} Barttorvik teams could not be matched:")
        for u in sorted(unmatched):
            print(f"  '{u}'")

    matched = crosswalk[crosswalk['match_method'] != 'unmatched']
    print(f"\nCrosswalk: {len(matched):,} / {len(bart_names):,} Barttorvik teams "
          f"matched to Kaggle IDs")
    print(f"  Manual overrides: {(crosswalk['match_method'] == 'manual').sum()}")
    print(f"  Fuzzy matches:    {(crosswalk['match_method'] == 'fuzzy').sum()}")
    return crosswalk
