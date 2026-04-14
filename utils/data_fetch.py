"""
Data fetching and caching utilities.

Functions for downloading data from external sources (Barttorvik, Kaggle)
with retry logic, rate-limit backoff, and local CSV caching so re-runs
don't re-hit APIs unnecessarily.
"""

import io
import time
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def safe_request(url, retries=3, delay=1.5, timeout=20, **kwargs):
    """GET request with retry logic and rate-limit backoff.

    If the server is busy or the network blips, we wait longer each retry
    (delay * attempt) so we don't hammer the site.

    Parameters
    ----------
    url : str
        The URL to fetch.
    retries : int
        Maximum number of attempts before giving up.
    delay : float
        Base delay between retries (seconds). Multiplied by attempt number.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    requests.Response or None
        The response object, or None if all retries failed.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"  [WARN] Failed after {retries} attempts: {url}\n         {e}")
    return None


def load_or_fetch(cache_path, fetch_fn, force_refresh=False):
    """Load from CSV cache if it exists, otherwise call ``fetch_fn()`` to build it.

    Defaults to reading from cache so normal notebook re-runs are fast and
    don't re-hit external APIs.

    Parameters
    ----------
    cache_path : str or Path
        Where to read/write the cached CSV.
    fetch_fn : callable
        Zero-argument function that returns a DataFrame when cache misses.
    force_refresh : bool
        If True, rebuild the CSV even if the cache file exists.

    Returns
    -------
    pd.DataFrame or None
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, low_memory=False)
        print(f"  [CACHE] Loaded {len(df):,} rows from {cache_path.name}")
        return df
    df = fetch_fn()
    if df is not None and len(df) > 0:
        df.to_csv(cache_path, index=False)
        print(f"  [SAVED] {len(df):,} rows → {cache_path.name}")
    return df


# ---------------------------------------------------------------------------
# Barttorvik
# ---------------------------------------------------------------------------

def fetch_barttorvik_season(year, normalize_fn=None):
    """Download one season of Barttorvik team ratings.

    Barttorvik gives *adjusted* (pace- and opponent-adjusted) team metrics.
    Raw points per game depend on tempo; adjusted metrics answer: "How many
    points would this team score/allow per 100 possessions against an average
    D1 defense/offense?" That makes teams comparable across styles.

    Parameters
    ----------
    year : int
        The season year to fetch.
    normalize_fn : callable, optional
        Column-normalization function (e.g., ``normalize_bart_columns``).
        Applied after loading to standardize column names across seasons.

    Returns
    -------
    pd.DataFrame or None
    """
    url = f"http://barttorvik.com/{year}_team_results.csv"
    resp = safe_request(url, retries=3, delay=1.0)
    if resp is None:
        print(f"  [WARN] Could not fetch {year}")
        return None
    try:
        text = resp.content.decode('utf-8-sig').strip()
        df = pd.read_csv(io.StringIO(text))
        if len(df) == 0:
            return None

        # Barttorvik CSV format changed over years: first column is sometimes
        # rank (numeric), sometimes a label. We detect and align so 'bart_rank'
        # is always the first column.
        first_col = df.columns[0]
        if pd.to_numeric(df[first_col], errors='coerce').notna().mean() > 0.9:
            # Newer format: first column is numeric rank
            df = df.rename(columns={first_col: 'bart_rank'})
        else:
            # Older format: header has extra rank field but data rows don't —
            # shift column names right by one to realign
            new_cols = list(df.columns[1:]) + ['__drop__']
            df.columns = new_cols
            df = df.drop(columns='__drop__', errors='ignore')
            df.insert(0, 'bart_rank', range(1, len(df) + 1))

        if normalize_fn is not None:
            df = normalize_fn(df)

        df['season'] = year

        for col in ['adj_o', 'adj_d', 'barthag', 'adj_t', 'wab']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # adj_em = Adjusted Efficiency Margin. Offense minus defense: positive
        # means the team outscores opponents per 100 possessions. This single
        # number summarizes overall team strength.
        if 'adj_o' in df.columns and 'adj_d' in df.columns:
            df['adj_em'] = df['adj_o'] - df['adj_d']

        return df

    except Exception as e:
        print(f"  [WARN] Parse error for {year}: {e}")
        return None


def fetch_all_barttorvik(seasons, normalize_fn=None):
    """Download all requested seasons of Barttorvik data.

    Parameters
    ----------
    seasons : iterable of int
        Season years to fetch.
    normalize_fn : callable, optional
        Passed through to ``fetch_barttorvik_season``.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all seasons.
    """
    from tqdm.auto import tqdm

    frames = []
    for year in tqdm(seasons, desc="Barttorvik seasons"):
        df = fetch_barttorvik_season(year, normalize_fn=normalize_fn)
        if df is not None:
            frames.append(df)
        time.sleep(0.3)
    if not frames:
        raise RuntimeError("Could not download any Barttorvik data.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------

def try_kaggle_download(competition, kaggle_dir):
    """Attempt to download + unzip competition files via Kaggle API.

    Silently skips if credentials are not configured.

    Parameters
    ----------
    competition : str
        Kaggle competition slug (e.g., ``"march-machine-learning-mania-2026"``).
    kaggle_dir : str or Path
        Directory to download files into.
    """
    try:
        import kaggle
        import zipfile

        kaggle_dir = Path(kaggle_dir)
        kaggle.api.authenticate()
        print("Kaggle API authenticated. Downloading competition files...")
        kaggle.api.competition_download_files(
            competition,
            path=str(kaggle_dir),
            quiet=False
        )

        zip_path = kaggle_dir / f"{competition}.zip"
        if zip_path.exists():
            print(f"Unzipping {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(kaggle_dir)
            print("Unzip complete.")
        else:
            print("No ZIP file found after download; assuming files already extracted.")

        print("Download complete.")
    except Exception as e:
        print(f"Kaggle API download skipped ({e}).")
        print(f"Place CSV files manually in: {Path(kaggle_dir).resolve()}")


def load_kaggle_file(filename, kaggle_dir, required=True):
    """Load a single CSV from the Kaggle data directory.

    Parameters
    ----------
    filename : str
        CSV filename (e.g., ``"MNCAATourneySeeds.csv"``).
    kaggle_dir : str or Path
        Directory containing Kaggle CSVs.
    required : bool
        If True, raise FileNotFoundError when file is missing.

    Returns
    -------
    pd.DataFrame or None
    """
    path = Path(kaggle_dir) / filename
    if not path.exists():
        msg = f"Missing: {path}\nPlace the Kaggle files in {Path(kaggle_dir).resolve()}"
        if required:
            raise FileNotFoundError(msg)
        else:
            print(f"  [SKIP] {filename} not found.")
            return None
    df = pd.read_csv(path, low_memory=False)
    print(f"  {filename}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df
