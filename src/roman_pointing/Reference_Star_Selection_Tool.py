import time
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import astropy.units as u
import astropy.coordinates as c
import pandas as pd
import requests

from roman_pointing.roman_observability import (
    get_target_coords,
    compute_roman_angles,
    compute_keepout,
)
from roman_pointing.utils import get_cache_dir

GRADE_COLUMNS = {
    (1,    "high"): "st_psfgrade_nfb1_high",
    (1,    "med"):  "st_psfgrade_nfb1_med",
    (3,    "high"): "st_psfgrade_specb3_high",
    (3,    "med"):  "st_psfgrade_specb3_med",
    (4,    "high"): "st_psfgrade_wfb4_high",
    (4,    "med"):  "st_psfgrade_wfb4_med",
    ("1w", "high"): "st_psfgrade_wfb4_high",
    ("1w", "med"):  "st_psfgrade_wfb4_med",
}

# Unique grade columns in definition order
ALL_GRADE_COLUMNS = list(dict.fromkeys(GRADE_COLUMNS.values()))

# Magnitude column name per band (derived at catalog-load time)
BAND_MAG_COL = {
    1:    "mag_v",
    3:    "mag_i",
    4:    "mag_i",
    "1w": "mag_v",
}

# Short band label used in display strings
BAND_LABEL = {
    1:    "V",
    3:    "I",
    4:    "I",
    "1w": "V",
}

REF_GRADES = ["A", "B", "C"]

# Star names to skip when building coordinates
SKIP_NAMES = {"-", "TBD", "?", ""}

SUN_MIN = 54        
SUN_MAX = 126         
MAX_PITCH_DIFF = 5.0  

# I-band magnitude pre-filter. None = disabled by default.
# Only applied for I-band modes (bands 3 and 4).
MAX_IMAG = None

SORT_MODES = (
    "valid_days",
    "closest_mag",
    "brightest",
    "faintest",
    "closest_pitch",
    "farthest_pitch",
)

SORT_MODE_LABELS = {
    "valid_days":     "Most valid days first",
    "closest_mag":    "Closest magnitude to science target",
    "brightest":      "Brightest first (ascending magnitude)",
    "faintest":       "Faintest first (descending magnitude)",
    "closest_pitch":  "Smallest pitch angle difference first",
    "farthest_pitch": "Largest pitch angle difference first",
}

# Sentinel value used in sort keys when a metric is unavailable
LARGE_SENTINEL = 1e9
CATALOG_URL = "https://corgidb.sioslab.com/fetch_refs.php"
DEFAULT_CACHE_PATH = Path(get_cache_dir()) / "ref_star_catalog_cache.csv"
MAX_CACHE_AGE_HOURS = 24.0

# Expected column order from the server endpoint
FETCH_COLUMNS = [
    "st_name",
    "main_id",
    "ra",
    "dec",
    "spectype",
    "sy_vmag",
    "sy_imag",
    "sy_dist",
    "sy_plx",
    "sy_pmra",
    "sy_pmdec",
    "st_radv",
    "st_psfgrade_nfb1_high",   
    "st_psfgrade_nfb1_med",    
    "st_psfgrade_specb3_high",
    "st_psfgrade_specb3_med",  
    "st_psfgrade_wfb4_high",   
    "st_psfgrade_wfb4_med",    
    "st_uddv",
    "st_uddi",
    "st_uddmeas",
    "st_lddmeas",
]

# Diameter columns — coerced to float; filled with NaN if absent from server
DIAMETER_COLS = ["st_uddv", "st_uddi", "st_uddmeas", "st_lddmeas"]


def safe_float(value):
    """Convert a value to float, returning None if it is missing or NaN."""
    if value is None:
        return None
    try:
        result = float(value)
        return None if np.isnan(result) else result
    except (TypeError, ValueError):
        return None


def get_ref_star_diameter(ref_row, band):
    """Return the uniform-disk diameter (mas) for a reference star catalog row.

    Tries columns in order: band-specific UDD (st_uddv or st_uddi),
    then st_uddmeas, then st_lddmeas.

    Args:
        ref_row (pandas.Series): One catalog row.
        band (int or str): Photometric band (1, '1w', 3, or 4).

    Returns:
        tuple: (diameter_mas, source_column), or (None, None) if not found.
    """
    primary = "st_uddv" if band in (1, "1w") else "st_uddi"
    for col in (primary, "st_uddmeas", "st_lddmeas"):
        val = safe_float(ref_row.get(col))
        if val is not None:
            return val, col
    return None, None


def get_science_target_diameter(sci_name, sci_coord, band):
    """Query VizieR JSDC v2 for the uniform-disk diameter of the science target.

    Performs a 5 arc-second cone search at the target's ICRS position.
    Prefers UDDV for V-band modes and UDDI for I-band modes, with a
    cross-band fallback if the primary value is absent.

    Args:
        sci_name (str): Target name, used only in log messages.
        sci_coord (astropy.coordinates.SkyCoord): Target position.
        band (int or str): Photometric band (1, '1w', 3, or 4).

    Returns:
        tuple: (diameter_mas, source_label), or (None, None) on failure.
    """
    try:
        from astropy.io.votable import parse_single_table

        icrs = sci_coord.icrs
        ra_deg = float(icrs.ra.deg)
        dec_deg = float(icrs.dec.deg)

        url = (
            "https://vizier.cds.unistra.fr/viz-bin/votable"
            f"?-source=II/346/jsdc_v2&-out=Name,UDDV,UDDI&-out.max=5"
            f"&-c={ra_deg:.6f},{dec_deg:+.6f}&-c.rs=5"
        )

        resp = requests.get(
            url,
            headers={"User-Agent": "RomanRefStarPicker/1.0"},
            timeout=20,
        )
        resp.raise_for_status()

        table = parse_single_table(BytesIO(resp.content)).to_table()

        if len(table) == 0:
            print(f"  Warning: no JSDC v2 match within 5\" for '{sci_name}' — diameter unavailable.")
            return None, None

        row = table[0]

        def parse_vot_float(col):
            try:
                v = float(row[col])
                return None if (np.isnan(v) or v <= 0) else v
            except Exception:
                return None

        uddv = parse_vot_float("UDDV")
        uddi = parse_vot_float("UDDI")

        if band in (1, "1w"):
            val, src = (uddv, "UDDV (VizieR JSDC v2)") if uddv else (uddi, "UDDI (VizieR JSDC v2)")
        else:
            val, src = (uddi, "UDDI (VizieR JSDC v2)") if uddi else (uddv, "UDDV (VizieR JSDC v2)")

        if val is None:
            print(f"  Warning: JSDC v2 match found for '{sci_name}' but UDD columns are empty.")
            return None, None

        print(f"  Science target diameter: {val:.4f} mas  [{src}]")
        return val, src

    except Exception as exc:
        print(f"  Warning: VizieR query failed for '{sci_name}': {exc} — diameter unavailable.")
        return None, None


def make_sort_key(sort_mode):
    """Return a comparator key function for sorting reference star result dicts.

    Results are always sorted primarily by grade_rank (ascending), then by
    the chosen secondary metric. The sort is stable, so ties retain order.

    Args:
        sort_mode (str): One of the values in SORT_MODES.

    Returns:
        callable: Key function for use with list.sort().

    Raises:
        ValueError: If sort_mode is not recognised.
    """
    if sort_mode == "valid_days":
        def key(ref):
            return (ref["grade_rank"], -ref["n_valid_days"])

    elif sort_mode == "closest_mag":
        def key(ref):
            diff = ref["mag_diff"] if ref["mag_diff"] is not None else LARGE_SENTINEL
            return (ref["grade_rank"], diff)

    elif sort_mode == "brightest":
        def key(ref):
            mag = ref["mag"] if ref["mag"] is not None else LARGE_SENTINEL
            return (ref["grade_rank"], mag)

    elif sort_mode == "faintest":
        def key(ref):
            mag = ref["mag"] if ref["mag"] is not None else -LARGE_SENTINEL
            return (ref["grade_rank"], -mag)

    elif sort_mode == "closest_pitch":
        def key(ref):
            pitch = ref["min_pitch_diff"] if ref["min_pitch_diff"] < 999 else LARGE_SENTINEL
            return (ref["grade_rank"], pitch)

    elif sort_mode == "farthest_pitch":
        def key(ref):
            pitch = ref["min_pitch_diff"] if ref["min_pitch_diff"] < 999 else -LARGE_SENTINEL
            return (ref["grade_rank"], -pitch)

    else:
        raise ValueError(f"Unknown sort_mode='{sort_mode}'. Valid options: {SORT_MODES}")

    return key


def cache_is_fresh(cache_path: Path, max_age_hours: float) -> bool:
    """Return True if the cache file exists and is younger than max_age_hours."""
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < max_age_hours * 3600


def fetch_catalog(url: str) -> pd.DataFrame:
    """Download the catalog from the given URL and return a raw DataFrame.

    The server returns a JSON array-of-arrays. Columns are mapped positionally
    to FETCH_COLUMNS; any additional server columns are treated as grade columns
    and named from ALL_GRADE_COLUMNS in order.

    Args:
        url (str): Catalog endpoint URL.

    Returns:
        pandas.DataFrame: Raw catalog with string-typed columns.

    Raises:
        RuntimeError: On network error, JSON decode failure, or empty response.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "RomanRefStarPicker/1.0"},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"HTTP fetch failed for {url!r}: {exc}") from exc

    try:
        raw = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"JSON decode failed for {url!r}: {exc}") from exc

    if not raw:
        raise RuntimeError(f"Empty response from {url!r}")

    data = np.vstack(raw).transpose()
    n_cols = len(data)
    n_base = len(FETCH_COLUMNS)

    if n_cols <= n_base:
        col_names = list(FETCH_COLUMNS[:n_cols])
    else:
        extra_names = ALL_GRADE_COLUMNS[: n_cols - n_base]
        col_names = list(FETCH_COLUMNS) + extra_names

    df = pd.DataFrame({name: col for name, col in zip(col_names, data)})
    print(f"  Catalog columns from server ({len(df.columns)}): {list(df.columns)}")
    return df


def coerce_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns, ensure all grade and diameter columns exist.

    Also derives sy_dist from parallax where distance is missing, and creates
    convenience mag_v / mag_i aliases.

    Args:
        df (pandas.DataFrame): Raw catalog DataFrame from fetch_catalog.

    Returns:
        pandas.DataFrame: Cleaned and enriched catalog.
    """
    numeric_cols = (
        "ra", "dec", "sy_vmag", "sy_imag", "sy_dist",
        "sy_plx", "sy_pmra", "sy_pmdec", "st_radv",
        "st_uddv", "st_uddi", "st_uddmeas", "st_lddmeas",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ALL_GRADE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in DIAMETER_COLS:
        if col not in df.columns:
            df[col] = np.nan

    if "sy_dist" in df.columns and "sy_plx" in df.columns:
        missing = df["sy_dist"].isna() & df["sy_plx"].notna() & (df["sy_plx"] > 0)
        n_derived = int(missing.sum())
        if n_derived:
            from astropy.coordinates import Distance
            df.loc[missing, "sy_dist"] = Distance(
                parallax=df.loc[missing, "sy_plx"].values * u.mas
            ).pc
            print(f"  Derived sy_dist from parallax for {n_derived} star(s).")

    df["mag_v"] = df.get("sy_vmag")
    df["mag_i"] = df.get("sy_imag")
    return df


def load_catalog(
    engine=None,
    url: str = CATALOG_URL,
    cache_path=None,
    max_cache_age_hours: float = MAX_CACHE_AGE_HOURS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load the reference star catalog, using a disk cache to avoid repeat fetches.

    Cache strategy:
        1. Fresh cache exists == return it immediately.
        2. Cache stale or missing fetch from URL, save, return.
        3. Fetch fails but stale cache exists → warn and use stale data.
        4. Fetch fails and no cache → raise RuntimeError.

    Args:
        engine: Ignored (kept for backwards compatibility).
        url (str): Catalog endpoint. Defaults to CATALOG_URL.
        cache_path: Path override for the on-disk CSV cache.
        max_cache_age_hours (float): Hours before the cache is considered stale.
        force_refresh (bool): Always fetch, even if the cache is current.

    Returns:
        pandas.DataFrame: One row per reference star.

    Raises:
        RuntimeError: If the fetch fails and no cache is available.
    """
    resolved_cache = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH

    if not force_refresh and cache_is_fresh(resolved_cache, max_cache_age_hours):
        print(f"Loading catalog from cache ({resolved_cache.name})...")
        df = pd.read_csv(resolved_cache, low_memory=False)
        print(f"Catalog loaded: {len(df)} reference star(s).")
        return df

    fetch_error = None
    print(f"Fetching catalog from {url} ...")
    try:
        df = fetch_catalog(url)
        df = coerce_catalog(df)
        resolved_cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(resolved_cache, index=False)
        print(f"  Catalog cached → {resolved_cache}")
        print(f"Catalog loaded: {len(df)} reference star(s).")
        return df
    except Exception as exc:
        fetch_error = exc
        print(f"  Fetch failed: {exc}")

    if resolved_cache.exists():
        warnings.warn(
            f"Live fetch failed ({fetch_error}). Using stale cache: {resolved_cache}",
            UserWarning,
            stacklevel=2,
        )
        df = pd.read_csv(resolved_cache, low_memory=False)
        print(f"Catalog loaded from stale cache: {len(df)} reference star(s).")
        return df

    raise RuntimeError(
        f"Could not load catalog: fetch failed ({fetch_error}) "
        f"and no cache exists at {resolved_cache}."
    )

def get_science_mag(sci_name, band, catalog=None, engine=None):
    """Look up the science target magnitude from the catalog or corgidb star endpoint.
    Args:
        sci_name (str): SIMBAD-resolvable science target name, e.g. '47 Uma'.
        band (int or str): Photometric band — 1 (V-band NFB), '1w' (V-band
            Wide FOV), 3 (I-band Spec), or 4 (I-band Wide FOV).
        catalog (pandas.DataFrame, optional): Loaded reference star catalog
            from load_catalog(). If None, the catalog lookup is skipped and
            fetch_star.php is queried directly.
        engine: Ignored. Kept for backwards compatibility.

    Returns:
        float or None: Magnitude value in the appropriate band, or None if
            not found in either the catalog or the star endpoint.
    """
    mag_col = "sy_vmag" if band in (1, "1w") else "sy_imag"
    band_label = BAND_LABEL.get(band, "?")

    # 1) Try the ref catalog first (works if science target is also a ref star)
    if catalog is not None:
        sci_norm = sci_name.strip().lower().lstrip("* ")
        mask = (
            catalog["main_id"].astype(str).str.strip().str.lstrip("* ").str.lower() == sci_norm
        ) | (
            catalog["st_name"].astype(str).str.strip().str.lstrip("* ").str.lower() == sci_norm
        )
        match = catalog[mask]
        if not match.empty:
            val = safe_float(match.iloc[0].get(mag_col))
            if val is not None:
                print(f"  Science target {band_label}-band mag: {val:.2f} (from catalog)")
                return val

    # 2) Fall back to fetch_star.php (the individual-star endpoint)
    try:
        import requests, numpy as np, pandas as pd
        url = "https://corgidb.sioslab.com/fetch_star.php"
        resp = requests.get(
            url,
            headers={"User-Agent": "RomanRefStarPicker/1.0"},
            params={"st_name": sci_name},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
        if raw:
            data = np.vstack(raw).transpose()
            star_cols = [
                "st_name", "main_id", "ra", "dec", "spectype",
                "sy_vmag", "sy_imag", "sy_dist", "sy_plx",
                "sy_pmra", "sy_pmdec", "st_radv",
            ]
            df = pd.DataFrame(
                {name: col for name, col in zip(star_cols, data)}
            )
            if not df.empty:
                val = safe_float(df.iloc[0].get(mag_col))
                if val is not None:
                    print(f"  Science target {band_label}-band mag: {val:.2f} (from fetch_star.php)")
                    return val
    except Exception as exc:
        print(f"  Warning: fetch_star.php lookup failed for '{sci_name}': {exc}")

    print(f"  Science target {band_label}-band mag not found — will sort brightest-first.")
    return None


def build_skycoord(star):
    """Build a BarycentricMeanEcliptic SkyCoord from a catalog row or dict.

    Includes proper motion, parallax/distance, and radial velocity when
    available. Constructed in ICRS J2000 then transformed to
    BarycentricMeanEcliptic.

    Args:
        star (dict or pandas.Series): Must contain 'ra' and 'dec' (degrees,
            J2000). Optional: 'sy_plx' (mas), 'sy_dist' (pc), 'sy_pmra'
            (mas/yr), 'sy_pmdec' (mas/yr), 'st_radv' (km/s).

    Returns:
        astropy.coordinates.SkyCoord: Position in BarycentricMeanEcliptic.
    """ 
    def get_field(key, fallback=None):
        raw = star[key] if isinstance(star, dict) else star.get(key, fallback)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        return float(raw)

    kwargs = dict(
        ra=get_field("ra") * u.degree,
        dec=get_field("dec") * u.degree,
        frame="icrs",
        equinox="J2000",
        obstime="J2000",
    )

    if get_field("sy_plx"):
        kwargs["distance"] = c.Distance(parallax=get_field("sy_plx") * u.mas)
    elif get_field("sy_dist"):
        kwargs["distance"] = get_field("sy_dist") * u.parsec

    if get_field("sy_pmra"):
        kwargs["pm_ra_cosdec"] = get_field("sy_pmra") * u.mas / u.yr
    if get_field("sy_pmdec"):
        kwargs["pm_dec"] = get_field("sy_pmdec") * u.mas / u.yr
    if get_field("st_radv"):
        kwargs["radial_velocity"] = get_field("st_radv") * u.km / u.s

    return c.SkyCoord(**kwargs).transform_to(c.BarycentricMeanEcliptic)



def get_observable_windows(times, keepout_array):
    """Extract contiguous observable windows from a boolean keepout array.

    Args:
        times (astropy.time.Time): Time array matching keepout_array.
        keepout_array (numpy.ndarray): Boolean array; True = observable.

    Returns:
        list of tuple: Each tuple is
            (start_time, end_time, start_str, end_str, duration_days).
    """
    windows = []
    in_window = False
    start_idx = 0

    for i, observable in enumerate(keepout_array):
        if observable and not in_window:
            in_window = True
            start_idx = i
        elif not observable and in_window:
            in_window = False
            windows.append((
                times[start_idx],
                times[i - 1],
                times[start_idx].iso.split("T")[0],
                times[i - 1].iso.split("T")[0],
                times[i - 1].mjd - times[start_idx].mjd,
            ))

    if in_window:
        windows.append((
            times[start_idx],
            times[-1],
            times[start_idx].iso.split("T")[0],
            times[-1].iso.split("T")[0],
            times[-1].mjd - times[start_idx].mjd,
        ))

    return windows


def check_ref_in_window(ref_coord, win_start, win_end, sci_pitch_in_window,
                        max_pitch_diff: float = MAX_PITCH_DIFF):
    """Check solar angle and pitch angle constraints for one reference star over a window.

    A day is valid if the reference star's solar angle is within [SUN_MIN,
    SUN_MAX] degrees AND the pitch difference to the science target is less
    than max_pitch_diff degrees.

    Args:
        ref_coord (astropy.coordinates.SkyCoord): Reference star position.
        win_start (astropy.time.Time): Window start.
        win_end (astropy.time.Time): Window end.
        sci_pitch_in_window (numpy.ndarray): Science target pitch angles (deg)
            for each day in the window.
        max_pitch_diff (float): Maximum allowed pitch difference in degrees.

    Returns:
        tuple: (passes, n_valid_days, min_pitch_diff, pitch_diff_series, valid_mask)
    """
    duration_days = win_end.mjd - win_start.mjd
    if duration_days <= 0:
        return False, 0, 999.0, np.array([]), np.array([], dtype=bool)

    start_str = win_start.isot if hasattr(win_start, "isot") else str(win_start)

    _, ref_sun_ang, _, ref_pitch = compute_roman_angles(
        ref_coord, start_str, duration_days, time_step=1.0
    )
    ref_sun_d = ref_sun_ang.to(u.degree).value
    ref_pitch_d = ref_pitch.to(u.degree).value

    min_len = min(len(sci_pitch_in_window), len(ref_sun_d), len(ref_pitch_d))
    solar_ok = (ref_sun_d[:min_len] > SUN_MIN) & (ref_sun_d[:min_len] < SUN_MAX)
    pitch_diff = np.abs(sci_pitch_in_window[:min_len] - ref_pitch_d[:min_len])
    pitch_ok = pitch_diff < max_pitch_diff

    pitch_diff_series = pitch_diff.copy().astype(float)
    pitch_diff_series[~solar_ok] = np.nan

    valid_mask = solar_ok & pitch_ok
    n_valid = int(np.sum(valid_mask))

    if n_valid == 0:
        return False, 0, 999.0, pitch_diff_series, valid_mask

    return (
        True,
        n_valid,
        float(np.min(pitch_diff[valid_mask])),
        pitch_diff_series,
        valid_mask,
    )


def select_ref_star(
    sci_name,
    analysis_start,
    analysis_days,
    band,
    contrast,
    catalog,
    engine=None,
    time_step=1.0,
    allowed_grades=None,
    sort_mode="valid_days",
    max_pitch_diff: float = MAX_PITCH_DIFF,
    max_imag: float = MAX_IMAG,
):
    """Find valid reference stars for each observable window of a science target.

    Candidates are filtered by grade, optional I-band brightness, and per-day
    solar / pitch angle constraints. Passing stars are sorted first by grade
    (A > B > C) then by the chosen sort_mode.

    Args:
        sci_name (str): SIMBAD-resolvable science target name.
        analysis_start (str): ISO start datetime, e.g. '2027-01-01T00:00:00'.
        analysis_days (float): Total analysis span in days.
        band (int or str): 1 (V-band NFB), 3 (I-band Spec),
            4 (I-band Wide FOV), or '1w' (V-band Wide FOV).
        contrast (str): 'high' or 'med'.
        catalog (pandas.DataFrame): Loaded reference star catalog.
        engine: Ignored (backwards compatibility).
        time_step (float): Angle calculation resolution in days. Default 1.0.
        allowed_grades (list of str): Grade tiers to include, in priority order.
            Defaults to ['A', 'B', 'C'].
        sort_mode (str): Secondary sort criterion. One of SORT_MODES.
        max_pitch_diff (float): Maximum pitch angle difference in degrees.
        max_imag (float or None): Maximum I-band magnitude for candidates.
            Ignored for V-band modes. None disables the filter.

    Returns:
        dict: Keys include 'science_target', 'band', 'contrast', 'grade_column',
            'allowed_grades', 'sort_mode', 'effective_sort', 'sci_mag',
            'sci_diameter', 'sci_diameter_src', 'visibility_pct', 'sort_method',
            'max_pitch_diff', 'max_imag', and 'observable_windows' (list of
            window dicts). On error, an 'error' key replaces 'observable_windows'.

    Raises:
        ValueError: On invalid band/contrast, sort_mode, or grade combination.
    """
    key = (band, contrast.lower())
    if key not in GRADE_COLUMNS:
        valid = ", ".join(f"band={b!r} contrast={ct}" for b, ct in GRADE_COLUMNS)
        raise ValueError(
            f"Unknown (band={band!r}, contrast='{contrast}'). Valid: {valid}"
        )

    if sort_mode not in SORT_MODES:
        raise ValueError(f"Unknown sort_mode='{sort_mode}'. Valid: {SORT_MODES}")

    grade_col = GRADE_COLUMNS[key]
    mag_col = BAND_MAG_COL[band]
    band_label = BAND_LABEL[band]

    print(f"\nGrade column: {grade_col}  |  Mag column: {mag_col}  |  Max pitch diff: {max_pitch_diff} deg")

    candidates = catalog.copy()

    # Optional I-band brightness pre-filter (I-band modes only)
    band_uses_imag = band in (3, 4)
    if max_imag is not None and band_uses_imag and "sy_imag" in candidates.columns:
        before = len(candidates)
        imag_vals = pd.to_numeric(candidates["sy_imag"], errors="coerce")
        candidates = candidates[imag_vals.isna() | (imag_vals <= max_imag)].copy()
        print(f"  I-mag filter (sy_imag <= {max_imag}): {before - len(candidates)} removed, {len(candidates)} remaining.")
    elif max_imag is not None and not band_uses_imag:
        print(f"  I-mag filter requested but band={band!r} uses V-band — skipped.")

    # Locate the column that actually contains A/B/C grade values
    def is_usable_grade_col(col_name):
        if col_name not in candidates.columns:
            return False
        sample = candidates[col_name].dropna()
        return len(sample) > 0 and sample.astype(str).str.match(r"^[ABC]$").any()

    grade_source = None
    for candidate_col in ("grade", grade_col, "st_psfgrade"):
        if is_usable_grade_col(candidate_col):
            grade_source = candidate_col
            break
    if grade_source is None:
        for col in candidates.columns:
            if is_usable_grade_col(col):
                grade_source = col
                break
    if grade_source is None:
        raise ValueError(
            f"No usable grade column found (expected A/B/C values). "
            f"Available columns: {list(catalog.columns)}"
        )

    print(f"  Grade source column: '{grade_source}'")
    candidates["grade"] = candidates[grade_source].astype(str).str.strip()

    active_grades = [g for g in (allowed_grades or REF_GRADES) if g in REF_GRADES]
    if not active_grades:
        raise ValueError(
            f"allowed_grades={allowed_grades!r} contains no valid values. Choose from {REF_GRADES}."
        )

    grade_rank_map = {g: i for i, g in enumerate(active_grades)}
    candidates = candidates[candidates["grade"].isin(active_grades)].copy()
    candidates = candidates.dropna(subset=[mag_col])
    candidates["grade_rank"] = candidates["grade"].map(grade_rank_map).fillna(99).astype(int)

    print(f"  Grade filter {active_grades}: {len(candidates)} candidate(s) remaining.")

    print(f"Querying SIMBAD for '{sci_name}'...")
    coords = get_target_coords([sci_name])
    if sci_name not in coords:
        return {"error": f"Science target '{sci_name}' not found in SIMBAD."}
    sci_coord = coords[sci_name]
    print(f"  Found '{sci_name}'.")

    print(f"Looking up {band_label}-band magnitude for '{sci_name}'...")
    sci_mag = get_science_mag(sci_name, band, catalog=catalog)

    print(f"Looking up diameter for '{sci_name}' from VizieR JSDC v2...")
    sci_diameter, sci_diameter_src = get_science_target_diameter(sci_name, sci_coord, band)

    if sort_mode == "closest_mag" and sci_mag is None:
        effective_sort = "brightest"
        sort_method = (
            f"grade ({'>'.join(active_grades)}) then brightest {band_label} "
            f"(closest_mag requested but science target magnitude not found)"
        )
    else:
        effective_sort = sort_mode
        sort_method = f"grade ({'>'.join(active_grades)}) then {SORT_MODE_LABELS[sort_mode].lower()}"

    print(f"\nComputing Roman visibility for '{sci_name}' over {analysis_days:.0f} days...")
    times, keepout, _ = compute_keepout(
        {sci_name: sci_coord}, analysis_start, analysis_days, time_step
    )
    sci_keepout = keepout[sci_name]
    visibility_pct = (np.sum(sci_keepout) / len(sci_keepout)) * 100
    print(f"  Observable {visibility_pct:.1f}% of the time.")

    windows = get_observable_windows(times, sci_keepout)
    if not windows:
        return {
            "science_target": sci_name,
            "band": band,
            "contrast": contrast,
            "grade_column": grade_col,
            "allowed_grades": active_grades,
            "sort_mode": sort_mode,
            "effective_sort": effective_sort,
            "sci_mag": sci_mag,
            "max_pitch_diff": max_pitch_diff,
            "max_imag": max_imag,
            "error": f"'{sci_name}' is never observable by Roman during this period.",
        }

    print(f"\nFound {len(windows)} observable window(s):")
    for i, (_, _, ws, we, wd) in enumerate(windows):
        print(f"  Window {i + 1}: {ws} to {we} ({wd:.1f} days)")

    _, _, _, sci_pitch_full = compute_roman_angles(
        sci_coord, analysis_start, analysis_days, time_step
    )
    sci_pitch_vals = sci_pitch_full.to(u.degree).value

    print("Building reference star coordinates...")
    ref_coords = {}
    for _, ref in candidates.iterrows():
        name = ref["main_id"]
        if not isinstance(name, str) or name.strip() in SKIP_NAMES:
            continue
        try:
            ref_coords[name] = build_skycoord(ref)
        except Exception as exc:
            print(f"  Warning: could not build coord for '{name}': {exc}")
    print(f"  Built coordinates for {len(ref_coords)} star(s).")

    sort_key = make_sort_key(effective_sort)
    results = []
    print(f"\nSort: {sort_method}")
    print("Searching for reference stars in each window...")

    for win_idx, (win_start, win_end, ws, we, wd) in enumerate(windows):
        print(f"\nWindow {win_idx + 1}: {ws} to {we} ({wd:.1f} days)")

        win_start_idx = int((win_start.mjd - times[0].mjd) / time_step)
        win_end_idx = int((win_end.mjd - times[0].mjd) / time_step)
        sci_pitch_win = sci_pitch_vals[win_start_idx: win_end_idx + 1]

        n_days_win = int(round(win_end.mjd - win_start.mjd)) + 1
        dates = [
            (win_start + i * u.day).to_value("iso", subfmt="date")
            for i in range(n_days_win)
        ]

        valid_refs = []
        pitch_series = {}

        for _, ref in candidates.iterrows():
            ref_name = ref["main_id"]
            if ref_name not in ref_coords:
                continue

            passes, n_days, min_pitch, pd_series, valid_mask = check_ref_in_window(
                ref_coords[ref_name], win_start, win_end, sci_pitch_win,
                max_pitch_diff=max_pitch_diff,
            )
            pitch_series[ref_name] = pd_series

            if not passes:
                continue

            valid_date_strs = [
                (win_start + int(d) * u.day).to_value("iso", subfmt="date")
                for d in np.where(valid_mask)[0]
            ]

            ref_mag = safe_float(ref.get(mag_col))
            mag_diff = (
                abs(ref_mag - sci_mag)
                if effective_sort == "closest_mag" and sci_mag is not None and ref_mag is not None
                else None
            )

            ref_diameter, ref_diameter_src = get_ref_star_diameter(ref, band)

            valid_refs.append({
                "reference_star": ref_name,
                "grade": ref["grade"],
                "grade_rank": int(ref["grade_rank"]),
                "mag": ref_mag,
                "mag_diff": mag_diff,
                "n_valid_days": n_days,
                "min_pitch_diff": min_pitch,
                "valid_dates": valid_date_strs,
                "diameter_mas": ref_diameter,
                "diameter_src": ref_diameter_src,
            })

        valid_refs.sort(key=sort_key)

        avail_df = pd.DataFrame(
            {r["reference_star"]: [d in set(r["valid_dates"]) for d in dates] for r in valid_refs},
            index=dates,
        )
        avail_df.index.name = "date"

        pitch_data = {}
        for r in valid_refs:
            name = r["reference_star"]
            series = pitch_series.get(name, np.array([]))
            vals = list(series[: len(dates)]) + [np.nan] * (len(dates) - len(series))
            pitch_data[name] = vals

        pitch_df = pd.DataFrame(pitch_data, index=dates)
        pitch_df.index.name = "date"

        print(f"  Found {len(valid_refs)} valid reference star(s).")

        results.append({
            "start": ws,
            "end": we,
            "duration_days": wd,
            "valid_refs": valid_refs,
            "best_ref": valid_refs[0] if valid_refs else None,
            "pitch_df": pitch_df,
            "avail_df": avail_df,
        })

    return {
        "science_target": sci_name,
        "band": band,
        "contrast": contrast,
        "grade_column": grade_col,
        "allowed_grades": active_grades,
        "sort_mode": sort_mode,
        "effective_sort": effective_sort,
        "sci_mag": sci_mag,
        "sci_diameter": sci_diameter,
        "sci_diameter_src": sci_diameter_src,
        "visibility_pct": visibility_pct,
        "sort_method": sort_method,
        "max_pitch_diff": max_pitch_diff,
        "max_imag": max_imag,
        "observable_windows": results,
    }


if __name__ == "__main__":
    print("ReferenceStarPicker\n")

    SCIENCE_TARGET = "47 Uma"
    BAND = "1w"          # 1=NFB V-band, 1w=Wide FOV V-band, 3=Spec I-band, 4=Wide FOV I-band
    CONTRAST = "high"
    ANALYSIS_START = "2026-12-01T00:00:00"
    ANALYSIS_DAYS = 365
    ALLOWED_GRADES = ["A", "B", "C"]
    SORT_MODE = "closest_mag"
    MAX_PITCH = 5.0
    MAX_I_MAG = None  # e.g. 1.5 to filter faint I-band candidates (bands 3 & 4 only)

    catalog = load_catalog()
    print(f"\nCatalog ready: {len(catalog)} reference stars.\n")

    result = select_ref_star(
        SCIENCE_TARGET, ANALYSIS_START, ANALYSIS_DAYS,
        band=BAND, contrast=CONTRAST, catalog=catalog,
        allowed_grades=ALLOWED_GRADES, sort_mode=SORT_MODE,
        max_pitch_diff=MAX_PITCH, max_imag=MAX_I_MAG,
    )

    print("\n" + "=" * 60)
    print(f"RESULTS: {result.get('science_target')} Band {result.get('band')} {result.get('contrast')}")
    print(f"Allowed grades : {result.get('allowed_grades')}")
    print(f"Sort mode      : {result.get('sort_mode')}")
    print(f"Max pitch diff : {result.get('max_pitch_diff')} deg")
    print(f"Max I-mag      : {result.get('max_imag')}")

    sci_mag = result.get("sci_mag")
    print(f"Science mag    : {f'{sci_mag:.2f}' if sci_mag else 'N/A'}")

    sci_diam = result.get("sci_diameter")
    sci_diam_src = result.get("sci_diameter_src") or ""
    print(f"Science diam   : {f'{sci_diam:.4f} mas [{sci_diam_src}]' if sci_diam else 'N/A'}")

    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print(f"Observable {result['visibility_pct']:.1f}% | {result['sort_method']}")
        print("=" * 60)
        for i, win in enumerate(result["observable_windows"]):
            print(f"\nWindow {i + 1}: {win['start']} -> {win['end']} ({win['duration_days']:.1f} days)")
            for ref in win["valid_refs"]:
                mag_str = (
                    f"Dmag={ref['mag_diff']:.2f}" if ref["mag_diff"] is not None
                    else f"mag={ref['mag']:.2f}" if ref["mag"] is not None
                    else "mag=N/A"
                )
                diam = ref.get("diameter_mas")
                diam_src = ref.get("diameter_src") or ""
                diam_str = f"diam={diam:.4f}mas [{diam_src}]" if diam else "diam=N/A"
                print(
                    f"  {ref['reference_star']:20s} grade={ref['grade']} "
                    f"{mag_str:14s} {ref['n_valid_days']:3d}d  "
                    f"pitch={ref['min_pitch_diff']:.4f}deg  {diam_str}"
                )