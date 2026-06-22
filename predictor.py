"""
Predict SOC and AGC for a farmer's field polygon via Earth Engine + XGBoost.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import os

import ee
import geemap
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_APP_DIR = Path(__file__).parent
SOC_MODEL_PATH = _APP_DIR / "models" / "soc_xgb_model.joblib"
AGC_MODEL_PATH = _APP_DIR / "models" / "agc_xgb_model.joblib"
EE_PROJECT = "soc-analysis-499103"

# Must match training feature order exactly
SOC_FEATURES = [
    "B2", "B3", "B4", "B8", "B11", "B12",
    "BSI", "NDVI", "SWIR_ratio", "RI", "Brightness",
    "DEM", "slope", "MAT", "MAT_seas", "MAP", "MAP_seas",
]
AGC_FEATURES = [
    "B4", "B8", "B11", "NDVI", "EVI", "NDMI", "NIRv",
    "DEM", "slope", "MAT", "MAT_seas", "MAP", "MAP_seas",
]
COV_COLS = ["DEM", "slope", "MAT", "MAT_seas", "MAP", "MAP_seas"]

_soc_model = None
_agc_model = None
_ee_ready = False


def _init():
    global _soc_model, _agc_model, _ee_ready
    if not _ee_ready:
        sa_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            import json as _json
            creds_dict = _json.loads(sa_json)
            credentials = ee.ServiceAccountCredentials(
                email=creds_dict["client_email"],
                key_data=sa_json,
            )
            ee.Initialize(credentials=credentials, project=EE_PROJECT)
        else:
            try:
                ee.Initialize(project=EE_PROJECT)
            except Exception:
                ee.Authenticate()
                ee.Initialize(project=EE_PROJECT)
        _ee_ready = True
    if _soc_model is None:
        _soc_model = joblib.load(SOC_MODEL_PATH)
    if _agc_model is None:
        _agc_model = joblib.load(AGC_MODEL_PATH)


def _mask_clouds(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    return image.updateMask(
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    )


def _worldcover_vegetation_mask() -> ee.Image:
    """ESA WorldCover 2021: exclude built-up (50), snow/ice (70), open water (80)."""
    wc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    return wc.neq(50).And(wc.neq(70)).And(wc.neq(80))


def _bare_soil_composite(geo: ee.Geometry,
                          date_start: str = "2019-01-01",
                          date_end: str   = "2024-01-01",
                          cloud_pct: int  = 30) -> ee.Image:
    """Bare-soil median composite over the given date range (NDVI<0.25, NBR2<0.07)."""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geo)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_clouds)
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    )
    composite = s2.median()
    ndvi = composite.normalizedDifference(["B8", "B4"])
    nbr2 = composite.normalizedDifference(["B11", "B12"])
    return composite.updateMask(
        ndvi.lt(0.25).And(nbr2.lt(0.07)).And(_worldcover_vegetation_mask())
    )


def _growing_composite(geo: ee.Geometry, hemisphere: str,
                        date_start: str = "2020-01-01",
                        date_end: str   = "2023-01-01",
                        use_month_filter: bool = True,
                        cloud_pct: int = 20) -> ee.Image:
    """Peak-vegetation composite over the given date range."""
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geo)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_clouds)
        .select(["B2", "B4", "B8", "B11"])
    )
    if use_month_filter:
        if hemisphere == "N":
            month_f = ee.Filter.calendarRange(5, 9, "month")
        else:
            month_f = ee.Filter.Or(
                ee.Filter.calendarRange(11, 12, "month"),
                ee.Filter.calendarRange(1, 3, "month"),
            )
        col = col.filter(month_f)
    composite = col.median()
    return composite.updateMask(_worldcover_vegetation_mask())


def _covariate_image() -> ee.Image:
    dem_raw = ee.Image("CGIAR/SRTM90_V4").select("elevation")
    dem = dem_raw.rename("DEM")
    slope = (
        ee.Terrain.slope(dem_raw.unmask(0)).rename("slope").updateMask(dem_raw.mask())
    )
    bio = ee.Image("WORLDCLIM/V1/BIO").select(
        ["bio01", "bio04", "bio12", "bio15"],
        ["MAT", "MAT_seas", "MAP", "MAP_seas"],
    )
    return dem.addBands(slope).addBands(bio)


def _sample(image: ee.Image, geo: ee.Geometry, n: int) -> pd.DataFrame:
    sampled = image.sample(
        region=geo, scale=20, numPixels=n, geometries=True, tileScale=4, seed=42
    )
    return geemap.ee_to_df(sampled)


def _extract_coords(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "longitude" in df.columns and "latitude" in df.columns:
        return df["longitude"].values, df["latitude"].values
    if ".geo" in df.columns:
        coords = df[".geo"].apply(lambda g: json.loads(g)["coordinates"])
        return coords.apply(lambda c: c[0]).values, coords.apply(lambda c: c[1]).values
    return np.full(len(df), np.nan), np.full(len(df), np.nan)


def _add_soc_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["BSI"] = ((d["B11"] + d["B4"]) - (d["B8"] + d["B2"])) / (
        (d["B11"] + d["B4"]) + (d["B8"] + d["B2"]) + 1e-6
    )
    d["NDVI"] = (d["B8"] - d["B4"]) / (d["B8"] + d["B4"] + 1e-6)
    d["SWIR_ratio"] = d["B11"] / (d["B12"] + 1e-6)
    d["RI"] = d["B4"] / (d["B2"] + 1e-6)
    d["Brightness"] = (d["B2"] + d["B3"] + d["B4"] + d["B8"]) / 4
    return d


def _add_agc_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["NDVI"] = (d["B8"] - d["B4"]) / (d["B8"] + d["B4"] + 1e-6)
    d["EVI"] = (
        2.5
        * (d["B8"] - d["B4"])
        / (d["B8"] + 6 * d["B4"] - 7.5 * d["B2"] + 1)
    )
    d["NDMI"] = (d["B8"] - d["B11"]) / (d["B8"] + d["B11"] + 1e-6)
    d["NIRv"] = d["NDVI"] * d["B8"] / 10000
    # NDBI: positive = built-up/bare, negative = vegetation
    d["NDBI"] = (d["B11"] - d["B8"]) / (d["B11"] + d["B8"] + 1e-6)
    return d


def _fill_covariates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in COV_COLS:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = d[c].fillna(d[c].median())
    return d


def _window(date_str: str, days: int = 30) -> tuple[str, str]:
    """Return (start, end) strings for a ±days window around a date string."""
    from datetime import datetime, timedelta
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (
        (d - timedelta(days=days)).strftime("%Y-%m-%d"),
        (d + timedelta(days=days)).strftime("%Y-%m-%d"),
    )


def preview_tiles(bounds: list[float], target_date: str = "") -> str:
    """
    Return an EE tile URL for the single Sentinel-2 scene (B4/B3/B2) closest in
    time to target_date within a ±45-day search window.  Clouds are kept as-is so
    the user can see the actual field condition on that date.
    If target_date is empty, falls back to the least-cloudy scene from 2022.
    """
    _init()
    west, south, east, north = bounds
    geo = ee.Geometry.Rectangle([west, south, east, north])

    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geo)

    if target_date:
        start, end = _window(target_date, 45)
        target_ms = ee.Date(target_date).millis()
        # Sort by absolute time distance to the target date; take the nearest scene
        s2 = (
            col.filterDate(start, end)
            .select(["B4", "B3", "B2"])
            .map(lambda img: img.set("timeDiff", img.date().millis().subtract(target_ms).abs()))
            .sort("timeDiff")
            .first()
        )
    else:
        s2 = (
            col.filterDate("2022-01-01", "2023-01-01")
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .select(["B4", "B3", "B2"])
            .first()
        )

    img = ee.Image(s2)
    # Resolve 2nd–98th percentile as real Python values so getMapId stretches correctly
    try:
        stats = img.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=geo,
            scale=200,
            bestEffort=True,
            maxPixels=1e7,
        ).getInfo()
        lo = [stats.get(f"B{b}_p2",  300) or 300  for b in ("4", "3", "2")]
        hi = [stats.get(f"B{b}_p98", 2500) or 2500 for b in ("4", "3", "2")]
    except Exception:
        lo, hi = [300, 300, 300], [2500, 2500, 2500]
    vis = {"bands": ["B4", "B3", "B2"], "min": lo, "max": hi, "gamma": 1.3}
    map_id = img.getMapId(vis)
    return map_id["tile_fetcher"].url_format


def _field_area_ha(polygon: list[list[float]]) -> float:
    lat0 = math.radians(sum(p[1] for p in polygon) / len(polygon))
    m_lat = 111320.0
    m_lng = 111320.0 * math.cos(lat0)
    n, area = len(polygon), 0.0
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = polygon[i][0] * m_lng, polygon[i][1] * m_lat
        x2, y2 = polygon[j][0] * m_lng, polygon[j][1] * m_lat
        area += x1 * y2 - x2 * y1
    return abs(area) / 2 / 10_000


MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def analyze_year(
    polygon: list[list[float]],
    year: int,
    progress: Callable[[str], None] = lambda _: None,
) -> dict:
    """Run predict_field for the 15th of every month in year, return time-series dict."""
    progress("Initializing Earth Engine and models…")
    _init()  # warm up once before spawning threads

    done_count = 0
    results: list[dict | None] = [None] * 12

    def run_month(m: int) -> tuple[int, dict | None]:
        date = f"{year}-{m:02d}-15"
        try:
            return m - 1, predict_field(polygon, date=date)
        except Exception as exc:
            logger.warning("Month %02d failed: %s", m, exc)
            return m - 1, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_month, m): m for m in range(1, 13)}
        for f in as_completed(futures):
            idx, r = f.result()
            results[idx] = r
            done_count += 1
            progress(f"Completed {done_count}/12 months…")

    area_ha = next((r["area_ha"] for r in results if r), None)
    return {
        "year": year,
        "months": MONTH_NAMES,
        "agc":     [r["agc_mean"] if r else None for r in results],
        "agc_std": [r["agc_std"]  if r else None for r in results],
        "soc":     [r["soc_mean"] if r else None for r in results],
        "soc_std": [r["soc_std"]  if r else None for r in results],
        "area_ha": area_ha,
    }


def predict_field(
    polygon: list[list[float]],
    progress: Callable[[str], None] = lambda _: None,
    date: str = "",
) -> dict:
    """
    polygon : list of [lng, lat] pairs
    date    : ISO date string (YYYY-MM-DD); empty → multi-year baseline
    Returns : dict with soc/agc stats and per-point data for the map.
    """
    soc_start, soc_end = _window(date, 30) if date else ("2019-01-01", "2024-01-01")
    agc_start, agc_end = _window(date, 45) if date else ("2020-01-01", "2023-01-01")
    if polygon[0] != polygon[-1]:
        polygon = polygon + [polygon[0]]

    progress("Initializing Earth Engine and loading models…")
    _init()

    geo = ee.Geometry.Polygon([[[lng, lat] for lng, lat in polygon]])
    mean_lat = sum(p[1] for p in polygon) / len(polygon)
    hemisphere = "N" if mean_lat >= 0 else "S"
    area_ha = _field_area_ha(polygon)
    n_samples = min(500, max(60, int(area_ha * 40)))

    progress(f"Field: {area_ha:.2f} ha  ·  Hemisphere: {'North' if hemisphere=='N' else 'South'}")

    # Relax cloud thresholds for specific-date queries: rely on per-pixel SCL masking
    soc_cloud = 60 if date else 30
    agc_cloud = 80 if date else 20

    progress(f"Building SOC bare-soil composite ({soc_start} → {soc_end})…")
    bare_img = _bare_soil_composite(geo, soc_start, soc_end, cloud_pct=soc_cloud)

    progress(f"Building AGC growing-season composite ({agc_start} → {agc_end})…")
    veg_img = _growing_composite(geo, hemisphere, agc_start, agc_end,
                                  use_month_filter=not bool(date), cloud_pct=agc_cloud)

    progress("Building terrain & climate covariate stack…")
    cov_img = _covariate_image()

    # ── SOC ──────────────────────────────────────────────────────────────
    progress(f"Sampling {n_samples} pixels for SOC (bare-soil)…")
    soc_bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    bare_with_cov = bare_img.addBands(cov_img)
    df_soc_raw = _sample(bare_with_cov, geo, n_samples)

    soc_points: list[dict] = []
    soc_mean = soc_median = soc_std = None

    if df_soc_raw.empty or not all(b in df_soc_raw.columns for b in soc_bands):
        progress("⚠ No bare-soil pixels found — SOC cannot be estimated for this field.")
    else:
        df_s = df_soc_raw.dropna(subset=soc_bands).copy()
        if len(df_s) > 0:
            df_s = _add_soc_features(df_s)
            df_s = _fill_covariates(df_s)
            preds = np.expm1(_soc_model.predict(df_s[SOC_FEATURES]))
            lngs, lats = _extract_coords(df_s)
            soc_mean = float(np.mean(preds))
            soc_median = float(np.median(preds))
            soc_std = float(np.std(preds))
            soc_points = [
                {"lat": float(lats[i]), "lng": float(lngs[i]), "soc": round(float(preds[i]), 1)}
                for i in range(len(preds))
                if not (np.isnan(lats[i]) or np.isnan(lngs[i]))
            ]
            progress(f"SOC: mean={soc_mean:.1f} g/kg  n={len(preds)} pixels")

    # ── AGC ──────────────────────────────────────────────────────────────
    progress(f"Sampling {n_samples} pixels for AGC (growing season)…")
    agc_s2_bands = ["B2", "B4", "B8", "B11"]
    veg_with_cov = veg_img.addBands(cov_img)
    df_agc_raw = _sample(veg_with_cov, geo, n_samples)

    agc_points: list[dict] = []
    agc_mean = agc_median = agc_std = None

    if df_agc_raw.empty or not all(b in df_agc_raw.columns for b in agc_s2_bands):
        progress("⚠ No valid growing-season pixels found — AGC cannot be estimated.")
    else:
        df_a = df_agc_raw.dropna(subset=agc_s2_bands).copy()
        if len(df_a) > 0:
            df_a = _add_agc_features(df_a)
            # Exclude buildings, solar panels, bare soil:
            #   NDVI > 0.2 → actual vegetation
            #   NDBI < 0   → NIR > SWIR, i.e. plant canopy not built-up surface
            df_a = df_a[(df_a["NDVI"] > 0.2) & (df_a["NDBI"] < 0)].copy()
        if len(df_a) == 0:
            progress("⚠ No vegetation pixels found — field may contain only buildings/solar panels.")
        else:
            df_a = _fill_covariates(df_a)
            preds = np.expm1(_agc_model.predict(df_a[AGC_FEATURES]))
            lngs, lats = _extract_coords(df_a)
            agc_mean = float(np.mean(preds))
            agc_median = float(np.median(preds))
            agc_std = float(np.std(preds))
            agc_points = [
                {"lat": float(lats[i]), "lng": float(lngs[i]), "agc": round(float(preds[i]), 1)}
                for i in range(len(preds))
                if not (np.isnan(lats[i]) or np.isnan(lngs[i]))
            ]
            progress(f"AGC: mean={agc_mean:.1f} MgC/ha  n={len(preds)} pixels")

    # ── Carbon stock estimates ────────────────────────────────────────────
    # SOC stock: SOC(g/kg) × bulk_density(1.3 g/cm³) × depth(0.3 m) × 10 → MgC/ha
    total_soc_MgC = soc_mean * 1.3 * 0.3 * 10 * area_ha if soc_mean else None
    total_agc_MgC = agc_mean * area_ha if agc_mean else None

    progress("Analysis complete!")

    return {
        "area_ha": round(area_ha, 3),
        "hemisphere": hemisphere,
        "date": date,
        "soc_mean":   round(soc_mean, 1)   if soc_mean   is not None else None,
        "soc_median": round(soc_median, 1) if soc_median is not None else None,
        "soc_std":    round(soc_std, 1)    if soc_std    is not None else None,
        "soc_n":      len(soc_points),
        "agc_mean":   round(agc_mean, 1)   if agc_mean   is not None else None,
        "agc_median": round(agc_median, 1) if agc_median is not None else None,
        "agc_std":    round(agc_std, 1)    if agc_std    is not None else None,
        "agc_n":      len(agc_points),
        "total_soc_MgC": round(total_soc_MgC, 1) if total_soc_MgC else None,
        "total_agc_MgC": round(total_agc_MgC, 1) if total_agc_MgC else None,
        "soc_points": soc_points,
        "agc_points": agc_points,
    }
