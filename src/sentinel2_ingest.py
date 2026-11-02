"""Sentinel-2 L2A ingestion: builds a per-date, field-level NDVI/EVI/NDWI series.

Bands are read over the field bounding box from a STAC provider (public AWS earth_search
by default, or Copernicus). Cloud, shadow and cirrus pixels are removed using the Scene
Classification Layer (SCL), and the clear-pixel means give one row of indices per usable
scene.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pystac_client import Client

from .config_loader import FieldConfig, resolve_env_credentials
from .indices import compute_evi, compute_ndvi, compute_ndwi, to_reflectance

logger = logging.getLogger(__name__)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

# Sensible provider presets so the config only has to name a provider.
_PROVIDER_PRESETS = {
    "earth_search": {
        "stac_url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "assets": {"red": "red", "nir": "nir", "blue": "blue", "swir": "swir16"},
    },
    "copernicus": {
        "stac_url": "https://catalogue.dataspace.copernicus.eu/stac",
        "collection": "SENTINEL-2",
        "assets": {"red": "B04", "nir": "B08", "blue": "B02", "swir": "B11"},
    },
}


def active_s2_config(cfg: FieldConfig) -> dict:
    """
    Resolve the effective Sentinel-2 settings (stac_url / collection / asset
    keys) from whatever the config provides, falling back to the preset for the
    chosen provider. Keeps backward-compatibility with the older config shape
    (top-level stac_url/collection/bands).
    """
    s2 = cfg.raw.get("sentinel2", {})
    provider = s2.get("provider", "copernicus")
    preset = _PROVIDER_PRESETS.get(provider, _PROVIDER_PRESETS["copernicus"])
    sub = s2.get(provider, {})  # optional per-provider override block

    assets = sub.get("assets") or s2.get("assets") or s2.get("bands") or preset["assets"]
    return {
        "provider": provider,
        "stac_url": sub.get("stac_url") or (s2.get("stac_url") if provider == "copernicus" else None) or preset["stac_url"],
        "collection": sub.get("collection") or (s2.get("collection") if provider == "copernicus" else None) or preset["collection"],
        "assets": assets,
        "max_cloud_cover_pct": s2.get("max_cloud_cover_pct", 20),
        # cap how many scenes we actually READ bands for (each scene = 4 remote
        # band reads). Reading 50 large scenes can take an hour; ~12 evenly
        # spread across the season gives a fine NDVI series in a few minutes.
        "max_scenes": int(s2.get("max_scenes", 12)),
    }


def subsample_scenes(items: list[dict], max_scenes: int) -> list[dict]:
    """Keep at most `max_scenes`, evenly spread across time, so the NDVI series
    still spans the whole window but we read far fewer scenes."""
    if max_scenes <= 0 or len(items) <= max_scenes:
        return items
    ordered = sorted(items, key=lambda it: (it.get("properties", {}) or {}).get("datetime", ""))
    idx = np.linspace(0, len(ordered) - 1, max_scenes).round().astype(int)
    seen, picked = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            picked.append(ordered[i])
    return picked


def get_access_token(client_id: str, client_secret: str) -> str:
    """OAuth2 client-credentials flow for the Copernicus Data Space Ecosystem."""
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_sentinel2_scenes(cfg: FieldConfig, max_items: int = 50) -> list[dict]:
    """
    Query a STAC catalogue for Sentinel-2 L2A scenes intersecting the configured
    field bbox and date range, filtered by cloud cover. Works identically for a
    field in Iowa, Punjab, Andalusia, or anywhere else on Earth -- the bbox is
    the only geographic input.
    """
    s2 = active_s2_config(cfg)
    client = Client.open(s2["stac_url"])

    search = client.search(
        collections=[s2["collection"]],
        bbox=cfg.bbox,
        datetime=f"{cfg.start_date}/{cfg.end_date}",
        query={"eo:cloud_cover": {"lt": s2["max_cloud_cover_pct"]}},
        max_items=max_items,
    )
    items = list(search.items())
    logger.info("Found %d Sentinel-2 scenes (%s) for field '%s'",
                len(items), s2["provider"], cfg.name)
    return [item.to_dict() for item in items]


def compute_indices_for_scene(band_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Given the four bands (red, nir, blue, swir) as raw digital-number arrays on a
    COMMON grid, scale to reflectance and compute NDVI/EVI/NDWI per pixel.
    Used for per-pixel index maps; `run_sentinel2_ingestion` uses the lighter
    band-mean path below for the field-level time series.
    """
    red = to_reflectance(band_arrays["red"])
    nir = to_reflectance(band_arrays["nir"])
    blue = to_reflectance(band_arrays["blue"])
    swir = to_reflectance(band_arrays["swir"])
    return {
        "ndvi": compute_ndvi(nir, red),
        "evi": compute_evi(nir, red, blue),
        "ndwi": compute_ndwi(nir, swir),
    }


# Band reading (COG windowed read over the field bbox)
def _href_to_gdal_path(href: str) -> str:
    """Map a STAC asset href to a GDAL virtual path GDAL can open."""
    if href.startswith("s3://"):
        return "/vsis3/" + href[len("s3://"):]
    if href.startswith("http://") or href.startswith("https://"):
        return "/vsicurl/" + href
    return href  # already a local path or /vsi... path


def _gdal_env(provider: str, token: str | None) -> dict:
    """
    GDAL/rasterio environment for reading remote COGs.

    * earth_search -> public 'sentinel-cogs' bucket in us-west-2, no signing.
    * copernicus   -> if reading CDSE assets directly you'd pass a bearer token;
                      the openEO path is recommended instead (see below).
    """
    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.jp2",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        # auto-retry flaky COG reads instead of failing the scene outright
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "GDAL_HTTP_TIMEOUT": "30",
    }
    if provider == "earth_search":
        env.update({"AWS_NO_SIGN_REQUEST": "YES", "AWS_REGION": "us-west-2"})
    if token:
        env["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"
    return env


def read_band_mean_over_bbox(href: str, bbox: tuple[float, float, float, float]) -> float:
    """
    Read a single band COG, clip to the field bbox, and return the spatial mean
    of valid pixels (NaN if the window has no valid data). rasterio is imported
    lazily so the rest of the pipeline works without it installed.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    gdal_path = _href_to_gdal_path(href)
    with rasterio.open(gdal_path) as src:
        # bbox is (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        arr = src.read(1, window=window, boundless=True, fill_value=0).astype("float32")

    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        return float("nan")
    return float(arr[valid].mean())


def _read_band_array(href: str, bbox: tuple[float, float, float, float],
                     out_shape: tuple[int, int] | None = None, cap: int = 256):
    """
    Read a band COG clipped to the field bbox as a 2-D array. If out_shape is
    given, resample to it (so all bands of a scene land on ONE common pixel grid
    despite 10 m vs 20 m native resolutions); otherwise derive it from the
    window, capped to `cap` px per side to keep reads small and fast.
    Returns (array, out_shape).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(_href_to_gdal_path(href)) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        if out_shape is None:
            h, w = max(1, int(round(window.height))), max(1, int(round(window.width)))
            if max(h, w) > cap:
                s = cap / max(h, w)
                h, w = max(1, int(h * s)), max(1, int(w * s))
            out_shape = (h, w)
        arr = src.read(1, window=window, out_shape=out_shape, boundless=True,
                       fill_value=0, resampling=Resampling.nearest).astype("float32")
    return arr, out_shape


# Sentinel-2 Scene Classification (SCL) codes to KEEP (clear land/water/snow).
# Dropped: 0 no-data, 1 defective, 2 dark-shadow, 3 cloud-shadow, 8 cloud-med,
# 9 cloud-high, 10 thin-cirrus.
_SCL_KEEP = {4, 5, 6, 7, 11}  # 4 veg, 5 bare, 6 water, 7 unclassified, 11 snow


def read_scene_indices(item: dict, cfg: FieldConfig) -> dict | None:
    """
    For one STAC scene, read the bands over the field bbox on a common grid,
    MASK cloud/shadow/cirrus pixels using the SCL scene-classification band,
    then compute field-level NDVI/EVI/NDWI from the mean of the CLEAR pixels
    only. Returns None (scene skipped) if assets are missing or too few clear
    pixels remain over the field -- so a cloudy scene no longer corrupts NDVI.
    """
    s2 = active_s2_config(cfg)
    assets = item.get("assets", {})

    hrefs = {}
    for band, asset_key in s2["assets"].items():
        asset = assets.get(asset_key)
        if not asset or "href" not in asset:
            logger.warning("Scene %s missing asset '%s' -- skipping scene", item.get("id"), asset_key)
            return None
        hrefs[band] = asset["href"]

    # SCL (cloud mask) asset -- 'scl' on Earth Search, 'SCL' on Copernicus
    scl_href = None
    for k in ("scl", "SCL"):
        a = assets.get(k)
        if a and "href" in a:
            scl_href = a["href"]
            break

    try:
        red, shp = _read_band_array(hrefs["red"], cfg.bbox)      # red fixes the grid
        nir, _ = _read_band_array(hrefs["nir"], cfg.bbox, out_shape=shp)
        blue, _ = _read_band_array(hrefs["blue"], cfg.bbox, out_shape=shp)
        swir, _ = _read_band_array(hrefs["swir"], cfg.bbox, out_shape=shp)
        scl = None
        if scl_href:
            scl, _ = _read_band_array(scl_href, cfg.bbox, out_shape=shp)
    except Exception as e:
        logger.warning("Scene %s: band read failed (%s) -- skipping", item.get("id"), e)
        return None

    valid = (np.isfinite(red) & np.isfinite(nir) & np.isfinite(blue) & np.isfinite(swir)
             & (red > 0) & (nir > 0))
    masked = False
    if scl is not None:
        valid &= np.isin(scl.astype(int), list(_SCL_KEEP))
        masked = True

    frac = float(valid.mean()) if valid.size else 0.0
    if valid.sum() < 10 or frac < 0.05:
        logger.info("Scene %s: only %.0f%% clear pixels over field -- skipping (too cloudy).",
                    item.get("id"), 100 * frac)
        return None

    # mean of CLEAR pixels only -> reflectance -> indices
    red_m, nir_m = red[valid].mean() / 10000.0, nir[valid].mean() / 10000.0
    blue_m, swir_m = blue[valid].mean() / 10000.0, swir[valid].mean() / 10000.0
    eps = 1e-8
    ndvi = (nir_m - red_m) / (nir_m + red_m + eps)
    evi = float(np.clip(2.5 * (nir_m - red_m) / (nir_m + 6.0 * red_m - 7.5 * blue_m + 1.0 + eps), -1, 1))
    ndwi = (nir_m - swir_m) / (nir_m + swir_m + eps)

    props = item.get("properties", {})
    return {
        "date": (props.get("datetime") or "")[:10],
        "scene_id": item.get("id"),
        "cloud_cover_pct": props.get("eo:cloud_cover"),
        "clear_pixel_pct": round(100 * frac, 1),
        "cloud_masked": masked,
        "ndvi": ndvi,
        "evi": evi,
        "ndwi": ndwi,
    }


def render_ndvi_map(item: dict, cfg: FieldConfig, out_path: str):
    """Compute a PER-PIXEL NDVI image for one scene over the field bbox, mask
    cloud/shadow pixels (transparent), colourise (red=low, green=vigorous) and
    save a PNG. Returns out_path, or None if it can't be produced (missing
    matplotlib, unreadable bands...). Used only for the dashboard map overlay."""
    s2 = active_s2_config(cfg)
    assets = item.get("assets", {})

    def _href(band):
        a = assets.get(s2["assets"][band])
        return a["href"] if a and "href" in a else None

    rh, nh = _href("red"), _href("nir")
    if not rh or not nh:
        return None
    scl_href = None
    for k in ("scl", "SCL"):
        a = assets.get(k)
        if a and "href" in a:
            scl_href = a["href"]
            break
    try:
        red, shp = _read_band_array(rh, cfg.bbox, cap=200)
        nir, _ = _read_band_array(nh, cfg.bbox, out_shape=shp)
        scl = None
        if scl_href:
            scl, _ = _read_band_array(scl_href, cfg.bbox, out_shape=shp)
    except Exception as e:
        logger.info("NDVI map: band read failed (%s).", e)
        return None

    eps = 1e-8
    ndvi = (nir - red) / (nir + red + eps)
    keep = (red > 0) & (nir > 0)
    if scl is not None:
        keep &= np.isin(scl.astype(int), list(_SCL_KEEP))
    ndvi = np.where(keep, np.clip(ndvi, -0.1, 0.9), np.nan)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        try:
            cmap = matplotlib.colormaps["RdYlGn"]           # matplotlib >= 3.7
        except Exception:
            import matplotlib.cm as cm
            cmap = cm.get_cmap("RdYlGn")                     # older matplotlib
        norm = mcolors.Normalize(vmin=0.0, vmax=0.8)
        rgba = cmap(norm(np.nan_to_num(ndvi, nan=0.0)))
        rgba[..., 3] = np.where(np.isfinite(ndvi), 0.88, 0.0)  # masked pixels transparent
        plt.imsave(out_path, rgba)
        logger.info("Wrote per-pixel NDVI map -> %s", out_path)
        return out_path
    except Exception as e:
        logger.info("NDVI map: render failed (%s).", e)
        return None


def build_index_timeseries(items: list[dict], cfg: FieldConfig) -> pd.DataFrame:
    """
    Turn a list of STAC scenes into a tidy per-date index time series
    DataFrame (one row per scene that could be read), sorted by date.
    Columns: date, scene_id, cloud_cover_pct, ndvi, evi, ndwi.
    """
    rows = []
    for it in items:
        rec = read_scene_indices(it, cfg)
        if rec is not None:
            rows.append(rec)

    cols = ["date", "scene_id", "cloud_cover_pct", "clear_pixel_pct", "cloud_masked",
            "ndvi", "evi", "ndwi"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df = df.reindex(columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_indices_timeseries_openeo(cfg: FieldConfig) -> pd.DataFrame:
    """
    MANDATED-STACK ALTERNATIVE (Copernicus): compute the field-mean NDVI/EVI/NDWI
    time series server-side via openEO on the Copernicus Data Space.

    This is the clean way to use the Copernicus openEO service
    without wrestling with CDSE's S3 band access. It needs the free Copernicus
    account (interactive `authenticate_oidc()` device-code login the first time).
    Left importable-but-optional so the rest of the pipeline never requires the
    `openeo` package.

    Uncomment / adapt when your Copernicus account is ready:

        import openeo
        con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
        w, s, e, n = cfg.bbox
        cube = con.load_collection(
            "SENTINEL2_L2A",
            spatial_extent={"west": w, "south": s, "east": e, "north": n},
            temporal_extent=[cfg.start_date, cfg.end_date],
            bands=["B02", "B04", "B08", "B11"],
            max_cloud_cover=active_s2_config(cfg)["max_cloud_cover_pct"],
        )
        b = {k: cube.band(v) for k, v in {"blue":"B02","red":"B04","nir":"B08","swir":"B11"}.items()}
        ndvi = (b["nir"] - b["red"]) / (b["nir"] + b["red"])
        # ... aggregate_spatial(mean) over the bbox polygon, download to a DataFrame
    """
    raise NotImplementedError(
        "openEO path is provided as a documented template -- fill in credentials "
        "and the aggregate_spatial step when your Copernicus account is ready. "
        "Until then, use provider: earth_search (default) which needs no account."
    )


def run_sentinel2_ingestion(cfg: FieldConfig) -> Path:
    """
    Entry point:
      1. search cloud-filtered scenes over the field bbox/date range
      2. read the four bands per scene, compute field-level NDVI/EVI/NDWI
      3. write a tidy per-date index time series CSV 

    Also writes a scene manifest CSV for provenance. Never hard-fails: if band
    reading isn't possible (missing rasterio, no egress, provider needs auth),
    it still writes the manifest and an empty index series, and logs why -- so
    the rest of the pipeline (weather, IoT, alignment) keeps working.

    Network calls only succeed from an environment with internet access to the
    STAC endpoint and the COG store -- run from your laptop / a cloud VM, not a
    network-restricted sandbox.
    """
    s2 = active_s2_config(cfg)

    if s2["provider"] == "copernicus":
        creds = resolve_env_credentials()
        if not creds["CDSE_CLIENT_ID"] or not creds["CDSE_CLIENT_SECRET"]:
            logger.warning(
                "provider=copernicus but CDSE_CLIENT_ID/SECRET not set. STAC search may "
                "still work, but direct band reads will not. For a no-account run, set "
                "sentinel2.provider: earth_search in the config."
            )

    try:
        items = search_sentinel2_scenes(cfg)
    except Exception as e:
        logger.error("Sentinel-2 STAC search failed: %s", e)
        items = []

    if not items:
        logger.warning("No scenes found -- widen the date range or cloud-cover threshold.")

    # Only READ bands for a bounded number of scenes (each scene = 4 remote band
    # reads). This is what keeps the run to a few minutes instead of ~an hour.
    n_found = len(items)
    items = subsample_scenes(items, s2["max_scenes"])
    if len(items) < n_found:
        logger.info("Reading bands for %d of %d scenes (evenly spread; max_scenes=%d).",
                    len(items), n_found, s2["max_scenes"])

    # scene manifest (provenance)
    manifest_path = cfg.processed_dir / f"{cfg.name}_sentinel2_scenes.csv"
    pd.DataFrame([
        {
            "scene_id": it.get("id"),
            "datetime": it.get("properties", {}).get("datetime"),
            "cloud_cover_pct": it.get("properties", {}).get("eo:cloud_cover"),
        }
        for it in items
    ]).to_csv(manifest_path, index=False)
    logger.info("Wrote scene manifest (%d scenes) to %s", len(items), manifest_path)

    # index time series (the index series)
    with _rasterio_env(s2["provider"]):
        index_df = build_index_timeseries(items, cfg)

    index_path = cfg.processed_dir / f"{cfg.name}_sentinel2_indices.csv"
    index_df.to_csv(index_path, index=False)
    if index_df.empty:
        logger.warning(
            "Wrote EMPTY index series to %s (no bands could be read). "
            "Check internet egress / provider / rasterio install.", index_path
        )
    else:
        logger.info("Wrote %d-date Sentinel-2 index series to %s", len(index_df), index_path)
        # per-pixel NDVI map of the CLEAREST scene -> lets the dashboard show a
        # true map of within-field vegetation (not just a mean).
        try:
            best = index_df.sort_values("clear_pixel_pct", ascending=False).iloc[0]
            item = next((it for it in items if it.get("id") == best["scene_id"]), None)
            if item is not None:
                png = cfg.processed_dir / f"{cfg.name}_ndvi_map.png"
                with _rasterio_env(s2["provider"]):
                    render_ndvi_map(item, cfg, str(png))
        except Exception as e:
            logger.info("NDVI map generation skipped (%s).", e)
    return index_path


class _rasterio_env:
    """Context manager that opens a rasterio.Env if rasterio is available, else a no-op."""
    def __init__(self, provider: str, token: str | None = None):
        self.opts = _gdal_env(provider, token)
        self._env = None

    def __enter__(self):
        try:
            import rasterio
            self._env = rasterio.Env(**self.opts)
            self._env.__enter__()
        except Exception as e:  # rasterio missing or env failure -> keep going
            logger.info("rasterio env not active (%s) -- band reads will be skipped if attempted", e)
            self._env = None
        return self

    def __exit__(self, *exc):
        if self._env is not None:
            self._env.__exit__(*exc)
        return False
