"""
Sentinel-2 L2A ingestion -> a per-date field-level NDVI/EVI/NDWI time series.

Global coverage: Sentinel-2 images the entire land surface of the Earth on a
~5 day revisit cycle, so this module places NO assumption about which
continent/country the field is in. Everything location-specific comes from
config/field_config.yaml's bbox.

Two data-access paths are supported (pick one via `sentinel2.provider` in the
config):

  * "earth_search"  (DEFAULT, dev-friendly)
      Element-84 Earth Search STAC over the public AWS "sentinel-cogs" bucket.
      No account, no key -- the COGs are public. Great for getting the pipeline
      running *today* and for testing. Same Sentinel-2 L2A data as Copernicus.

  * "copernicus"    (the mandated-stack path for your final submission)
      Copernicus Data Space STAC. Reading individual bands from CDSE requires
      either their S3 (eodata) credentials or, more cleanly, the openEO service
      (`fetch_indices_timeseries_openeo` below). The project brief mandates the
      Copernicus stack, so switch to this before you hand in -- but you can
      prototype everything on "earth_search" first.

What "a field-level index time series" means here:
  For each cloud-filtered scene, we read the four bands (red/nir/blue/swir)
  clipped to the field bbox, take the spatial MEAN of each band over the field,
  then compute NDVI/EVI/NDWI from those means. That gives one (date -> index)
  row per scene -- exactly the sparse satellite series that Week-2's alignment
  step (src/align.py) merges with the dense weather/soil series.

  (Computing the index from band-means is the standard field-level aggregation
  and sidesteps the fact that S2 bands live on different resolution grids -- 10m
  for red/nir/blue, 20m for swir. If you later want per-pixel index *maps*,
  resample the bands to a common grid first; `compute_indices_for_scene` already
  takes aligned arrays for that.)
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
    }


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


# ---------------------------------------------------------------------------
# Band reading (COG windowed read over the field bbox)
# ---------------------------------------------------------------------------
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


def read_scene_indices(item: dict, cfg: FieldConfig) -> dict | None:
    """
    For one STAC scene, read the four band means over the field bbox and turn
    them into field-level NDVI/EVI/NDWI. Returns None if the bands can't be read
    (e.g. asset missing, no valid pixels) so the caller can skip the scene
    without failing the whole pipeline.
    """
    s2 = active_s2_config(cfg)
    assets = item.get("assets", {})

    means = {}
    for band, asset_key in s2["assets"].items():
        asset = assets.get(asset_key)
        if not asset or "href" not in asset:
            logger.warning("Scene %s missing asset '%s' -- skipping scene", item.get("id"), asset_key)
            return None
        try:
            means[band] = read_band_mean_over_bbox(asset["href"], cfg.bbox)
        except Exception as e:
            logger.warning("Scene %s: failed reading band '%s' (%s): %s",
                           item.get("id"), band, asset_key, e)
            return None

    if any(np.isnan(v) for v in means.values()):
        logger.info("Scene %s has no valid pixels over the field bbox -- skipping", item.get("id"))
        return None

    # scale digital numbers -> reflectance, then indices from band means
    red, nir = means["red"] / 10000.0, means["nir"] / 10000.0
    blue, swir = means["blue"] / 10000.0, means["swir"] / 10000.0
    eps = 1e-8
    ndvi = (nir - red) / (nir + red + eps)
    evi = float(np.clip(2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps), -1, 1))
    ndwi = (nir - swir) / (nir + swir + eps)

    props = item.get("properties", {})
    return {
        "date": (props.get("datetime") or "")[:10],
        "scene_id": item.get("id"),
        "cloud_cover_pct": props.get("eo:cloud_cover"),
        "ndvi": ndvi,
        "evi": evi,
        "ndwi": ndwi,
    }


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

    cols = ["date", "scene_id", "cloud_cover_pct", "ndvi", "evi", "ndwi"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)[cols]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_indices_timeseries_openeo(cfg: FieldConfig) -> pd.DataFrame:
    """
    MANDATED-STACK ALTERNATIVE (Copernicus): compute the field-mean NDVI/EVI/NDWI
    time series server-side via openEO on the Copernicus Data Space.

    This is the clean way to satisfy the brief's "Copernicus STAC API" mandate
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
    Week-1/2 entry point:
      1. search cloud-filtered scenes over the field bbox/date range
      2. read the four bands per scene, compute field-level NDVI/EVI/NDWI
      3. write a tidy per-date index time series CSV (the input Week-2 aligns)

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

    # index time series (the real Week-2 input)
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
