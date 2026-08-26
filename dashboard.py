"""
Assistant d'irrigation — interface interactive (N'IMPORTE QUELLE parcelle
sur Terre).

Étapes pour l'utilisateur :
  1. Rechercher un lieu (pays / région / ville) — ou utiliser un exemple.
  2. Dessiner un rectangle sur la carte (l'outil carré, en haut à droite).
  3. Choisir une analyse :
       • ANALYSE RAPIDE (temps réel) — recommandation immédiate à partir de
         données mondiales instantanées (Open-Meteo : météo + humidité du sol).
       • ANALYSE SATELLITE COMPLÈTE (2–4 min) — lance le vrai pipeline
         Sentinel-2 + météo + capteur : ajoute les indices de végétation
         (NDVI/EVI/NDWI) et fait tourner l'agent sur cette parcelle précise.

Run:  python -m streamlit run dashboard.py --server.fileWatcherType none
Carte interactive:  pip install streamlit-folium folium
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load a .env at the project root so GROQ_API_KEY (and any other keys) are set
# permanently — no need to export them in every terminal session.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

try:
    from src.config_loader import load_config
    from src.indices import flag_ndvi_anomaly
    _cfg = load_config("config/field_config.yaml")
    CFG_CENTER = [float(_cfg.centroid[1]), float(_cfg.centroid[0])]  # [lat, lon]
except Exception:
    CFG_CENTER = [42.03, -93.61]
    def flag_ndvi_anomaly(cur, mean, std):  # fallback if src not importable
        return bool(std and cur < mean - 2 * std)

HAVE_MAP = True
try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
except Exception:
    HAVE_MAP = False

st.set_page_config(page_title="AgriSat — Irrigation & Fertilisation", page_icon=None, layout="wide")

# ============================================================ theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{
  --bg:#f3f1e7; --surface:#fffdf7; --surface-2:#f7f5ea; --ink:#1e261b; --ink-soft:#4c5545;
  --muted:#8a9078; --line:#e4e0cf; --green:#2f6b45; --green-deep:#1f4d31;
  --wheat:#c58a2e; --sky:#3f7a8c; --earth:#8a6d3b;
}
#MainMenu, [data-testid="stToolbar"], footer {visibility:hidden; height:0;}
header[data-testid="stHeader"]{display:none;}
.stApp{
  background-color:#f3f1e7;
  background-image:
    radial-gradient(1100px 480px at 84% -14%, rgba(63,122,140,.10), transparent 60%),
    radial-gradient(900px 420px at -5% 0%, rgba(47,107,69,.07), transparent 55%),
    repeating-linear-gradient(180deg, rgba(47,107,69,.028) 0 1px, transparent 1px 30px);
}
.block-container{padding-top:2rem; padding-bottom:3.5rem; max-width:1160px;}
html,body,[class*="css"],[data-testid="stMarkdownContainer"]{font-family:'IBM Plex Sans',system-ui,sans-serif; color:var(--ink);}
.mono,.kpi-value,.reco-action{font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;}
h1,h2,h3,.serif{font-family:'Fraunces',Georgia,serif;}

/* ---- header ---- */
.topbar{display:flex; justify-content:space-between; align-items:center;
  padding:6px 0 18px; margin-bottom:8px; border-bottom:1px solid var(--line);}
.brandwrap{display:flex; align-items:center; gap:14px;}
.mark{width:44px; height:44px; border-radius:12px; flex:none;
  background:linear-gradient(145deg,var(--green),var(--green-deep));
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 6px 16px rgba(31,77,49,.28);}
.brand small{display:block; font-family:'IBM Plex Mono',monospace; font-size:.62rem;
  letter-spacing:.26em; text-transform:uppercase; color:var(--green); font-weight:600; margin-bottom:2px;}
.brand b{font-family:'Fraunces',serif; font-size:1.5rem; font-weight:600; letter-spacing:-.01em; color:var(--ink);}
.meta{text-align:right; font-size:.8rem; color:var(--muted); line-height:1.55;} .meta b{color:var(--ink);}

/* ---- section rails (replace numbered steps) ---- */
.rail{display:flex; align-items:center; gap:13px; margin:32px 0 14px;}
.rail .idx{font-family:'IBM Plex Mono',monospace; font-size:.7rem; font-weight:600; letter-spacing:.05em;
  color:#fff; background:linear-gradient(145deg,var(--green),var(--green-deep));
  border-radius:7px; padding:4px 9px; box-shadow:0 3px 8px rgba(31,77,49,.22);}
.rail .t{font-family:'Fraunces',serif; font-size:1.18rem; font-weight:600; color:var(--ink);}
.rail .ln{flex:1; height:1px; background:linear-gradient(90deg,var(--line),transparent);}

/* ---- recommendation hero ---- */
.reco{position:relative; border-radius:22px; padding:30px 34px; color:#fff;
  background:linear-gradient(135deg, var(--c), color-mix(in srgb, var(--c) 62%, #12261a));
  box-shadow:0 16px 40px rgba(20,32,20,.20); overflow:hidden;}
.reco-motif{position:absolute; right:-10px; bottom:-24px; opacity:.16; pointer-events:none;}
.reco-eyebrow{font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; opacity:.9;}
.reco-action{font-family:'Fraunces',serif; font-size:3.1rem; font-weight:600; line-height:1.02; margin:8px 0 6px;}
.reco-headline{font-size:1.12rem; opacity:.97; max-width:70%;}
.reco-badge{position:absolute; top:26px; right:30px; font-family:'IBM Plex Mono',monospace;
  font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; background:rgba(255,255,255,.16);
  padding:6px 12px; border-radius:20px; backdrop-filter:blur(2px);}

/* ---- why card ---- */
.why{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:18px 22px; margin-top:16px;
  box-shadow:0 2px 10px rgba(30,38,27,.04);}
.why .lbl{font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-bottom:9px;}
.why ul{margin:0; padding-left:2px; list-style:none;} .why li{margin:7px 0; color:var(--ink-soft); font-size:.96rem; padding-left:18px; position:relative;}
.why li::before{content:""; position:absolute; left:0; top:.55em; width:6px; height:6px; border-radius:50%; background:var(--green); opacity:.55;}

/* ---- KPI grid ---- */
.kpis{display:grid; grid-template-columns:repeat(4,1fr); gap:15px; margin-top:10px;}
.kpi{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:17px 19px;
  box-shadow:0 2px 10px rgba(30,38,27,.04); transition:transform .16s ease, box-shadow .16s ease;}
.kpi:hover{transform:translateY(-3px); box-shadow:0 14px 30px rgba(30,38,27,.10);}
.kpi-head{display:flex; align-items:center; gap:9px; margin-bottom:2px;}
.kpi-head svg{width:19px; height:19px; flex:none;}
.kpi-label{font-family:'IBM Plex Mono',monospace; font-size:.62rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);}
.kpi-value{font-size:1.95rem; font-weight:500; line-height:1.1; margin-top:8px; color:var(--ink);}
.kpi-unit{font-family:'IBM Plex Sans',sans-serif; font-size:.78rem; color:var(--muted); margin-left:3px;}
.pill{display:inline-block; margin-top:9px; font-size:.72rem; font-weight:600; padding:3px 11px; border-radius:20px; color:var(--p); background:color-mix(in srgb, var(--p) 15%, #fff);}
/* soil gauge inside a KPI */
.gaugewrap{position:relative; height:15px; margin-top:11px;}
.gauge{position:absolute; top:4px; left:0; right:0; height:8px; border-radius:5px; overflow:hidden; display:flex; box-shadow:inset 0 0 0 1px rgba(0,0,0,.04);}
.gz{height:100%;} .gz-r{background:#e0b3a3;} .gz-a{background:#e7cd94;} .gz-g{background:#b6d2ab;}
.gm{position:absolute; top:0; width:3px; height:15px; border-radius:2px; background:var(--ink); box-shadow:0 0 0 2px var(--surface);}

.eyebrow{font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin:26px 0 9px;}
.stButton>button{background:linear-gradient(145deg,var(--green),var(--green-deep)); color:#fff; border:0; border-radius:12px;
  padding:.62rem 1.2rem; font-family:'IBM Plex Sans',sans-serif; font-weight:600; width:100%;
  box-shadow:0 6px 16px rgba(31,77,49,.22); transition:transform .12s ease, box-shadow .12s ease;}
.stButton>button:hover{transform:translateY(-1px); box-shadow:0 10px 22px rgba(31,77,49,.30); color:#fff;}
hr{border-color:var(--line);}

/* ---- alerts ---- */
.alert-head{display:flex; align-items:baseline; gap:12px; margin:30px 0 8px;}
.alert-head .n{font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:var(--muted);}
.alerts{display:flex; flex-direction:column; gap:10px;}
.alert{display:flex; align-items:flex-start; gap:14px; background:var(--surface); border:1px solid var(--line);
  border-left:3px solid var(--s); border-radius:14px; padding:14px 18px; box-shadow:0 2px 8px rgba(30,38,27,.03);}
.alert .dot{width:10px; height:10px; border-radius:50%; background:var(--s); margin-top:5px; flex:none; box-shadow:0 0 0 4px color-mix(in srgb, var(--s) 16%, transparent);}
.alert .txt b{font-size:.94rem; color:var(--ink);} .alert .txt span{display:block; color:var(--ink-soft); font-size:.86rem; margin-top:3px;}
.alert .sev{margin-left:auto; font-family:'IBM Plex Mono',monospace; font-size:.6rem; letter-spacing:.14em; text-transform:uppercase; color:var(--s); padding-top:3px; flex:none;}

/* ---- selection card ---- */
.selcard{position:relative; overflow:hidden; background:linear-gradient(135deg,#eef3e6,#e7efdd);
  border:1px solid #d6e0cf; border-radius:16px; padding:18px 22px; margin-top:14px;}
.selcard .lbl{font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; color:var(--green); margin-bottom:6px;}
.selcard .coord{font-family:'IBM Plex Mono',monospace; font-size:1.15rem; font-weight:600; color:var(--ink);}
.selcard .sub{color:var(--ink-soft); font-size:.85rem; margin-top:4px;}
</style>
""", unsafe_allow_html=True)

COLORS = {"irriguer": "#1f6f8b", "attendre": "#2f7d4f", "surveiller": "#b5822a"}
ACT_FR = {"irriguer": "Irriguer", "attendre": "Attendre", "surveiller": "Surveiller"}
SEV = {"critique": "#b5482a", "attention": "#b5822a", "info": "#3d6b8a", "ok": "#2f7d4f"}
SEV_RANK = {"critique": 3, "attention": 2, "info": 1, "ok": 0}
API = "https://api.open-meteo.com/v1/forecast"
SOIL_MODELS = [
    ("gfs_seamless", ["soil_moisture_10_to_40cm", "soil_moisture_0_to_10cm"]),
    ("icon_seamless", ["soil_moisture_9_to_27cm", "soil_moisture_3_to_9cm",
                       "soil_moisture_1_to_3cm", "soil_moisture_0_to_1cm"]),
]

# ---- inline vector iconography (no emoji) ----
_S = ('viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
      'stroke-linecap="round" stroke-linejoin="round"')
ICON = {
    "soil": f'<svg {_S}><path d="M12 3c2.4 2.9 3.8 5 3.8 7.1A3.8 3.8 0 0 1 8.2 10C8.2 8 9.6 5.9 12 3Z"/>'
            '<path d="M3 16.5c2 0 2 1.1 4.5 1.1S9.5 16.5 12 16.5s2 1.1 4.5 1.1S18.5 16.5 21 16.5"/>'
            '<path d="M3 20c2 0 2 1.1 4.5 1.1S9.5 20 12 20"/></svg>',
    "ndvi": f'<svg {_S}><path d="M5 19c0-7.7 5.8-12.5 14-12.5 0 7.7-5 13.5-12.7 13.5H5Z"/>'
            '<path d="M5.5 18.5c3-3.8 6-6 9-7.4"/></svg>',
    "rain": f'<svg {_S}><path d="M7 15a4 4 0 0 1 .3-8A5 5 0 0 1 17 8.5a3.5 3.5 0 0 1-.4 6.5"/>'
            '<path d="M8 18.5 7 21M12 18.5 11 21M16 18.5 15 21"/></svg>',
    "et0": f'<svg {_S}><circle cx="12" cy="12" r="3.6"/>'
           '<path d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.5 5.5l1.5 1.5M16.9 16.9l1.6 1.6'
           'M18.5 5.5 17 7M7 17l-1.5 1.5"/></svg>',
}
ICON_COLOR = {"soil": "#6b7f9a", "ndvi": "#2f6b45", "rain": "#3f7a8c", "et0": "#c58a2e"}

MARK_SVG = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M12 21v-8"/><path d="M12 13c0-3 2.6-5 6-5 0 3-2.6 5-6 5Z"/>'
            '<path d="M12 15.5c0-2.6-2.1-4.5-5-4.5 0 2.6 2.1 4.5 5 4.5Z"/></svg>')

MOTIF_SVG = ('<svg class="reco-motif" width="230" height="150" viewBox="0 0 230 150" fill="none" '
             'stroke="#fff" stroke-width="2">'
             '<path d="M0 150 Q115 66 230 150"/><path d="M0 150 Q115 92 230 150"/>'
             '<path d="M0 150 Q115 118 230 150"/><path d="M18 150 Q115 42 214 150"/></svg>')


def gauge_html(v):
    """Linear soil-moisture gauge with stress/marginal/comfortable zones + marker."""
    if v is None or pd.isna(v):
        return ""
    pct = max(0.0, min(1.0, float(v) / 0.45)) * 100
    return ('<div class="gaugewrap"><div class="gauge">'
            '<div class="gz gz-r" style="width:44.4%"></div>'
            '<div class="gz gz-a" style="width:22.2%"></div>'
            '<div class="gz gz-g" style="width:33.4%"></div></div>'
            f'<div class="gm" style="left:{pct:.1f}%"></div></div>')


def kpi_card(icon_key, label, value_html, pill_label, pill_color, extra=""):
    ic = f'<span style="color:{ICON_COLOR.get(icon_key, "#2f6b45")}">{ICON.get(icon_key, "")}</span>'
    return (f'<div class="kpi"><div class="kpi-head">{ic}<span class="kpi-label">{label}</span></div>'
            f'<div class="kpi-value">{value_html}</div>'
            f'<span class="pill" style="--p:{pill_color}">{pill_label}</span>{extra}</div>')


# ============================================================ data helpers
def _soil_daily(lat, lon, model, varset, past_days, days):
    s = requests.get(API, params={"latitude": lat, "longitude": lon, "hourly": ",".join(varset),
                                   "models": model, "past_days": past_days, "forecast_days": days,
                                   "timezone": "auto"}, timeout=25)
    s.raise_for_status()
    h = s.json().get("hourly", {})
    key = next((k for k in varset if k in h and any(v is not None for v in h[k])), None)
    if not key:
        return None, None
    hs = pd.DataFrame({"t": pd.to_datetime(h["time"]), "sm": pd.to_numeric(h[key], errors="coerce")})
    hs["date"] = hs["t"].dt.normalize()
    return hs.groupby("date", as_index=False)["sm"].mean(), key


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live(lat: float, lon: float, past_days: int = 90, days: int = 7):
    """Live global weather + modelled soil moisture for any point on Earth."""
    w = requests.get(API, params={"latitude": lat, "longitude": lon,
                                   "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_mean",
                                   "past_days": past_days, "forecast_days": days, "timezone": "auto"}, timeout=25)
    w.raise_for_status()
    jd = w.json()["daily"]
    dd = pd.DataFrame({"date": pd.to_datetime(jd["time"]),
                       "precip": pd.to_numeric(jd["precipitation_sum"], errors="coerce"),
                       "et0": pd.to_numeric(jd["et0_fao_evapotranspiration"], errors="coerce"),
                       "temp": pd.to_numeric(jd["temperature_2m_mean"], errors="coerce")})
    soil_daily, soil_key = None, None
    for model, varset in SOIL_MODELS:
        try:
            soil_daily, soil_key = _soil_daily(lat, lon, model, varset, past_days, days)
            if soil_daily is not None and len(soil_daily):
                break
        except Exception:
            continue
    return dd, soil_daily, soil_key


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_bits(lat, lon):
    """Next-7-day rain, ET0 and the minimum forecast soil moisture."""
    dd, soil_daily, _ = fetch_live(lat, lon, past_days=5, days=7)
    today = pd.Timestamp.now().normalize()
    fut = dd[dd["date"] > today]
    rain = float(fut["precip"].sum()) if len(fut) else float(dd["precip"].tail(7).sum())
    et0 = float(fut["et0"].sum()) if len(fut) else float(dd["et0"].tail(7).sum())
    fc_min = None
    if soil_daily is not None and len(soil_daily):
        sf = soil_daily[soil_daily["date"] > today]
        fc_min = float(sf["sm"].min()) if len(sf) else None
    return rain, et0, fc_min


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(name: str, count: int = 6):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": name, "count": count, "language": "fr", "format": "json"}, timeout=15)
        r.raise_for_status()
        return [{"name": x["name"], "lat": x["latitude"], "lon": x["longitude"],
                 "country": x.get("country", ""), "admin1": x.get("admin1", "")}
                for x in r.json().get("results", [])]
    except Exception:
        return []


# ============================================================ rule engine
def recommend(soil, soil_missing, anomaly, rain_next, et0_next, soil_fc_min=None, ndvi=None):
    bare = ndvi is not None and not pd.isna(ndvi) and ndvi < 0.20  # little/no active canopy
    if soil_missing:
        return ("surveiller", "Humidité du sol indisponible",
                ["La donnée d'humidité du sol n'est pas disponible pour ce lieu.",
                 "Choisir une autre parcelle, ou relancer l'analyse."])
    if anomaly and soil >= 0.25:
        return ("surveiller", "Anomalie de végétation à investiguer",
                [f"NDVI anormalement bas alors que le sol est correct ({soil:.2f} m³/m³).",
                 "Cause probable : ravageur, maladie ou carence — pas un manque d'eau. Inspecter la parcelle."])
    fc_dip = soil_fc_min is not None and soil_fc_min < 0.22
    if soil < 0.20 or (soil < 0.27 and rain_next < 10 and et0_next > 25) or (fc_dip and rain_next < 10):
        zone = ("en zone de stress" if soil < 0.20
                else (f"chute prévue à {soil_fc_min:.2f} m³/m³" if fc_dip else "marginale"))
        # Reconcile with vegetation: if NDVI shows little/no crop, irrigation is
        # questionable (you irrigate a crop, not bare ground) -> caveat it.
        if bare:
            return ("surveiller", "Sol sec mais couvert végétal très faible",
                    [f"Humidité du sol {soil:.2f} m³/m³ ({zone}) — condition d'irrigation remplie.",
                     f"Mais NDVI très bas ({ndvi:.2f}) : sol nu ou culture récoltée/absente.",
                     "Vérifier qu'une culture est bien en place avant d'irriguer."])
        return ("irriguer", "Irriguer dans les 1 à 2 jours",
                [f"Humidité du sol {soil:.2f} m³/m³ ({zone}).",
                 f"Pluie prévue faible ({rain_next:.0f} mm) et forte demande évaporative ({et0_next:.0f} mm sur 7 j)."])
    reasons = []
    if rain_next >= 10:
        reasons.append(f"Pluie suffisante attendue ({rain_next:.0f} mm) — elle rechargera le sol.")
    reasons.append(f"Humidité du sol {soil:.2f} m³/m³ — pas de stress imminent.")
    return ("attendre", "Pas d'irrigation nécessaire pour l'instant", reasons)


def soil_label(v):
    if pd.isna(v):
        return "—", "#7b8476"
    if v < 0.20:
        return "Stress", "#b5482a"
    if v < 0.30:
        return "Marginal", "#b5822a"
    return "Confortable", "#2f7d4f"


FERT_COLOR = {"recommandée": "#c58a2e", "différer": "#3f7a8c", "aucune": "#2f7d4f", "indéterminé": "#8a9078"}


def fertilize(ndvi, ndvi_available, soil, soil_missing):
    """Fertilization decision (rule-based). Distinguishes a NUTRIENT problem
    (weak canopy while water is adequate) from a WATER problem (weak canopy but
    dry soil) — you don't fertilize a thirsty crop. Needs NDVI (satellite)."""
    if not ndvi_available or ndvi is None or pd.isna(ndvi):
        return ("indéterminé", "Nutrition non évaluée",
                "L'analyse satellite (NDVI) est nécessaire pour évaluer la fertilisation.")
    if ndvi < 0.20:
        return ("aucune", "Pas de culture active",
                f"Couvert très faible (NDVI {ndvi:.2f}) : sol nu ou récolté — fertilisation sans objet.")
    if soil_missing:
        return ("indéterminé", "Nutrition non évaluée",
                "Humidité du sol indisponible : impossible de distinguer carence et stress hydrique.")
    if ndvi < 0.45 and soil >= 0.22:
        return ("recommandée", "Apport d'azote à envisager",
                f"Vigueur limitée (NDVI {ndvi:.2f}) alors que l'eau est suffisante ({soil:.2f} m³/m³) : "
                "le facteur limitant n'est pas l'eau mais probablement l'azote. Envisager un apport azoté, "
                "idéalement confirmé par un test de sol.")
    if ndvi < 0.45 and soil < 0.22:
        return ("différer", "Corriger l'eau avant de fertiliser",
                f"Couvert faible (NDVI {ndvi:.2f}) mais sol sec ({soil:.2f} m³/m³) : le stress est hydrique, "
                "pas nutritif. Fertiliser un couvert assoiffé gaspille l'engrais — irriguer d'abord, réévaluer ensuite.")
    return ("aucune", "Nutrition adéquate",
            f"Vigueur satisfaisante (NDVI {ndvi:.2f}) : aucun apport nécessaire pour l'instant.")


def build_alerts(soil, soil_missing, anomaly, action, headline, rain_next, et0_next, soil_fc_min, ndvi=None):
    a = []
    bare = ndvi is not None and not pd.isna(ndvi) and ndvi < 0.20
    if soil_missing:
        a.append(("critique", "Humidité du sol indisponible",
                  "Aucune donnée de sol — décision d'irrigation suspendue."))
    if action == "irriguer":
        a.append(("critique", "Irrigation requise", headline + "."))
    if bare:
        a.append(("attention", "Couvert végétal très faible",
                  f"NDVI {ndvi:.2f} : sol nu ou culture récoltée/absente — "
                  "vérifier la présence d'une culture avant d'irriguer."))
    if anomaly:
        a.append(("attention", "Anomalie de végétation",
                  "NDVI anormalement bas — inspecter la parcelle (ravageur, maladie ou carence)."))
    if (soil_fc_min is not None) and soil_fc_min < 0.25 and action != "irriguer":
        a.append(("attention", "Baisse d'humidité projetée",
                  f"La prévision descend jusqu'à ~{soil_fc_min:.2f} m³/m³ d'ici 7 jours."))
    has_warn = any(s in ("critique", "attention") for s, _, _ in a)
    if (not soil_missing) and rain_next < 10 and et0_next > 25 and not has_warn:
        a.append(("info", "Bilan hydrique déficitaire",
                  f"Pluie faible ({rain_next:.0f} mm) et forte demande évaporative "
                  f"({et0_next:.0f} mm sur 7 j) — à surveiller."))
    if not a:
        a.append(("ok", "Aucune alerte active", "Parcelle stable — aucune action requise."))
    return a


def ndvi_anomaly_from_df(df):
    if "ndvi" not in df or df["ndvi"].notna().sum() < 5:
        return False, float("nan")
    obs = (df[df.get("ndvi_observed", True) == True]["ndvi"].dropna()
           if "ndvi_observed" in df else df["ndvi"].dropna())
    if len(obs) < 5:
        return False, float("nan")
    cur = float(obs.iloc[-1])
    return flag_ndvi_anomaly(cur, float(obs[:-1].mean()), float(obs[:-1].std())), cur


def bbox_from_drawing(feat):
    try:
        coords = feat["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
        return (round(min(lons), 5), round(min(lats), 5), round(max(lons), 5), round(max(lats), 5))
    except Exception:
        return None


def box_around(lat, lon, halfkm=1.2):
    dlat = halfkm / 111.0
    dlon = halfkm / (111.0 * max(float(np.cos(np.radians(lat))), 0.1))
    return (round(float(lon) - dlon, 5), round(float(lat) - dlat, 5),
            round(float(lon) + dlon, 5), round(float(lat) + dlat, 5))


def set_selection(lat, lon, bbox=None, zoom=12):
    st.session_state.center = [lat, lon]
    st.session_state.zoom = zoom
    st.session_state.sel = {"bbox": bbox or box_around(lat, lon),
                            "centroid": (round(lon, 5), round(lat, 5))}


# ============================================================ full pipeline (satellite)
def make_zone_config(clat, clon, bbox, max_km=2.5, window_days=120, max_scenes=10):
    """Config for the on-demand satellite run. To keep it to a few minutes we
    (a) cap the read area to a field-sized box around the centre — a huge drawn
    box means huge per-scene reads — (b) use a recent ~4-month window, and
    (c) read bands for only a handful of evenly-spread scenes."""
    base = yaml.safe_load(open(ROOT / "config" / "field_config.yaml"))
    cfg = copy.deepcopy(base)
    name = ("zone_%.4f_%.4f" % (clat, clon)).replace("-", "m").replace(".", "p")
    # clamp the AOI to ~max_km around the centroid (satellite reads a field, not a region)
    b = box_around(clat, clon, max_km / 2.0)
    cfg["field"]["name"] = name
    cfg["field"]["bbox"] = {"min_lon": float(b[0]), "min_lat": float(b[1]),
                            "max_lon": float(b[2]), "max_lat": float(b[3])}
    cfg["field"]["centroid"] = {"lon": float(clon), "lat": float(clat)}
    today = pd.Timestamp.now().normalize()
    cfg["time_range"]["start_date"] = str((today - pd.Timedelta(days=window_days)).date())
    cfg["time_range"]["end_date"] = str(today.date())
    cfg.setdefault("sentinel2", {})["max_scenes"] = max_scenes
    cfg["sentinel2"]["max_cloud_cover_pct"] = 15
    cfg.setdefault("ismn", {})["data_path"] = "data/raw/__no_station__"  # force simulator fallback
    path = ROOT / "config" / f"_{name}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return path, name


SOIL_SRC_LABEL = {
    "ismn_real": "capteur ISMN réel (station proche)",
    "open_meteo_modeled": "Open-Meteo modélisée — repli mondial (aucune station requise)",
    "simulated": "simulateur IoT (amorcé sur Open-Meteo)",
}


def run_full_pipeline(clat, clon, bbox):
    from src.pipeline import run_pipeline
    path, name = make_zone_config(clat, clon, bbox)
    res = run_pipeline(str(path))
    zbbox = load_config(str(path)).bbox  # the clamped bbox the satellite actually read
    return res.get("aligned_daily"), str(path), res.get("iot_source"), zbbox, name


# ============================================================ agent hooks
def _agent_from_csv(csv_path, clon, clat):
    import types
    import src.agent as agent
    agent._CFG = types.SimpleNamespace(centroid=(clon, clat))
    agent._CSV = Path(csv_path)
    return agent.run_agent("Faut-il irriguer cette parcelle dans les prochains jours ?",
                           "openai/gpt-oss-120b", days=7, backend="groq")


def agent_live(clat, clon):
    dd, soil_daily, _ = fetch_live(round(clat, 3), round(clon, 3))
    today = pd.Timestamp.now().normalize()
    d = dd.rename(columns={"precip": "precip_sum_mm", "et0": "et0_sum_mm", "temp": "temp_mean_c"})
    d = d[d["date"] <= today].copy()
    if soil_daily is not None and len(soil_daily):
        d = d.merge(soil_daily.rename(columns={"sm": "soil_moisture_sensor_mean"}), on="date", how="left")
    else:
        d["soil_moisture_sensor_mean"] = np.nan
    d["soil_moisture_sensor_mean"] = d["soil_moisture_sensor_mean"].interpolate().ffill().bfill()
    csv = Path(tempfile.gettempdir()) / "live_field_aligned.csv"
    d.to_csv(csv, index=False)
    return _agent_from_csv(csv, clon, clat)


# ============================================================ shared render
def render_results(clat, clon, *, soil_now, soil_missing, hist, ndvi, ndvi_available,
                   anomaly, source_label, agent_fn, soil_source, ndvi_map=None, map_bounds=None):
    rain_next, et0_next, soil_fc_min = forecast_bits(round(clat, 3), round(clon, 3))
    ndvi_val = ndvi if (ndvi_available and not pd.isna(ndvi)) else None
    action, headline, reasons = recommend(soil_now, soil_missing, anomaly, rain_next, et0_next,
                                          soil_fc_min, ndvi=ndvi_val)
    color = COLORS[action]

    st.markdown(f"""
    <div class="reco" style="--c:{color}">
      {MOTIF_SVG}
      <div class="reco-badge">{source_label}</div>
      <div class="reco-eyebrow">Recommandation · {clat:.3f}, {clon:.3f}</div>
      <div class="reco-action">{ACT_FR[action]}</div>
      <div class="reco-headline">{headline}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="why"><div class="lbl">Pourquoi</div><ul>'
                + "".join(f"<li>{r}</li>" for r in reasons) + '</ul></div>', unsafe_allow_html=True)

    # ---- fertilization decision (second recommendation) ----
    fsev, fhead, freason = fertilize(ndvi_val, ndvi_available, soil_now, soil_missing)
    fc = FERT_COLOR[fsev]
    st.markdown(
        f'<div class="why" style="border-left:3px solid {fc};">'
        f'<div class="lbl" style="color:{fc};">Fertilisation · {fsev}</div>'
        f'<div style="font-family:Fraunces,serif;font-size:1.12rem;color:var(--ink);margin-bottom:5px;">{fhead}</div>'
        f'<div style="color:var(--ink-soft);font-size:.94rem;line-height:1.5;">{freason}</div></div>',
        unsafe_allow_html=True)

    slab, scol = soil_label(soil_now)
    soil_txt = "—" if soil_missing else f"{soil_now:.2f}"
    if ndvi_available and not pd.isna(ndvi):
        nd_txt = f"{ndvi:.2f}"
        # label reflects the NDVI VALUE (canopy density), not just anomaly status
        if anomaly:
            nd_lab, nd_col = "Anomalie", "#b5482a"
        elif ndvi < 0.20:
            nd_lab, nd_col = "Sol nu / très faible", "#8a6d3b"
        elif ndvi < 0.40:
            nd_lab, nd_col = "Couvert faible", "#b5822a"
        else:
            nd_lab, nd_col = "Sain", "#2f7d4f"
    else:
        nd_txt, nd_lab, nd_col = "—", "analyse satellite", "#7b8476"
    et0_lab, et0_col = ("Élevée", "#b5822a") if et0_next > 25 else ("Modérée", "#2f7d4f")
    rain_col = "#2f7d4f" if rain_next >= 10 else "#7b8476"
    soil_gauge = gauge_html(soil_now) if not soil_missing else ""
    kpis_html = (
        kpi_card("soil", "Humidité du sol",
                 f'{soil_txt}<span class="kpi-unit">m³/m³</span>', slab, scol, extra=soil_gauge)
        + kpi_card("ndvi", "État du couvert · NDVI", nd_txt, nd_lab, nd_col)
        + kpi_card("rain", "Pluie prévue · 7 j",
                   f'{rain_next:.0f}<span class="kpi-unit">mm</span>', "prévision", rain_col)
        + kpi_card("et0", "Demande ET₀ · 7 j",
                   f'{et0_next:.0f}<span class="kpi-unit">mm</span>', et0_lab, et0_col)
    )
    st.markdown(f'<div class="kpis">{kpis_html}</div>', unsafe_allow_html=True)

    alerts = build_alerts(soil_now, soil_missing, anomaly, action, headline, rain_next, et0_next,
                          soil_fc_min, ndvi=ndvi_val)
    if fsev == "recommandée":
        alerts = [a for a in alerts if a[0] != "ok"]
        alerts.append(("attention", "Fertilisation azotée à envisager",
                       "Vigueur limitée avec une eau suffisante — évaluer un apport d'azote."))
    active = [x for x in alerts if x[0] != "ok"]
    worst = max((s for s, _, _ in alerts), key=lambda s: SEV_RANK[s])
    next_check = (pd.Timestamp.now().normalize()
                  + pd.Timedelta(days={"critique": 1, "attention": 2, "info": 3, "ok": 5}[worst])).date()
    n_txt = (f"{len(active)} alerte" + ("s" if len(active) > 1 else "") + " active"
             + ("s" if len(active) > 1 else "")) if active else "aucune alerte"
    st.markdown('<div class="alert-head"><div class="eyebrow" style="margin:0">Alertes</div>'
                f'<div class="n">{n_txt} · prochaine vérification conseillée le {next_check}</div></div>',
                unsafe_allow_html=True)
    rows = "".join(f'<div class="alert" style="--s:{SEV[s]}"><div class="dot"></div>'
                   f'<div class="txt"><b>{t}</b><span>{d}</span></div><div class="sev">{s}</div></div>'
                   for s, t, d in alerts)
    st.markdown(f'<div class="alerts">{rows}</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Humidité du sol · tendance</div>', unsafe_allow_html=True)
    cd = hist.dropna().rename(columns={hist.columns[1]: "sol"}) if hist is not None and len(hist) else None
    if cd is not None and len(cd):
        cd.columns = ["date", "sol"]
        try:
            import altair as alt
            band = alt.Chart(pd.DataFrame({"y0": [0], "y1": [0.20]})).mark_rect(
                color="#b5482a", opacity=0.06).encode(y="y0:Q", y2="y1:Q")
            line = alt.Chart(cd).mark_line(color="#2f6b45", strokeWidth=2.2).encode(
                x=alt.X("date:T", title=None), y=alt.Y("sol:Q", title="m³/m³", scale=alt.Scale(zero=False)))
            thr = alt.Chart(pd.DataFrame({"y": [0.20, 0.30]})).mark_rule(
                strokeDash=[4, 4], color="#9aa48f").encode(y="y:Q")
            st.altair_chart((band + thr + line).properties(height=280).configure_view(strokeWidth=0),
                            width="stretch")
        except Exception:
            st.line_chart(cd.set_index("date"), height=280)
    st.caption(f"Humidité du sol : {soil_source}. Sous 0,20 m³/m³ la culture est en stress ; "
               "au-dessus de 0,30 elle est à l'aise.")
    if not ndvi_available:
        st.caption("Indices de végétation (NDVI) non calculés en mode rapide — lancez l'analyse "
                   "satellite complète pour les obtenir sur cette parcelle.")

    # ---- per-pixel NDVI map overlay (satellite mode) ----
    if ndvi_map and map_bounds and HAVE_MAP:
        st.markdown('<div class="eyebrow">Carte NDVI de la parcelle</div>', unsafe_allow_html=True)
        try:
            from streamlit.components.v1 import html as st_html
            mnlon, mnlat, mxlon, mxlat = map_bounds
            nm = folium.Map(location=[(mnlat + mxlat) / 2, (mnlon + mxlon) / 2],
                            zoom_start=14, tiles="OpenStreetMap", control_scale=True)
            folium.raster_layers.ImageOverlay(
                image=ndvi_map, bounds=[[mnlat, mnlon], [mxlat, mxlon]], opacity=0.85).add_to(nm)
            folium.Rectangle([[mnlat, mnlon], [mxlat, mxlon]], color="#2f6b45",
                             fill=False, weight=1.5).add_to(nm)
            st_html(nm._repr_html_(), height=380)
            st.caption("NDVI par pixel (Sentinel-2, scène la plus claire) — rouge = couvert faible, "
                       "vert = couvert vigoureux. Pixels nuageux masqués (transparents). "
                       "Révèle la variabilité intra-parcellaire.")
        except Exception as e:
            st.caption(f"Carte NDVI indisponible : {type(e).__name__}")

    st.markdown('<div class="eyebrow">Explication détaillée</div>', unsafe_allow_html=True)
    st.write("L'assistant relit l'ensemble des données de cette parcelle et rédige une recommandation argumentée.")
    if st.button("Obtenir l'explication détaillée", key="explain"):
        with st.spinner("Analyse de la parcelle par l'agent…"):
            try:
                st.success("Explication de l'assistant")
                st.write(agent_fn())
            except Exception as e:
                st.error(f"Assistant indisponible : {type(e).__name__}: {e}\n\n"
                         "Vérifie que la clé GROQ_API_KEY est bien définie.")


# ============================================================ header
st.markdown(f"""
<div class="topbar">
  <div class="brandwrap">
    <div class="mark">{MARK_SVG}</div>
    <div class="brand"><small>Aide à la décision agronomique</small><b>AgriSat · Irrigation &amp; Fertilisation</b></div>
  </div>
  <div class="meta">Fusion satellite · météo · capteurs de sol<br>Couverture <b>mondiale</b></div>
</div>""", unsafe_allow_html=True)

# ============================================================ state
st.session_state.setdefault("center", CFG_CENTER)
st.session_state.setdefault("zoom", 11)
st.session_state.setdefault("sel", None)

# ---- STEP 1: search / examples ----
st.markdown('<div class="rail"><span class="idx">01</span><span class="t">Localiser la parcelle</span>'
            '<span class="ln"></span></div>', unsafe_allow_html=True)
sc1, sc2 = st.columns([5, 1])
q = sc1.text_input("lieu", label_visibility="collapsed",
                   placeholder="Rechercher un pays, une région ou une ville — ex. « Lleida, Espagne »")
if sc2.button("Rechercher") and q.strip():
    st.session_state.geo = geocode(q.strip())
geo = st.session_state.get("geo") or []
if geo:
    labels = [", ".join(p for p in [g["name"], g["admin1"], g["country"]] if p) for g in geo]
    gi = st.selectbox("Résultats de recherche", range(len(geo)), format_func=lambda i: labels[i])
    if st.button("Centrer la carte sur ce lieu"):
        set_selection(geo[gi]["lat"], geo[gi]["lon"])
        st.session_state.geo = []
        st.rerun()

# ---- STEP 2: draw ----
st.markdown('<div class="rail"><span class="idx">02</span><span class="t">Délimiter la parcelle</span>'
            '<span class="ln"></span></div>', unsafe_allow_html=True)
if HAVE_MAP:
    st.caption("Cliquez l'outil rectangle (icône carrée, en haut à droite de la carte) puis tracez la parcelle.")
    m = folium.Map(location=st.session_state.center, zoom_start=st.session_state.zoom,
                   tiles="OpenStreetMap", control_scale=True)
    Draw(export=False, position="topright",
         draw_options={"rectangle": {"shapeOptions": {"color": "#2f6b45"}}, "polygon": False,
                       "polyline": False, "circle": False, "circlemarker": False, "marker": False},
         edit_options={"edit": False}).add_to(m)
    if st.session_state.sel:
        b = st.session_state.sel["bbox"]; c = st.session_state.sel["centroid"]
        folium.Rectangle([[b[1], b[0]], [b[3], b[2]]], color="#2f6b45", fill=True,
                         fill_opacity=0.12, weight=2).add_to(m)
        folium.CircleMarker([c[1], c[0]], radius=6, color="#2f6b45", fill=True, fill_opacity=0.9).add_to(m)
    out = st_folium(m, height=460, use_container_width=True, key="map",
                    returned_objects=["last_active_drawing"])
    if out and out.get("last_active_drawing"):
        bb = bbox_from_drawing(out["last_active_drawing"])
        if bb and (st.session_state.sel is None or bb != st.session_state.sel["bbox"]):
            st.session_state.sel = {"bbox": bb, "centroid": (round((bb[0] + bb[2]) / 2, 5),
                                                             round((bb[1] + bb[3]) / 2, 5))}
            st.rerun()
else:
    st.warning("Carte interactive non installée. Exécutez : pip install streamlit-folium folium — "
               "puis relancez. En attendant, saisissez les coordonnées :")
    c1, c2, c3 = st.columns(3)
    latv = c1.number_input("Latitude", value=float(st.session_state.center[0]), format="%.5f")
    lonv = c2.number_input("Longitude", value=float(st.session_state.center[1]), format="%.5f")
    hk = c3.number_input("Demi-côté (km)", value=1.5, min_value=0.2, max_value=10.0, step=0.5)
    st.session_state.sel = {"bbox": box_around(latv, lonv, hk), "centroid": (round(lonv, 5), round(latv, 5))}

# ---- STEP 3: analyse ----
sel = st.session_state.sel
st.markdown('<div class="rail"><span class="idx">03</span><span class="t">Lancer l\'analyse</span>'
            '<span class="ln"></span></div>', unsafe_allow_html=True)
if sel:
    clon, clat = sel["centroid"]; b = sel["bbox"]
    st.markdown(f'<div class="selcard">{MOTIF_SVG.replace("reco-motif", "reco-motif").replace("#fff", "#2f6b45")}'
                f'<div class="lbl">Parcelle sélectionnée</div>'
                f'<div class="coord">{clat:.4f}, {clon:.4f}</div>'
                f'<div class="sub">Emprise ~{abs(b[2]-b[0])*85:.1f} × {abs(b[3]-b[1])*111:.1f} km</div></div>',
                unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    if a1.button("Analyse rapide (temps réel)"):
        st.session_state.mode = "live"; st.session_state.analyzed = sel
        st.session_state.pop("full", None)
    if a2.button("Analyse satellite complète (NDVI)"):
        st.session_state.mode = "full"; st.session_state.analyzed = sel
        with st.spinner("Analyse satellite en cours — l'imagerie Sentinel-2 est lue sur une zone d'environ "
                        "2,5 km autour du centre, sur les dates récentes les plus claires…"):
            try:
                csv, cfgpath, soil_src, zbbox, zname = run_full_pipeline(clat, clon, sel["bbox"])
                st.session_state.full = {"csv": csv, "centroid": (clon, clat), "soil_src": soil_src,
                                         "bbox": list(zbbox), "name": zname}
                if not csv:
                    st.error("Le pipeline n'a pas produit de table alignée (imagerie peut-être trop nuageuse "
                             "sur la période). Essayez une autre parcelle ou l'analyse rapide.")
            except Exception as e:
                st.error(f"Analyse satellite indisponible : {type(e).__name__}: {e}")
    st.caption("Rapide = recommandation immédiate (météo + humidité de sol mondiales). "
               "Satellite = ajoute les indices de végétation (NDVI/EVI/NDWI) et l'agent.")
else:
    st.info("Sélectionnez d'abord une parcelle (Étapes 1–2) pour activer l'analyse.")

# ============================================================ RESULTS
st.markdown('<hr>', unsafe_allow_html=True)
mode = st.session_state.get("mode")
asel = st.session_state.get("analyzed")

if mode == "full" and st.session_state.get("full", {}).get("csv"):
    full = st.session_state["full"]
    clon, clat = full["centroid"]
    try:
        df = pd.read_csv(full["csv"], parse_dates=["date"]).sort_values("date")
        soil = df["soil_moisture_sensor_mean"].dropna()
        soil_now = float(soil.iloc[-1]) if len(soil) else float("nan")
        soil_missing = not len(soil)
        anomaly, ndvi = ndvi_anomaly_from_df(df)
        hist = df[["date", "soil_moisture_sensor_mean"]]
        _png = Path(full["csv"]).parent / f'{full.get("name", "")}_ndvi_map.png'
        ndvi_map = str(_png) if _png.exists() else None
        render_results(clat, clon, soil_now=soil_now, soil_missing=soil_missing, hist=hist,
                       ndvi=ndvi, ndvi_available=("ndvi" in df and df["ndvi"].notna().any()),
                       anomaly=anomaly, source_label="Analyse satellite",
                       agent_fn=lambda: _agent_from_csv(full["csv"], clon, clat),
                       soil_source=SOIL_SRC_LABEL.get(full.get("soil_src"), "pipeline complet"),
                       ndvi_map=ndvi_map, map_bounds=full.get("bbox"))
    except Exception as e:
        st.error(f"Lecture des résultats impossible : {type(e).__name__}: {e}")

elif mode == "live" and asel:
    clon, clat = asel["centroid"]
    with st.spinner("Analyse rapide (données mondiales en temps réel)…"):
        try:
            dd, soil_daily, soil_key = fetch_live(round(clat, 3), round(clon, 3))
            ok = True
        except Exception as e:
            st.error(f"Données indisponibles pour ce lieu : {type(e).__name__}: {e}"); ok = False
    if ok:
        today = pd.Timestamp.now().normalize()
        if soil_daily is not None and len(soil_daily):
            hist = soil_daily[soil_daily["date"] <= today][["date", "sm"]]
            soil_now = float(hist["sm"].iloc[-1]) if len(hist) else float(soil_daily["sm"].iloc[-1])
            soil_missing = False
        else:
            hist = pd.DataFrame(columns=["date", "sm"]); soil_now, soil_missing = float("nan"), True
        render_results(clat, clon, soil_now=soil_now, soil_missing=soil_missing, hist=hist,
                       ndvi=float("nan"), ndvi_available=False, anomaly=False,
                       source_label="Analyse rapide",
                       agent_fn=lambda: agent_live(clat, clon),
                       soil_source=f"Open-Meteo modélisée ({soil_key or 'n/d'})")

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div style="color:#7b8476; font-size:.8rem;">Fusion satellite, météo et capteurs de sol · '
            'aide à la décision d\'irrigation et de fertilisation · sélection de parcelle mondiale</div>', unsafe_allow_html=True)
