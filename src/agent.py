"""Fusion agent: a LangGraph ReAct agent that fuses vegetation indices, weather and
forecasted soil moisture into an irrigation and fertilization recommendation with a
traceable explanation. Backends: a local model via Ollama, or a hosted model via Groq.

    python -m src.agent --config config/field_config.yaml
    python -m src.agent --backend groq --model openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import requests

from .config_loader import load_config
from .indices import flag_ndvi_anomaly

# Load GROQ_API_KEY (and other keys) from a project-root .env if present, so the
# key can be set once in a file instead of exported in every shell session.
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    load_dotenv(_Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# module-level context so the @tool functions can reach the config/data
_CFG = None
_CSV = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _aligned() -> pd.DataFrame:
    df = pd.read_csv(_CSV, parse_dates=["date"]).sort_values("date")
    return df


def _target_series() -> pd.Series:
    df = _aligned().set_index("date")
    s = df["soil_moisture_sensor_mean"].astype(float).asfreq("D")
    return s.interpolate(method="time").ffill().bfill()


# ---------------------------------------------------------------------------
# TOOLS  (imported lazily so the module loads even before langchain is installed)
# ---------------------------------------------------------------------------
def _build_tools():
    from langchain_core.tools import tool

    @tool
    def get_current_field_status() -> str:
        """Latest observed state of the field: most recent NDVI/EVI/NDWI, current
        soil moisture, and rainfall + ET0 accumulated over the last 7 days."""
        df = _aligned()
        last = df.iloc[-1]
        last7 = df.tail(7)
        if "ndvi" in df and df["ndvi"].notna().any():
            obs = df[df.get("ndvi_observed", False) == True] if "ndvi_observed" in df else df
            obs = obs[obs["ndvi"].notna()]
            ndvi_line = (f"NDVI={obs['ndvi'].iloc[-1]:.2f} (last real image {obs['date'].iloc[-1].date()})"
                         if len(obs) else "NDVI unavailable")
        else:
            ndvi_line = "NDVI unavailable (no satellite data for this field)"
        evi_txt = f"{last['evi']:.2f}" if "evi" in df and pd.notna(last.get("evi")) else "n/d"
        ndwi_txt = f"{last['ndwi']:.2f}" if "ndwi" in df and pd.notna(last.get("ndwi")) else "n/d"

        # Weather/soil columns are present only when the aligned table carries them
        # (they can be absent for a fresh drawn parcel); degrade to "n/d" rather than crash.
        def _sum(col):
            return f"{last7[col].sum():.1f}" if col in df else "n/d"

        def _mean(col):
            return f"{last7[col].mean():.1f}" if col in df else "n/d"

        soil_txt = (f"{last['soil_moisture_sensor_mean']:.3f} m3/m3"
                    if "soil_moisture_sensor_mean" in df and pd.notna(last.get("soil_moisture_sensor_mean"))
                    else "n/d")
        return (
            f"Date: {last['date'].date()}\n"
            f"{ndvi_line}; EVI={evi_txt}; NDWI={ndwi_txt}\n"
            f"Soil moisture (sensor): {soil_txt}\n"
            f"Last 7 days: rain={_sum('precip_sum_mm')} mm, "
            f"ET0={_sum('et0_sum_mm')} mm, "
            f"mean temp={_mean('temp_mean_c')} C"
        )

    @tool
    def get_weather_outlook(days: int = 7) -> str:
        """Real weather FORECAST for the coming days at the field: daily rainfall
        (mm) and reference evapotranspiration ET0 (mm). Source: Open-Meteo."""
        lon, lat = _CFG.centroid
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon,
                        "daily": "precipitation_sum,et0_fao_evapotranspiration",
                        "forecast_days": int(min(max(days, 1), 16)), "timezone": "auto"},
                timeout=30,
            )
            r.raise_for_status()
            d = r.json()["daily"]
            rain, et0 = d["precipitation_sum"], d["et0_fao_evapotranspiration"]
            lines = [f"  {dt}: rain={p} mm, ET0={e} mm" for dt, p, e in zip(d["time"], rain, et0)]
            return (f"Weather outlook ({days} d): total rain={np.nansum(rain):.1f} mm, "
                    f"total ET0={np.nansum(et0):.1f} mm\n" + "\n".join(lines))
        except Exception as e:
            return f"Weather outlook unavailable ({e})."

    @tool
    def get_soil_moisture_forecast(days: int = 7) -> str:
        """Forecast of soil moisture for the next days (Chronos-2). Reports the
        predicted values and whether the soil is drying or recharging."""
        s = _target_series()
        try:
            from .forecast import run_chronos
            point, _ = run_chronos(s, int(days))
            point = np.asarray(point, float)[:days]
        except Exception as e:
            # light fallback: persist last value with the recent daily drift
            drift = float(np.mean(np.diff(s.values[-14:]))) if len(s) > 15 else 0.0
            point = s.values[-1] + drift * np.arange(1, days + 1)
            logger.warning("Chronos forecast failed (%s) -- using trend fallback.", e)
        trend = "drying" if point[-1] < s.values[-1] else "recharging"
        delta = point[-1] - s.values[-1]
        vals = ", ".join(f"{v:.3f}" for v in point)
        return (f"Soil-moisture forecast ({days} d): [{vals}] m3/m3\n"
                f"Now={s.values[-1]:.3f} -> day{days}={point[-1]:.3f} "
                f"({delta:+.3f}, {trend})")

    @tool
    def check_ndvi_anomaly() -> str:
        """Check whether the latest NDVI is abnormally low versus the field's own
        history (a canopy-stress early warning)."""
        df = _aligned()
        if "ndvi" not in df or df["ndvi"].notna().sum() < 5:
            return "Not enough NDVI history to assess an anomaly."
        obs = df[df.get("ndvi_observed", True) == True]["ndvi"].dropna() if "ndvi_observed" in df \
            else df["ndvi"].dropna()
        hist_mean, hist_std = float(obs[:-1].mean()), float(obs[:-1].std())
        current = float(obs.iloc[-1])
        anomaly = flag_ndvi_anomaly(current, hist_mean, hist_std)
        return (f"Current NDVI={current:.2f}; historical mean={hist_mean:.2f} "
                f"(std={hist_std:.2f}). Anomaly (stress) flag: {anomaly}")

    return [get_current_field_status, get_weather_outlook,
            get_soil_moisture_forecast, check_ndvi_anomaly]


SYSTEM_PROMPT = """You are an agronomy decision assistant for precision irrigation of a single corn field.
Use ONLY the four tools, and ALWAYS call all four before deciding:
1. get_current_field_status  2. get_weather_outlook  3. get_soil_moisture_forecast  4. check_ndvi_anomaly

Then decide by applying the rules IN ORDER. Base the decision on the FORECAST soil moisture and the
weather to come, NOT only on today's value -- act BEFORE stress, not after.

Soil-moisture zones (m3/m3): comfortable > 0.30, marginal 0.20-0.30, stressed < 0.20.

Choose EXACTLY ONE action:

- SURVEILLER (monitor / investigate -- do NOT irrigate) when the situation is ambiguous or NOT water-driven:
    * the soil-moisture sensor reading is missing/unreliable (you cannot decide safely), OR
    * NDVI is VERY LOW (< ~0.20) -- the field is likely BARE SOIL or a harvested/absent crop. You
      irrigate a CROP, not bare ground, so do NOT recommend irrigation on soil moisture alone even if
      the soil is dry; recommend verifying a crop is actually present first, OR
    * NDVI is anomalously low BUT soil moisture is adequate (> ~0.25) -- the stress is not water
      (likely pest/disease/nutrient), OR
    * the signals conflict (e.g. low NDVI but wet soil).

- IRRIGUER when water is genuinely the limiting factor -- ALL of:
    * a crop is actually present (NDVI >= ~0.20; if NDVI < ~0.20 choose SURVEILLER instead), AND
    * forecast soil moisture falls to/below ~0.22 (toward/into the stress zone), AND
    * little rain is expected (< ~10 mm), AND
    * evaporative demand ET0 is high.
  This holds EVEN IF today's soil is still comfortable: a forecast drop with no rain means irrigate NOW.

- ATTENDRE only when neither of the above applies: soil stays comfortable, OR meaningful rain
  (>= ~10 mm) will recharge it.

Do NOT default to ATTENDRE. If the IRRIGUER rule is met, say irriguer. If it is a SURVEILLER case, say surveiller.

FERTILIZATION (a SEPARATE decision from irrigation — decide it too):
- FERTILISER (envisager un apport d'azote) when the canopy under-performs but water is NOT the problem:
  NDVI is low (< ~0.45) AND soil moisture is adequate (>= ~0.22). A weak canopy with enough water points
  to a nutrient (nitrogen) limitation, not thirst. Suggest confirming with a soil test.
- DIFFÉRER (corriger l'eau d'abord) when NDVI is low BUT soil is dry (< ~0.22): the problem is water, not
  nutrients — fertilizing a water-stressed crop wastes input. Fix water first, reassess.
- AUCUN apport when NDVI is healthy (>= ~0.45), or when NDVI < ~0.20 (bare/harvested — no crop to feed).

Answer in FRENCH, concisely, in this structure:
- Recommandation irrigation: (irriguer / attendre / surveiller) + quand et combien si applicable.
- Recommandation fertilisation: (fertiliser / différer / aucun) + azote si applicable.
- Justification: cite the ACTUAL NUMBERS you obtained from the tools (soil moisture now and
  forecast, expected rain and ET0, NDVI). Every claim must trace to a tool result.
- Confiance et limites: one line on uncertainty.
Never invent numbers you did not get from a tool."""


def get_llm(backend: str = "ollama", model: str | None = None, temperature: float = 0.0):
    """Chat model for the chosen backend.

    backend='ollama' -> LOCAL model via Ollama (free, offline, no key; e.g. qwen2.5:7b).
    backend='groq'   -> Groq API (needs GROQ_API_KEY + internet, but much stronger models,
                        e.g. llama-3.3-70b-versatile, and very fast).
    """
    backend = (backend or "ollama").lower()
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "qwen2.5:7b", temperature=temperature)
    if backend == "groq":
        from langchain_groq import ChatGroq  # pip install langchain-groq ; set GROQ_API_KEY
        return ChatGroq(model=model or "openai/gpt-oss-120b", temperature=temperature)
    raise ValueError(f"Unknown backend '{backend}' -- use 'ollama' or 'groq'")


def run_agent(query: str, model_name: str, days: int, backend: str = "ollama"):
    from langgraph.prebuilt import create_react_agent

    tools = _build_tools()
    llm = get_llm(backend, model_name)
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    user_msg = f"{query} (horizon: {days} jours)"
    logger.info("Asking agent (%s): %s", model_name, user_msg)
    result = agent.invoke({"messages": [("user", user_msg)]})

    final = result["messages"][-1].content
    print("\n" + "=" * 70)
    print("RECOMMANDATION DE L'AGENT DE FUSION")
    print("=" * 70)
    print(final)
    print("=" * 70)

    # trace of which tools were called (for the report)
    calls = []
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            calls.append(tc.get("name"))
    if calls:
        print("Outils appelés (trace):", ", ".join(calls))
    return final


def main():
    global _CFG, _CSV
    p = argparse.ArgumentParser(description="Fusion agent (Ollama or Groq)")
    p.add_argument("--config", default="config/field_config.yaml")
    p.add_argument("--backend", default="groq", choices=["ollama", "groq"],
                   help="groq=API (needs GROQ_API_KEY, stronger — DEFAULT) | ollama=local (free/offline)")
    p.add_argument("--model", default=None,
                   help="model id; default openai/gpt-oss-120b (groq) or qwen2.5:7b (ollama)")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--query", default="Faut-il irriguer cette parcelle dans les prochains jours ? "
                                      "Donne une recommandation d'irrigation et de fertilisation.")
    args = p.parse_args()
    model = args.model or ("openai/gpt-oss-120b" if args.backend == "groq" else "qwen2.5:7b")

    _CFG = load_config(args.config)
    _CSV = _CFG.processed_dir / f"{_CFG.name}_aligned_daily.csv"
    if not _CSV.exists():
        raise FileNotFoundError(f"{_CSV} not found. Run `python -m src.pipeline` first.")

    run_agent(args.query, model, args.days, backend=args.backend)


if __name__ == "__main__":
    main()
