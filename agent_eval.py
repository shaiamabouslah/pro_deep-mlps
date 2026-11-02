"""Decision-quality evaluation of the fusion agent on a set of labelled scenarios,
including stress-tests, scored against a naive baseline.

    python agent_eval.py --backend groq --model openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STRESS = 0.20        # m3/m3 -- below this the crop is water-stressed
COMFORT = 0.30       # above this soil moisture is comfortable

# 1. Labelled decision scenarios
#    action in {"irriguer","attendre","surveiller"}
#    fields: soil_now, soil_fc (7-day forecast), rain (mm/7d), et0 (mm/7d),
#            ndvi, ndvi_mean, ndvi_std, tag, expected, why
def _s(id, tag, soil_now, soil_fc, rain, et0, ndvi, ndvi_mean, ndvi_std, expected, why):
    return dict(id=id, tag=tag, soil_now=soil_now, soil_fc=soil_fc, rain=rain, et0=et0,
                ndvi=ndvi, ndvi_mean=ndvi_mean, ndvi_std=ndvi_std, expected=expected, why=why)

SCENARIOS = [
    # Easy / unambiguous (both the agent and the naive rule should get these)
    _s("E1", "easy", 0.38, 0.36, 18, 20, 0.62, 0.60, 0.05, "attendre",
       "soil comfortable and stable, rain covers demand"),
    _s("E2", "easy", 0.16, 0.14, 2, 34, 0.40, 0.55, 0.06, "irriguer",
       "already stressed, drying further, no rain, high ET0"),
    _s("E3", "easy", 0.33, 0.31, 25, 18, 0.58, 0.57, 0.05, "attendre",
       "comfortable, heavy rain incoming"),

    # Forecast-driven: comfortable NOW but heading into stress (naive rule MISSES)
    _s("F1", "forecast", 0.31, 0.19, 3, 33, 0.50, 0.52, 0.05, "irriguer",
       "comfortable now but forecast drops below stress, dry + high ET0"),
    _s("F2", "forecast", 0.29, 0.21, 4, 30, 0.48, 0.50, 0.05, "irriguer",
       "marginal, forecast to stress edge, little rain, high demand"),

    # Rain-saves-it: low NOW but big rain coming (naive rule OVER-irrigates)
    _s("R1", "rain", 0.22, 0.30, 28, 16, 0.51, 0.52, 0.05, "attendre",
       "marginal now but strong rain will recharge; don't waste water"),
    _s("R2", "rain", 0.19, 0.28, 22, 15, 0.47, 0.49, 0.05, "attendre",
       "stressed now but heavy rain imminent -> wait and re-check"),

    # NDVI anomaly: canopy stress but soil is fine (water isn't the problem)
    _s("A1", "anomaly", 0.34, 0.33, 12, 19, 0.30, 0.55, 0.06, "surveiller",
       "clear NDVI anomaly but soil comfortable -> investigate (pest/nutrient), not irrigation"),

    # Heatwave stress-test: very high ET0 pulling soil down fast
    _s("H1", "heatwave", 0.27, 0.18, 1, 45, 0.49, 0.50, 0.05, "irriguer",
       "heatwave: extreme ET0, soil crashing into stress, no rain"),

    # Drought onset stress-test: steady decline, persistent dryness
    _s("D1", "drought_onset", 0.24, 0.17, 0, 32, 0.44, 0.50, 0.06, "irriguer",
       "drought onset: monotonic drydown to stress, zero rain"),

    # Sensor dropout stress-test: soil reading missing/unreliable
    _s("S1", "sensor_dropout", None, None, 5, 28, 0.50, 0.51, 0.05, "surveiller",
       "soil sensor missing -> cannot confidently decide; monitor / restore sensor before acting"),

    # Conflicting signals stress-test: low NDVI but HIGH soil moisture
    _s("C1", "conflicting", 0.36, 0.34, 8, 22, 0.28, 0.54, 0.06, "surveiller",
       "low NDVI (stress) but soil is wet -> not water stress; investigate before irrigating"),
    _s("C2", "conflicting", 0.18, 0.17, 0, 20, 0.60, 0.58, 0.05, "irriguer",
       "healthy canopy but soil genuinely stressed, dry -> irrigate"),
]

ACTIONS = ("irriguer", "attendre", "surveiller")


# 2. Baselines / parsing
def naive_baseline(sc: dict) -> str:
    """Ignores forecast, weather and NDVI: irrigate iff current soil < 0.25."""
    if sc["soil_now"] is None:
        return "irriguer"   # naive rule has no notion of 'uncertain' -> a failure mode we want to expose
    return "irriguer" if sc["soil_now"] < 0.25 else "attendre"


def parse_action(text: str) -> str | None:
    """Extract the action. The eval prompt asks the agent to end with
    'ACTION: <irriguer|attendre|surveiller>'; fall back to keyword search."""
    if not text:
        return None
    m = re.search(r"ACTION\s*[:=]\s*(irriguer|attendre|surveiller)", text, re.I)
    if m:
        return m.group(1).lower()
    t = text.lower()
    # priority: explicit irrigation > monitor/investigate > wait
    if re.search(r"\birrigu", t):
        return "irriguer"
    if re.search(r"surveill|investigat|inspect|v[ée]rifi", t):
        return "surveiller"
    if re.search(r"attendre|patienter|ne pas irriguer", t):
        return "attendre"
    return None


# 3. Scenario-backed tools + agent runner
def build_scenario_tools(sc: dict):
    """Four @tool functions that return THIS scenario's numbers -- same names and
    contract as src/agent.py, so the agent tool-calls identically."""
    from langchain_core.tools import tool

    @tool
    def get_current_field_status() -> str:
        """Latest field state: NDVI/EVI/NDWI, current soil moisture, recent rain & ET0."""
        soil = "unavailable (sensor dropout)" if sc["soil_now"] is None else f"{sc['soil_now']:.3f} m3/m3"
        return (f"NDVI={sc['ndvi']:.2f}; soil moisture (sensor): {soil}; "
                f"last 7 days rain={sc['rain']:.0f} mm, ET0={sc['et0']:.0f} mm.")

    @tool
    def get_weather_outlook(days: int = 7) -> str:
        """Weather forecast: total rain (mm) and ET0 (mm) for the coming days."""
        return f"Weather outlook ({days} d): total rain={sc['rain']:.0f} mm, total ET0={sc['et0']:.0f} mm."

    @tool
    def get_soil_moisture_forecast(days: int = 7) -> str:
        """Forecast of soil moisture for the coming days."""
        if sc["soil_fc"] is None:
            return "Soil-moisture forecast unavailable (no reliable sensor history)."
        now = sc["soil_now"]
        trend = "drying" if (now is not None and sc["soil_fc"] < now) else "recharging"
        base = f"Soil-moisture forecast (day {days}) = {sc['soil_fc']:.3f} m3/m3 ({trend})."
        return base if now is None else f"Now={now:.3f} -> {base}"

    @tool
    def check_ndvi_anomaly() -> str:
        """Whether the latest NDVI is abnormally low vs the field's history."""
        z = (sc["ndvi"] - sc["ndvi_mean"]) / sc["ndvi_std"] if sc["ndvi_std"] else 0.0
        return (f"Current NDVI={sc['ndvi']:.2f}; historical mean={sc['ndvi_mean']:.2f} "
                f"(std={sc['ndvi_std']:.2f}); z={z:.2f}. Anomaly (stress) flag: {z <= -1.5}")

    return [get_current_field_status, get_weather_outlook,
            get_soil_moisture_forecast, check_ndvi_anomaly]


EVAL_SUFFIX = ("\n\nAprès ta justification, termine IMPÉRATIVEMENT par une seule ligne :\n"
               "ACTION: irriguer   (ou)   ACTION: attendre   (ou)   ACTION: surveiller")


def run_agent_on_scenario(sc: dict, model_name: str, backend: str = "ollama"):
    """Returns (predicted_action, n_tools_called, all_four_called, raw_text)."""
    from langgraph.prebuilt import create_react_agent
    from src.agent import SYSTEM_PROMPT, get_llm

    tools = build_scenario_tools(sc)
    llm = get_llm(backend, model_name)
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    result = agent.invoke({"messages": [("user",
        "Faut-il irriguer cette parcelle dans les prochains jours ?" + EVAL_SUFFIX)]})

    msgs = result["messages"]
    text = msgs[-1].content
    called = []
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            called.append(tc.get("name"))
    tool_names = {"get_current_field_status", "get_weather_outlook",
                  "get_soil_moisture_forecast", "check_ndvi_anomaly"}
    all_four = tool_names.issubset(set(called))
    return parse_action(text), len(set(called)), all_four, text


# 4. Main
def main():
    p = argparse.ArgumentParser(description="Agent decision-quality evaluation")
    p.add_argument("--backend", default="ollama", choices=["ollama", "groq"],
                   help="ollama=local 7B (free/offline) | groq=API (needs GROQ_API_KEY, ~70B, stronger)")
    p.add_argument("--model", default=None,
                   help="model id; default qwen2.5:7b (ollama) or llama-3.3-70b-versatile (groq)")
    p.add_argument("--limit", type=int, default=0, help="run only the first N scenarios")
    p.add_argument("--dry-run", action="store_true",
                   help="offline plumbing test: no LLM, uses the naive rule as the 'agent'")
    p.add_argument("--out", default="data/processed")
    args = p.parse_args()
    model = args.model or ("llama-3.3-70b-versatile" if args.backend == "groq" else "qwen2.5:7b")
    if not args.dry_run:
        logger.info("Backend=%s  model=%s", args.backend, model)

    scenarios = SCENARIOS[: args.limit] if args.limit else SCENARIOS
    rows = []
    for i, sc in enumerate(scenarios, 1):
        if args.dry_run:
            pred, ntools, all_four, _ = naive_baseline(sc), 0, False, ""
        else:
            logger.info("Scenario %d/%d [%s] ...", i, len(scenarios), sc["id"])
            try:
                pred, ntools, all_four, _ = run_agent_on_scenario(sc, model, args.backend)
            except Exception as e:
                logger.warning("  scenario %s failed: %s: %s", sc["id"], type(e).__name__, e)
                pred, ntools, all_four = None, 0, False
        base = naive_baseline(sc)
        rows.append({
            "id": sc["id"], "tag": sc["tag"], "expected": sc["expected"],
            "agent": pred, "agent_correct": pred == sc["expected"],
            "naive": base, "naive_correct": base == sc["expected"],
            "tools_called": ntools, "plan_complete": all_four, "why": sc["why"],
        })

    df = pd.DataFrame(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = "dryrun" if args.dry_run else args.backend
    results_path = out / f"agent_eval_{tag}.csv"
    df.to_csv(results_path, index=False)

    n = len(df)
    agent_acc = df["agent_correct"].mean() if not args.dry_run else float("nan")
    naive_acc = df["naive_correct"].mean()
    plan_rate = df["plan_complete"].mean() if not args.dry_run else float("nan")

    print("\n=== Agent decision-quality evaluation ===")
    print(df[["id", "tag", "expected", "agent", "agent_correct", "naive", "naive_correct", "plan_complete"]]
          .to_string(index=False))
    print(f"\nScenarios: {n}")
    if not args.dry_run:
        print(f"AGENT accuracy         : {agent_acc:.0%}")
        print(f"Naive-baseline accuracy: {naive_acc:.0%}   (irrigate iff soil<0.25)")
        print(f"Plan completion (4/4 tools called): {plan_rate:.0%}")
        # where fusion beats the naive rule
        gained = df[(df.agent_correct) & (~df.naive_correct)]
        lost = df[(~df.agent_correct) & (df.naive_correct)]
        print(f"\nCases the agent gets right that the naive rule gets wrong: "
              f"{len(gained)}  ({', '.join(gained['id']) or '-'})")
        print(f"Cases the naive rule gets right that the agent gets wrong: "
              f"{len(lost)}  ({', '.join(lost['id']) or '-'})")
        stress = df[df.tag.isin(["heatwave", "drought_onset", "sensor_dropout", "conflicting"])]
        if len(stress):
            print(f"\nStress-test accuracy (heatwave/drought/dropout/conflicting): "
                  f"{stress['agent_correct'].mean():.0%}  ({stress['agent_correct'].sum()}/{len(stress)})")
        print("\nConfusion (rows=expected, cols=agent):")
        print(pd.crosstab(df["expected"], df["agent"]).to_string())
    else:
        print(f"[dry-run] Naive-baseline accuracy: {naive_acc:.0%} "
              f"(plumbing OK -- run without --dry-run against Ollama for the real agent).")
    print(f"\nWrote: {results_path}")


if __name__ == "__main__":
    main()
