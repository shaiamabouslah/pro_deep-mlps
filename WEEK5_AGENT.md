# Week 5 — The fusion agent

A **LangGraph ReAct agent** (mandated stack §6) that fuses the three signals into
an irrigation / fertilization recommendation **with a traceable explanation**.
Its brain is your local **Ollama** model — free, offline, no API key.

## What it does

The agent is given a question ("should I irrigate?") and reasons by calling four
tools, each backed by the real data from Weeks 1-4:

| Tool | Returns | Source |
|---|---|---|
| `get_current_field_status` | latest NDVI/EVI/NDWI, soil moisture, last-7-day rain & ET0 | aligned table |
| `get_weather_outlook` | **real** rain & ET0 forecast for the coming days | Open-Meteo (live) |
| `get_soil_moisture_forecast` | where soil moisture is heading | Chronos-2 |
| `check_ndvi_anomaly` | is canopy vigour abnormally low? | NDVI history |

It then outputs, in French: a **recommendation**, a **justification citing the
actual numbers**, and a **confidence/limits** line. Every claim traces to a tool.

## 1. Prerequisites

Ollama must be running and the model pulled (you already have both):

```bash
ollama list          # should show qwen2.5:7b and qwen2.5:3b
```

Install the agent libraries:

```bash
.venv\Scripts\activate
pip install -r requirements-agent.txt
```

## 2. Run

```bash
python -m src.agent --config config/field_config.yaml
```

Options:
- `--model qwen2.5:3b` — lighter/faster if the 7b is slow on your machine.
- `--days 7` — decision horizon.
- `--query "..."` — ask your own question.

The first run may be slow (the model loads into RAM). The console prints the
recommendation and, at the end, the **trace of tools the agent called** — useful
evidence for the report.

## 3. Example of expected output

```
RECOMMANDATION DE L'AGENT DE FUSION
- Recommandation: Attendre. Ne pas irriguer avant 3-4 jours.
- Justification: humidite du sol actuelle 0.41 m3/m3 (confortable), prevision
  a 7 jours 0.38 (-0.03, legere baisse), pluie attendue 12 mm sur 5 jours,
  ET0 21 mm; NDVI 0.44 sans anomalie. La pluie couvre une bonne part de la demande.
- Confiance et limites: prevision meteo et sol incertaines au-dela de ~5 jours.
Outils appelés (trace): get_current_field_status, get_weather_outlook,
get_soil_moisture_forecast, check_ndvi_anomaly
```

## 4. Troubleshooting

- **"Connection refused" / Ollama errors** → Ollama isn't running. Launch the
  Ollama app (or run `ollama serve`), then retry.
- **Very slow / freezes** → switch to `--model qwen2.5:3b`. Running a 7b model +
  Chronos at once is heavy; the 3b is much lighter.
- **The agent doesn't call tools / rambles** → small models occasionally skip
  tool calls. Re-run (temperature is 0 but sampling can still vary), or use the
  7b which is more reliable at tool-calling than the 3b.
- **Chronos forecast tool slow** → it loads the model each call; that's expected.
  The tool has a lightweight trend fallback if Chronos fails.

## Next (Weeks 6-8)

- **Week 6** — richer, structured explanations (attach the evidence to each
  recommendation explicitly) and evaluate the agent on GeoBenchX-style tasks.
- **Week 7** — map dashboard (Streamlit + Folium) surfacing indices, forecast,
  recommendation and NDVI-anomaly alerts.
- **Week 8** — final validation on a sample field/season + report.
