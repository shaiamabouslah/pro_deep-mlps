# Phase 1 (Semaines 1–2) — Guide pas à pas

Ce guide t'accompagne pour finir la **Phase 1** du projet. La Phase 1 a **trois
livrables** :

1. **Note « état de l'art » de 3–4 pages** (brief §3) — *à écrire AVANT le code*.
   → Document séparé (Word/PDF), **pas encore fait**.
2. **Semaine 1 — Ingestion + indices** → Sentinel-2 + météo + IoT, NDVI/EVI/NDWI.
   → **Fait** (ce dépôt).
3. **Semaine 2 — Série temporelle multi-source propre et alignée** → une seule
   table journalière fusionnant les 3 sources.
   → **Fait** (`src/align.py`, sortie `<champ>_aligned_daily.csv`).

Ce dépôt couvre la partie **code** (livrables 2 et 3). Le livrable 1 (la note)
reste à rédiger — voir la dernière section.

---

## Étape 0 — Vérifier que tout tourne, hors-ligne (5 min, aucun compte)

Depuis un terminal, à la racine du projet :

```bash
pip install -r requirements.txt

# démo bout-en-bout Semaine 1 -> Semaine 2 avec des données synthétiques
python demo_offline.py

# tests unitaires (aucun réseau)
python tests/test_indices.py
python tests/test_iot_simulator.py
python tests/test_ismn_ingest.py
python tests/test_align.py
```

Tu dois voir la table alignée résumée à l'écran et le fichier
`outputs/demo_aligned_daily.csv` créé. **C'est ta preuve que la chaîne complète
fonctionne** avant de dépenser le moindre quota d'API.

---

## Étape 1 — Créer les comptes (à faire tout de suite : il y a un délai)

| Source | Compte ? | Lien |
|---|---|---|
| **Copernicus Data Space** (Sentinel-2, stack mandaté) | Oui, gratuit | https://dataspace.copernicus.eu |
| **ISMN** (capteurs sol réels, optionnel) | Oui, gratuit | https://ismn.earth |
| **Open-Meteo** (météo + sol) | Non | — |

Le stack **mandaté** par le sujet, c'est Copernicus. Mais tu n'as pas besoin
d'attendre ton compte pour lancer le pipeline réel (voir Étape 3).

---

## Étape 2 — Pointer la config sur ton champ

Édite **uniquement** `config/field_config.yaml` :

```yaml
field:
  name: "mon_champ"
  bbox: { min_lon: .., min_lat: .., max_lon: .., max_lat: .. }
  centroid: { lon: .., lat: .. }
time_range:
  start_date: "2026-05-01"
  end_date:   "2026-07-01"
```

Aucun autre fichier à toucher : tout le pipeline est piloté par cette config.

---

## Étape 3 — Lancer le pipeline réel

Le pipeline a **deux fournisseurs** Sentinel-2 (clé `sentinel2.provider`) :

- `earth_search` (**par défaut**) : COGs publics AWS, **sans compte** → pour
  développer/tester **maintenant**. Mêmes données Sentinel-2 L2A.
- `copernicus` : le stack **mandaté** pour ton rendu final.

**a) Tout de suite, sans compte** (provider = `earth_search`, déjà par défaut) :

```bash
python -m src.pipeline --config config/field_config.yaml
```

Sortie dans `data/processed/` :
- `<champ>_weather_soil.csv` — météo + humidité sol horaire
- `<champ>_iot_soil_moisture.csv` (ou `_ismn_soil_moisture.csv`) — humidité sol capteurs
- `<champ>_sentinel2_indices.csv` — **série NDVI/EVI/NDWI par date** (Semaine 1)
- `<champ>_aligned_daily.csv` — **la table alignée (livrable Semaine 2)**

**b) Avant le rendu, bascule sur le stack mandaté** : mets
`sentinel2.provider: copernicus` dans la config, puis fournis tes identifiants
CDSE ou utilise la voie openEO (`src/sentinel2_ingest.fetch_indices_timeseries_openeo`,
gabarit fourni).

Le pipeline ne « plante » jamais : si une source échoue, il continue avec les
autres et écrit quand même la table alignée avec ce qu'il a.

---

## Étape 4 — Comprendre / vérifier la table alignée (Semaine 2)

`<champ>_aligned_daily.csv` = **une ligne par jour**, avec :

- `ndvi`, `evi`, `ndwi` — indices satellite (épars, **remplis en avant** jusqu'à
  10 jours) + colonne `ndvi_observed` = `True` seulement les jours d'une vraie image.
- `temp_mean_c`, `temp_min_c`, `temp_max_c`, `precip_sum_mm`, `et0_sum_mm`,
  `soil_temp_0cm_c` — agrégats météo journaliers.
- `soil_moisture_0_to_1cm_mean`, … — humidité sol **modélisée** (Open-Meteo) par profondeur.
- `soil_moisture_sensor_mean` + `soil_moisture_sensor_10cm/30cm/…` — humidité sol **capteurs**.

Contrôles qualité rapides à faire (et à mentionner dans ton rapport) :
- `ndvi_observed.sum()` = nombre de vraies images sur la période (revisite ~5 j).
- Taux de valeurs manquantes par colonne.
- Cohérence physique : NDVI ∈ [−1, 1], humidité sol ∈ [0, ~0.5] m³/m³.

C'est **cette table** que consommera le prévisionniste de la Semaine 3.

---

## Étape 5 — Le livrable qu'il te reste : la note « état de l'art »

Le sujet (§3) l'exige, 3–4 pages, à écrire **avant** de justifier tes choix.
Elle doit positionner ton projet face à :

- **Agents géospatiaux** : GeoBenchX (benchmark), GeoAgentic-RAG, ReAct/LangGraph.
- **Modèles de séries temporelles fondation** : Chronos-2 (Amazon, avec covariables),
  TimesFM (Google), Lag-Llama ; baselines supervisées TFT / N-HiTS.
- **Accès données** : Copernicus Data Space STAC, Google Earth Engine, Open-Meteo, TorchGeo.
- **RAG de prévision** : TS-RAG.

> Quand tu veux, je peux te rechercher l'info à jour sur chacun de ces systèmes
> et **rédiger un premier jet** de cette note de 3–4 pages avec toi.

---

## Récap Phase 1

| Livrable | État |
|---|---|
| Note état-de-l'art (3–4 p.) | ⬜ à faire |
| Semaine 1 — ingestion + indices | ✅ |
| Semaine 2 — table alignée multi-source | ✅ |

**Ensuite (Semaine 3)** : prévisionniste court-terme d'humidité sol / besoin en
eau (Chronos-2 / TimesFM en zéro-shot) vs baseline TFT/N-HiTS, entraîné sur
`<champ>_aligned_daily.csv`.
