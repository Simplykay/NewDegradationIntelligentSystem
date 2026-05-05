# CLAUDE.md — Cotton Seed Quality Intelligence System
## Complete Project Reference: Modeling · Dashboard · Deployment

> **Read fully before writing any code.** This single file replaces the previous
> three (CLAUDE_v2, CLAUDE_DASHBOARD_v2, CLAUDE_DEPLOYMENT_v2).

> **PROJECT STATE — VERSION 5 (May 2026):**
> - **Active sources: `lineage` + `cotton_weather_features.xlsx` ONLY**
> - `cotton_weather_features.xlsx` — pre-engineered aggregated weather table. 1,716 rows, 2019–2025, all 6 states. One row per `variety + state + pa_year`. **Replaces all previous weather files entirely** (`2026_cotton_with_weather`, `cottons3_2025`, `weathers3_2025` all dropped).
> - All s3 files and old weather files ignored entirely.
> - **`WG_Current` and `WG_Initial_7Day_Cnt` REMOVED from ALL models** — these test results are not available at prediction time.
> - `Variety` removed from all models — replaced by `rm` (Relative Maturity)
> - `NAWF` features removed entirely
> - `CT_Initial` validity rule: null if retest within 14 days
> - `pp14_cum_dd60` (pre-engineered in weather file) replaces planned `pp_day5/10` features
> - 12 new weather features now available: solar radiation, VPD, heat stress days, boll fill windows, pre-harvest and post-defoliation windows
> - M5 named: rm-Band GDD Profiler
> - M6 (Survival / Shelf-Life Predictor) is the primary business deliverable
> - Dashboard must be a standalone Streamlit app — not notebook-dependent

---

# 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Critical Constants](#3-critical-constants)
4. [Modeling — Features, Models, Pipeline](#4-modeling)
5. [Dashboard — FastAPI + Streamlit](#5-dashboard)
6. [Deployment — Docker, GitHub, CI/CD](#6-deployment)
7. [Executive Summary Page Spec](#7-executive-summary-page)
8. [Data Entry Reference](#8-data-entry-reference)
9. [Glossary](#9-glossary)
10. [Hard Rules — Things to Never Do](#10-hard-rules)
11. [Session Start Checklist](#11-session-start-checklist)

---

# 1. Project Overview

Predicts cotton seed quality degradation using CT score threshold of 60%.
CT < 60 = **Degraded**. Six models answer progressively deeper questions:

- **M1** — Is this lot currently degraded? (binary)
- **M2** — What CT score does this lot have right now? (regression)
- **M3** — Which quality tier — Degraded / At Risk / High Quality? (3-class)
- **M4** — Should we reject this Stage 1 intake lot immediately? (binary screen)
- **M5** — How does heat accumulation affect CT by maturity group? (rm-band profiler)
- **M6** — *How many seasons until this lot's CT drops below 60?* **(primary deliverable)**

**Why `rm` instead of `Variety`:**
New cotton varieties release every season. A model trained on variety names cannot
generalise to varieties it has never seen. `rm` (Relative Maturity, e.g. 110-day) is
always available for any variety — including ones released next season.

---

# 2. Directory Structure

```
Degradation Prediction Intelligent System/
├── CLAUDE.md                                ← This file
├── Data/
│   └── raw/
│       ├── vw_cotton_lineage_and_quality_june_fg_all_cols.csv  ← PRIMARY (70,570 rows)
│       ├── vw_cotton_qlty_rslt_all_cols.csv                    ← SUPPLEMENT (88,290 rows)
│       └── cotton_weather_features.xlsx                         ← WEATHER (1,716 rows, 2019–2025)
│                                                                   Pre-engineered. One row per variety+state+pa_year.
│                                                                   Replaces ALL previous weather files.
│       # ALL other files dropped entirely:
│       #   2026_cotton_with_weather.csv  — replaced by cotton_weather_features.xlsx
│       #   cottons3_2025.csv             — no longer needed
│       #   weathers3_2025.csv            — no longer needed
│       #   cottonCSs3_2025.csv, cottonCS2s3_2025.csv, weatherCSs3_2025.csv — features absent
├── models/
│   ├── m1_binary_classifier.pkl
│   ├── m2_ct_regressor.pkl
│   ├── m3_3class_classifier.pkl
│   ├── m4_stage1_screen.pkl
│   ├── m5_rmband_profiler.pkl
│   ├── m6_cox_ph.pkl
│   ├── m6_aft_weibull.pkl
│   └── model_metadata.json
├── notebooks/
│   └── Cotton_Seed_Degradation_Prediction.ipynb
├── app/
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/   (eda, predictions, survival, inventory, health)
│   │   ├── services/  (data_loader, model_service, eda_service, survival_service)
│   │   ├── schemas/
│   │   └── config.py
│   └── dashboard/
│       ├── app.py
│       ├── pages/     (01-09 pages — see Section 5)
│       ├── components/(charts, cards, filters, theme)
│       └── api_client.py
├── requirements.txt
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── .env.example
├── .gitignore
└── run.sh / run.bat
```

---

# 3. Critical Constants

```python
CT_THRESHOLD           = 60          # Degradation boundary
RANDOM_STATE           = 42
CURRENT_YEAR           = 2026
CT_INITIAL_RETEST_DAYS = 14          # CT_Initial validity window

# Temporal split — NEVER randomise
TRAIN_SEASONS = [2017, 2019, 2020, 2021]   # ~49,100 CT-tested lots
VAL_SEASONS   = [2022]                      # ~3,432  — stress-test year
TEST_SEASONS  = [2023, 2024]               # ~4,532  — out-of-sample
HOLDOUT       = [2025]                      # ~944    — never used in tuning
```

---

# 4. Modeling

## 4.1 Feature Sets

### Core Features — lineage (ACTIVE)
```python
CORE_FEATURES = [
    # WG_Current REMOVED — not available at prediction time (test not yet run)
    # WG_Initial_7Day_Cnt REMOVED — same reason
    'CT_Initial',           # 37.2% — validity rule applied first
    'Moisture',             # 87.3%
    'Mechanical_Damage',    # 85.2%
    'Actual_Seed_Per_LB',   # 86.0%
    'rm',                   # Relative Maturity — REPLACES Variety
    'Stage',                # 93.6% — stratify on this
    'season_age',           # CURRENT_YEAR - SEASON_YR
    'RR_Lateral_Strip_PCT',
    'Cry1Ac_Bollgard_Strip_Test',
    'Cry2Ab_Bollgard_Strip_Test',
]
CAT_FEATURES = ['Origin_Region', 'Grower_Region']
```

### Weather Features — cotton_weather_features.xlsx (ACTIVE — pre-engineered)
```python
# All features below come pre-aggregated. Join key: variety + state + pa_year.
# No daily-row processing required — file already has one row per field-season.

WEATHER_FEATURES = [
    # ── Core thermal / moisture ──────────────────────────────────────────────
    'cumulated_dd60',            # seasonal cumulative DD60 — PRIMARY thermal signal
    'avg_soil_moisture',         # mean seasonal soil moisture — drought proxy
    'dd_60',                     # total DD60 for season (same as cumulated at season end)
    'irrigation_type',           # 11 system types — water stress proxy
    'maczone',                   # management zone 1-8

    # ── New: full-season environmental signals ───────────────────────────────
    'total_precipitation',       # seasonal total precip (89 nulls — use with care)
    'total_solar_radiation',     # seasonal solar radiation — full coverage
    'season_heat_stress_days',   # days above heat stress threshold — full coverage
    'season_avg_vpd',            # vapour pressure deficit mean (~25% null — supplementary)

    # ── New: post-planting window (14 days) ──────────────────────────────────
    'pp14_cum_dd60',             # replaces pp_day5/10 — cumulative DD60 days 0-14
    'pp14_total_precip',         # precipitation in first 14 days post-planting

    # ── New: boll fill period ────────────────────────────────────────────────
    'boll_fill_cum_dd60',        # heat accumulation during boll fill
    'boll_fill_total_precip',    # precipitation during boll fill
    'boll_fill_heat_stress_days',# heat stress days during boll fill

    # ── New: pre-harvest 30 days ─────────────────────────────────────────────
    'pre_harvest30_cum_dd60',    # heat accumulation in 30 days before harvest
    'pre_harvest30_total_precip',# precipitation in 30 days before harvest

    # ── New: post-defoliation period ─────────────────────────────────────────
    'post_defol_cumulated_dd60', # heat accumulation post-defoliation (32 nulls)
    'post_defol_avg_soilmoisture',# soil moisture post-defoliation (32 nulls)
    'post_defol_total_precip',   # precipitation post-defoliation (121 nulls)
]
# NOTE: season_avg_vpd (~25% null) and post_defol features (~7% null) are supplementary.
# total_precipitation (89 nulls) — impute median or drop depending on coverage after join.
```

### Engineered Features (computed in pipeline Step 4)
```python
ENGINEERED_FEATURES = [
    'season_age',                # CURRENT_YEAR - SEASON_YR
    'irrigation_is_dryland',     # 1 = gravity/surface (FurrowSurge, Flood), 0 = pressurised
    'ct_distance_to_threshold',  # CT_Current - 60 (survival signal; negative = already degraded)
    'cumulated_soil_moisture',   # full-season soil moisture sum (if not already in weather file)
    'rm_band',                   # categorical from rm value
    # NOTE: pp14_cum_dd60 and pp14_total_precip come PRE-ENGINEERED from cotton_weather_features.xlsx
    # pp_day5/10 features are DROPPED — replaced by pp14_cum_dd60
    # WG-derived features (vigor gap, WG_current proxy) DROPPED — WG not available at prediction time
]
```
```

### rm Band Definitions (used in M5)
```python
RM_BANDS = {
    'Ultra-Early': (0,   100),
    'Early':       (100, 110),
    'Mid-Early':   (110, 120),
    'Mid-Full':    (120, 130),
    'Full':        (130, 999),
}
def get_rm_band(rm):
    for name, (lo, hi) in RM_BANDS.items():
        if lo <= rm < hi: return name
    return 'Unknown'
```

## 4.2 The 6 Models

| ID | Name | Type | Algorithm | Min Metric |
|----|------|------|-----------|------------|
| M1 | Degradation Classifier | Binary | XGBoost + class_weight | AUC ≥ 0.80 |
| M2 | CT Score Regressor | Regression | LightGBM | RMSE < 10 pts |
| M3 | 3-Class Classifier | 3-class + SHAP | XGBoost softmax | Macro-F1 ≥ 0.72 |
| M4 | Stage 1 Screen | Binary (Stage 1 only) | XGBoost | AUC ≥ 0.75 |
| M5 | rm-Band GDD Profiler | Per-rm-band regression | LightGBM | RMSE < 12 pts |
| **M6** | **Survival / Shelf-Life** | Time-to-event | Cox PH + Weibull AFT | C-index ≥ 0.70 |

```python
# M1, M2, M3 share the same feature scope
# WG_Current and WG_Initial_7Day_Cnt REMOVED — not available at prediction time
M1_FEATURES = CORE_FEATURES + WEATHER_FEATURES + ENGINEERED_FEATURES

# M4 — intake only, no joins required
# WG not available at intake — CT_Initial used as early quality signal instead
M4_FEATURES = ['Moisture', 'Mechanical_Damage', 'Actual_Seed_Per_LB', 'rm', 'CT_Initial']
M4_CAT      = ['Origin_Region']
M4_THRESHOLD = 0.40   # lower than default — maximise recall on degraded

# M5 — grouped by rm band, not by Variety
M5_FEATURES = ['rm', 'cumulated_dd60', 'avg_soil_moisture',
               'irrigation_is_dryland', 'season_age']
M5_CAT      = ['Origin_Region', 'irrigation_type']
M5_MIN_RECORDS_PER_BAND = 30

# M6 — survival
SURVIVAL_FEATURES = [
    'season_age',                # duration
    # event derived from CT_Current < 60
    # WG_Current REMOVED — not available at prediction time
    'CT_Initial', 'Moisture', 'Mechanical_Damage',
    'rm', 'Stage', 'cumulated_dd60', 'avg_soil_moisture',
    'irrigation_type', 'ct_distance_to_threshold',
    'season_heat_stress_days', 'pp14_cum_dd60',   # new weather features
]
# Encode Origin_Region as dummies drop_first=True
```

## 4.3 Data Quality Rules — MANDATORY

```python
def apply_data_quality_rules(df):
    # 1. Remove CT > WG anomalies (biologically impossible)
    mask = (df['CT_Current'].notna() & df['WG_Current'].notna()
            & (df['CT_Current'] > df['WG_Current']))
    df   = df[~mask].copy()

    # 2. Drop corrupted column
    df = df.drop(columns=['Cleanout_PCT'], errors='ignore')

    # 3. Cap FFA at 5.0
    if 'FFA' in df.columns:
        df['FFA'] = df['FFA'].clip(upper=5.0)

    # 4. Filter Seed_Temperature to 50-110°F
    if 'Seed_Temperature' in df.columns:
        df['Seed_Temperature'] = df['Seed_Temperature'].where(
            df['Seed_Temperature'].between(50, 110), np.nan)

    # 5. Drop NAWF and scouting columns
    drop = ['nodes_above_white_flower_1', 'nodes_above_white_flower_2',
            'nodes_above_white_flower_3', 'defoliation_1', 'seeding_rate']
    df   = df.drop(columns=[c for c in drop if c in df.columns])

    # 6. CT_Initial validity rule — null if retest within 14 days
    if all(c in df.columns for c in ['CT_Initial_DT','CT_Current_DT','CT_Initial']):
        df['CT_Initial_DT'] = pd.to_datetime(df['CT_Initial_DT'], errors='coerce')
        df['CT_Current_DT'] = pd.to_datetime(df['CT_Current_DT'], errors='coerce')
        gap     = (df['CT_Current_DT'] - df['CT_Initial_DT']).dt.days
        invalid = gap.notna() & gap.between(0, CT_INITIAL_RETEST_DAYS)
        df.loc[invalid, 'CT_Initial'] = np.nan

    # 7. Targets
    df['degraded_binary'] = (df['CT_Current'] < CT_THRESHOLD).astype('Int64')
    df['quality_class']   = pd.cut(df['CT_Current'],
                                    bins=[-0.01, 60, 80, 100], labels=[0,1,2])
    df['season_age']      = CURRENT_YEAR - df['SEASON_YR']
    return df
```

## 4.4 Two-Source Pipeline (5 Clean Active Steps)

```
STEP 1  Load lineage anchor. Apply data quality rules. Create targets.
        Supplement with quality_results on INSPCT_LOT_NBR if needed.

STEP 2  Load cotton_weather_features.xlsx (already aggregated — 1 row per
        variety + state + pa_year). No daily-row processing needed.
        Log row count at runtime: expect ~1,716 records, years 2019–2025.

STEP 3  Join lineage → weather features.
        Join key: variety + state + pa_year (exact match).
        Log match rate at runtime.

STEP 4  Feature engineering: compute ENGINEERED_FEATURES.
        ct_distance_to_threshold, irrigation_is_dryland, season_age, rm_band.
        pp14_cum_dd60 and all other weather windows come pre-engineered from Step 2.
        Apply CT_Initial validity rule (14-day retest rule).

STEP 5  Temporal split on SEASON_YR.
```

```python
def log_join_coverage(df, col, step):
    n   = df[col].notna().sum()
    pct = n / len(df) * 100
    tag = '⚠️ LOW' if pct < 10 else '✅'
    print(f'  {tag} JOIN [{step}]: {n:,}/{len(df):,} ({pct:.1f}%)')
```

---

# 5. Dashboard

## 5.1 Architecture

**FastAPI** serves all data and predictions. **Streamlit** consumes the API.
Both run as Docker services on the same network.

```
Streamlit (port 8501) → http calls → FastAPI (port 8000) → models + data
```

## 5.2 Design System

```python
COLORS = {
    "bg_primary":    "#0A0F0D",  "bg_card":   "#111A14",
    "accent_orange": "#F97316",  "accent_green": "#22C55E",
    "accent_amber":  "#F59E0B",  "accent_teal":  "#14B8A6",
    "accent_blue":   "#3B82F6",  "text_primary": "#F0FDF4",
    "text_secondary":"#86EFAC",  "border_subtle":"#1E3A2A",
    "degraded":      "#EF4444",  "at_risk":      "#F59E0B",
    "high_quality":  "#22C55E",
}
FONTS = {
    "display": "'Space Mono', monospace",       # numbers
    "body":    "'DM Sans', sans-serif",          # text
    "mono":    "'JetBrains Mono', monospace",    # data labels
}
```

All Plotly charts apply `PLOTLY_THEME` via `apply_theme()` — dark background,
green gridlines, JetBrains Mono hover labels. Never use `st.pyplot`.

## 5.3 FastAPI Endpoints

```
/health            GET   ping + readiness check

/eda
  /overview              dataset shapes, null rates, coverage
  /ct-distribution       histogram + class breakdown
  /seasonal-trends       mean CT + degradation % by SEASON_YR
  /regional              stats by Origin_Region
  /stage-analysis        quality gradient by Stage
  /rm-rankings           top-N by rm band (replaces variety-rankings)
  /wg-ct-gap             vigor gap stats + false-pass count
  /physical-quality      Moisture, Mechanical_Damage, FFA stats
  /correlation-matrix    feature correlation data
  /weather-summary       2026_weather summary only
  /irrigation-breakdown  from 2026_weather

/predict
  /single                  POST  M1+M2+M3 combined response
  /batch                   POST  CSV upload → predictions CSV
  /m1/feature-importance   GET   SHAP values
  /m2/feature-importance   GET   SHAP values
  /model-metrics           GET   AUC, RMSE, F1 from test set

/survival
  /km-overall              GET   KM curve data (all lots)
  /km-by-region            GET   stratified by Origin_Region
  /km-by-stage             GET   stratified by Stage
  /km-by-rm-band           GET   stratified by rm band
  /predict-lot             POST  AFT shelf-life prediction per lot
  /predict-batch           POST  batch shelf-life predictions
  /cox-hazard-ratios       GET   Cox PH coefficients
  /logrank-test            GET   p-values by group
  /median-shelflife        GET   portfolio percentiles

/inventory
  /risk-summary            count/value by risk tier
  /high-risk-lots          degradation prob > 0.7
  /false-pass-lots         WG pass but CT fail
  /expiring-soon           predicted shelf-life < 6 months
  /regional-risk-map       risk per region
  /rm-risk-table           per-rm-band risk + shelf-life
  /stage1-reject-screen    M4 predictions
  /business-questions      master KPI endpoint
```

## 5.4 Pydantic Schemas

```python
class LotInput(BaseModel):
    rm:                 Optional[float] = None  # PRIMARY variety proxy
    origin_region:      str
    season_yr:          int
    stage:              Optional[int]   = None
    wg_current:         Optional[float] = None
    ct_initial:         Optional[float] = None  # validated server-side
    moisture:           Optional[float] = None
    mechanical_damage:  Optional[float] = None
    actual_seed_per_lb: Optional[float] = None
    cumulated_dd60:     Optional[float] = None
    avg_soil_moisture:  Optional[float] = None
    irrigation_type:    Optional[str]   = None
    maczone:            Optional[str]   = None

class SinglePredictionResponse(BaseModel):
    lot_id:                       Optional[str]
    degradation_prob:             float
    predicted_ct_score:           float
    quality_class:                str
    risk_level:                   str
    predicted_shelf_life_seasons: Optional[float]
    rm_band:                      Optional[str]
    recommendation:               str
    shap_top_features:            Optional[List[dict]]
```

## 5.5 Dashboard Pages (9 total)

| # | Page | Business Question |
|---|------|-------------------|
| 01 | 🌾 Portfolio Overview | What is portfolio-wide quality status? Which regions at risk? |
| 02 | 🔬 Quality Deep Dive | What does CT distribution look like? How many false-pass? |
| 03 | 🌡️ Field Intelligence | How does irrigation/heat affect CT? (2026_weather only) |
| 04 | ⏱️ Shelf-Life Predictor | **How many seasons does this lot have left? (M6 hero page)** |
| 05 | 🏭 Pipeline Stage Analysis | At which stage is quality lost most? |
| 06 | 🌿 rm-Band Profiler | Which maturity bands perform best? GDD curves per band. |
| 07 | 📊 Full EDA | Complete data exploration — every distribution, every chart |
| 08 | 🔮 Batch Predictions | Score entire inventory at once |
| 09 | 📋 Executive Summary | Plain-English overview of everything (see Section 7) |

## 5.6 Page 04 — Shelf-Life Predictor (Hero Page Spec)

Sidebar inputs use `rm` slider, not a Variety dropdown:
```python
rm_val = st.slider("Relative Maturity (rm)",
                    min_value=80.0, max_value=145.0, value=110.0, step=1.0)
st.caption("Bands: Ultra-Early <100 | Early 100-110 | Mid-Early 110-120 | "
           "Mid-Full 120-130 | Full >130")
```

Layout:
```
┌─────────────────────┬──────────────────────────────────┐
│ LOT INPUTS          │ SURVIVAL CURVE OUTPUT            │
│ rm slider           │ Animated KM curve for this lot   │
│ Region, Stage,      │ Median survival highlighted      │
│ WG, Moisture,       │ ┌──────────────────────────────┐ │
│ Mech_Damage,        │ │ 📅 Predicted Shelf-Life     │ │
│ Irrigation, DD60,   │ │ ████  2.4 SEASONS           │ │
│ Soil Moisture       │ │ until CT drops below 60%    │ │
│ [PREDICT]           │ └──────────────────────────────┘ │
├─────────────────────┴──────────────────────────────────┤
│ KM CURVES BY REGION (multi-line, log-rank p-value)     │
├────────────────────────────────────────────────────────┤
│ KM CURVES BY STAGE                                     │
├──────────────────────┬─────────────────────────────────┤
│ COX PH FOREST PLOT   │ AFT SHELF-LIFE DISTRIBUTION     │
└──────────────────────┴─────────────────────────────────┘

ACTION TABLE: "Sell-by Priority" — lots sorted by shortest shelf-life
```

## 5.7 Page 06 — rm-Band Profiler

Sidebar uses radio buttons for rm bands, not a Variety search:
```python
band = st.radio("Select rm Band", [
    "Ultra-Early (rm < 100)",
    "Early (100-110)",
    "Mid-Early (110-120)",
    "Mid-Full (120-130)",
    "Full (rm > 130)",
])
```

Each band shows: scorecard (CT mean, Degraded %, Shelf-life), GDD curve scatter
with M5 trend line, radar chart for cross-band comparison.

---

# 6. Deployment

## 6.1 Repository Setup

```bash
git init && git branch -M main
git lfs install && git lfs track "models/*.pkl"
```

`.gitignore` essentials:
```
__pycache__/  .venv/
Data/raw/*.csv          # raw data not committed
models/*.pkl            # use Git LFS
.env                    # production secrets never committed
.ipynb_checkpoints/
logs/  *.log
```

## 6.2 Docker

**Dockerfile.api**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/api/ ./api/
COPY app/config.py ./
RUN mkdir -p /app/data/raw /app/models
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**Dockerfile.dashboard**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/dashboard/ ./dashboard/
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--theme.base=dark", \
     "--theme.primaryColor=#22C55E", \
     "--theme.backgroundColor=#0A0F0D", \
     "--theme.secondaryBackgroundColor=#111A14", \
     "--theme.textColor=#F0FDF4"]
```

**docker-compose.yml**
```yaml
version: '3.9'
services:
  api:
    build: { context: ., dockerfile: Dockerfile.api }
    container_name: cotton_seed_api
    ports: ["8000:8000"]
    environment:
      - DATA_PATH=/app/data/raw
      - MODEL_PATH=/app/models
      - CT_THRESHOLD=60
      - CURRENT_YEAR=2026
      - CT_INITIAL_RETEST_DAYS=14
    volumes:
      - ./Data/raw:/app/data/raw:ro
      - ./models:/app/models:ro
    restart: unless-stopped
    networks: [cotton_net]

  dashboard:
    build: { context: ., dockerfile: Dockerfile.dashboard }
    container_name: cotton_seed_dashboard
    ports: ["8501:8501"]
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      api: { condition: service_healthy }
    restart: unless-stopped
    networks: [cotton_net]

networks:
  cotton_net: { driver: bridge }
```

Run:
```bash
docker-compose up -d --build
docker-compose logs -f
curl http://localhost:8000/health
open http://localhost:8501
```

## 6.3 GitHub Actions

`.github/workflows/ci.yml` — lint + test on every push/PR.
`.github/workflows/deploy.yml` — build images, push to GHCR, SSH deploy on merge to main.

Secrets required: `SERVER_HOST`, `SERVER_USER`, `SERVER_SSH_KEY`.

## 6.4 Production

Nginx reverse proxy with WebSocket upgrade for Streamlit.
SSL via Let's Encrypt / Certbot. `.env` injected via secrets manager,
never committed. Data + models mounted read-only.

## 6.5 Pre-Deployment Checklist

- [ ] `pytest tests/` passes
- [ ] `ruff check app/` clean
- [ ] `.env` not committed (verify in `.gitignore`)
- [ ] `docker-compose build` succeeds
- [ ] `curl localhost:8000/health` returns `"status": "healthy"`
- [ ] Dashboard loads at `localhost:8501`
- [ ] Data + model volumes mounted read-only
- [ ] `m5_rmband_profiler.pkl` present (not old `m5_gdd_profiler.pkl`)
- [ ] `model_metadata.json` has `rm_bands` key, no Variety in feature lists
- [ ] No s3 file paths anywhere in `config.py` or `.env`

## 6.6 Common Problems

| Problem | Cause | Fix |
|---------|-------|-----|
| `rm` column all null | rm not recorded at intake | Pull from seed bag label at lot creation |
| Weather join 0 rows | GUID mismatch on pa_feature_id | Use composite: variety + state + pa_year |
| Variety in results | Old artifact loaded | Reload — confirm `m5_rmband_profiler.pkl` |
| CT_Initial not nulled | Validity rule not running | Confirm `apply_data_quality_rules()` called before `model.fit()` |
| Streamlit WebSocket disconnect | Nginx not upgraded | Add `Upgrade` headers to nginx config |
| Port 8000 in use | Other process | `lsof -ti:8000 \| xargs kill -9` |

---

# 7. Executive Summary Page

This is **Page 09** of the dashboard — stakeholder-friendly overview.

## Section A — What Was Built (8 page cards + summary)

Render a 2×4 grid of dark cards. Each card: icon, page name, what it does,
which business questions it answers. Page 04 (Shelf-Life Predictor) marked
PRIMARY with blue glow border.

## Section B — Headline Findings (6 numbered cards)

1. **18.7% of lots are currently degraded** — not the crisis it may seem;
   the larger concern is the 61.4% At Risk band.
2. **Arizona is highest-risk region at 29.4%** despite producing the most seed.
3. **8,516 lots are False-Pass** — pass warm germination, fail cool test.
4. **Quality improves dramatically through pipeline** — 70% degraded at intake
   to 5.8% at final product.
5. **2017 and 2019 best seasons; 2018 and 2022 worst** — driven by weather, not process.
6. **Median lot has 2-3 growing seasons of viable shelf-life remaining.**

## Section C — Models in Plain English

Each model gets: what it does (plain), how it works (plain), what it outputs,
what metric tells you it's working. M5 described as "groups by maturity rating
not variety name — works for new varieties too." M6 highlighted as primary.

## Section D — Challenges Faced (honest)

1. Cleanout_PCT corrupted (range -92,618 to 3,512) — column dropped.
2. 915 CT > WG anomalies — data capture failures, excluded from training.
3. New varieties every season — solved by using rm instead of variety names.
4. 2026 field operations data not yet available — pipeline runs cleanly on
   what is available; no waiting required.
5. Sparse FFA coverage (7.3%) — used as supplementary signal only.
6. Class imbalance — handled with `scale_pos_weight` in XGBoost.

## Section E — Recommendations

**Immediate:** Make rm mandatory at intake. Add range validation to data entry.
Investigate AZ degradation rate. Implement Stage 1 early CT testing.

**Short-term:** Deploy M6 as primary inventory tool. Review high-risk variety
contracts. When 2026 field operations data arrives, run the field enrichment step.

**Medium-term:** Collect repeat CT measurements over time. Build variety-specific
CT thresholds. Integrate IoT storage sensors.

## Section F — Glossary (search-enabled component)

47 terms defined in plain English. See Section 9.

---

# 8. Data Entry Reference

## 8.1 Lineage Dataset

| Field | Valid Range | Common Errors | Action |
|-------|-------------|---------------|--------|
| `CT_Current` | 0.0 – 100.0 | Above 100, negatives | Exclude; flag for re-entry |
| `WG_Current` | 0.0 – 100.0 | CT > WG records | Exclude both; flag failure |
| `CT_Initial` | 0.0 – 100.0 | **Invalid if retest ≤14 days later** | Null before modeling |
| `CT_Initial_DT` | Valid date | Missing | **Required** for validity rule |
| `CT_Current_DT` | Valid date, after CT_Initial_DT | Missing | **Required** for validity rule |
| `Moisture` | 3.0 – 16.0% | Above 20% suspect | Cap at 20% after review |
| `Mechanical_Damage` | 0.0 – 50.0% | Above 60% | Cap at 60% |
| `FFA` | 0.0 – 5.0 | Max of 70 found — likely 7.0 mistyped | **Cap at 5.0** |
| `Actual_Seed_Per_LB` | 3,000 – 10,000 | <1,000 or >15,000 | Flag for scale calibration |
| `Cleanout_PCT` | **EXCLUDED** | -92,618 to 3,512 | Drop entirely |
| `rm` | 80 – 145 | Missing (~14.7% null) | **Record from seed bag at intake** |
| `Stage` | 1, 2, 3, 4, 5 | Missing | Default to 1 if not yet processed |
| `SEASON_YR` | 2000 – current+1 | Future, pre-2000 | Reject at entry |
| `Origin_Region` | AZ, TX, CA, MS, AR, NM | Misspellings, full names | Standardise to 2-letter UPPER |
| `RR_Lateral_Strip_PCT` | 0.0 – 100.0 | <90% | **Escalate immediately** |
| `Cry1Ac_Bollgard_Strip_Test` | 0.0 – 100.0 | <90% | **Escalate immediately** |

## 8.2 2026 Weather Dataset

| Field | Valid Range | Notes |
|-------|-------------|-------|
| `pa_feature_id` | GUID string | Primary key |
| `cumulated_dd60` | 0 – 6,000 | Monotonically increasing per season |
| `dd_60` | 0 – 40 | Daily increment |
| `avg_soil_moisture` | 0.0 – 1.0 | Outside range = sensor error |
| `irrigation_type` | Approved list | Enforce dropdown |
| `maczone` | 1 – 8 | Match field boundary records |
| `planting_date` | Apr–Jun of season | Must precede defoliation_date |
| `defoliation_date` | Aug–Oct | Must precede harvest_date |
| `harvest_date` | Sep–Nov | Last in sequence |
| `variety` + `state` + `pa_year` | Composite key | Must match lineage exactly |

**Approved Irrigation Types:**
```
FurrowSurge | FurrowConventional | CenterPivotNBC | CenterPivotNAC
CenterPivotLEPA | CenterPivotLPHE | Flood | SubsurfaceDrip
GatedPipePhaucet | Dryland | Furrow | Sprinkler
```

## 8.3 Fields No Longer Used

| Field | Reason |
|-------|--------|
| `Variety` (as model input) | New varieties every season — use rm |
| `nodes_above_white_flower_*` | Removed per stakeholder |
| `defoliation_1` (s3) | s3 file not in 2026 |
| All cottons3 fields | s3 file not in 2026 |
| All cottonCSs3 fields | s3 file not in 2026 |
| All weatherCSs3 fields | s3 file not in 2026 |
| `seeding_rate` | 100% null — not collected |
| `Cleanout_PCT` | Data corruption |

---

# 9. Glossary

**A**
- **AFT Model**: Accelerated Failure Time — predicts actual days/seasons until degradation event for a specific lot.
- **AUC-ROC**: Number 0-1 measuring how well a classifier separates two groups; 1.0 perfect, 0.5 random.

**C**
- **C-Index**: Survival model accuracy metric; 0.7+ acceptable.
- **Censored**: A lot that has not yet experienced degradation at last observation.
- **CT (Cool Test)**: Lab test of seed vigor at ~65°F; reveals stress tolerance.
- **Cox PH Model**: Cox Proportional Hazards — outputs hazard ratios showing which factors accelerate degradation.

**D**
- **DD60**: Degree Days above 60°F — daily heat accumulation.
- **Degraded**: A lot with CT < 60.
- **Duration (Survival)**: Time axis = season_age = CURRENT_YEAR − SEASON_YR.

**E**
- **Event (Survival)**: CT_Current dropping below 60.
- **EDA**: Exploratory Data Analysis.

**F**
- **False-Pass Lot**: WG ≥ 80 but CT < 60 — fails under field stress despite passing lab test.
- **FFA**: Free Fatty Acids — early rancidity signal.
- **FurrowSurge**: Most common irrigation system (~34%).

**G**
- **GDD**: Growing Degree Days; same concept as DD60.

**H**
- **Hazard Ratio**: Cox PH output; >1 increases risk, <1 protective.
- **High Quality**: CT ≥ 80.

**I**
- **Imputation**: Filling missing values with statistical estimate (median).
- **Irrigation Type**: Water delivery system; influences water stress.

**K**
- **Kaplan-Meier Curve**: Survival probability over time.

**L**
- **LightGBM**: Fast gradient boosting; used for M2, M5.
- **Lineage Dataset**: Primary quality database (70,570 records).
- **Log-Rank Test**: Compares survival curves for statistical difference.
- **Lot**: Single trackable batch of seed.

**M**
- **Maczone**: Management zone (1-8); geographic stratifier.
- **Mechanical Damage**: % seeds physically damaged.

**N**
- **Null**: Missing value in a dataset.

**O**
- **Optuna**: Hyperparameter tuning framework.
- **Origin_Region**: US state where cotton grown.

**R**
- **rm (Relative Maturity)**: Agronomic rating (e.g. 110-day). **PRIMARY variety proxy in models** — stable across seasons, available for any variety including new ones.
- **rm Band**: Grouping by rm range — Ultra-Early (<100), Early (100-110), Mid-Early (110-120), Mid-Full (120-130), Full (>130).
- **rm-Band GDD Profiler (M5)**: Predicts CT score from heat accumulation grouped by rm band. Replaces variety-specific profiling.
- **RMSE**: Root Mean Squared Error — average prediction error.

**S**
- **s3 Files**: 2025-only datasets (cottons3, cottonCSs3, cottonCS2s3, weatherCSs3, weathers3) — **not used in this project**.
- **Season Age**: Years since harvest = CURRENT_YEAR − SEASON_YR.
- **SHAP**: Shapley values explaining individual model predictions.
- **Stage 1-5**: Pipeline position; Stage 1 = harvest intake (70% degraded), Stage 5 = final product (5.8% degraded).
- **Survival Analysis**: Statistical method for time-to-event modeling.

**T**
- **Temporal Split**: Time-ordered train/test split to prevent leakage.
- **Threshold (CT=60)**: Quality boundary used throughout.

**V**
- **Variety**: Cultivar name. **NOT used in models** — new varieties every season; use rm instead.
- **VPD**: Vapour Pressure Deficit — atmospheric water demand. (Not currently used — was in s3 files.)

**W**
- **Weibull AFT**: Survival model variant predicting direct shelf-life days per lot.
- **WG (Warm Germination)**: Lab test at ~85°F under ideal conditions; basic viability.

**X**
- **XGBoost**: Gradient boosting algorithm; used for M1, M3, M4.

**2**
- **2026_cotton_with_weather**: Primary weather dataset — daily field-level records for 2025-2026 season. **One of two active data sources.**

---

# 10. Hard Rules

```python
FORBIDDEN_FEATURES = [
    'Variety',                    # use rm
    'WG_Current',                 # NOT available at prediction time — NEVER use as feature
    'WG_Initial_7Day_Cnt',        # NOT available at prediction time — NEVER use as feature
    'nodes_above_white_flower_1',
    'nodes_above_white_flower_2',
    'nodes_above_white_flower_3',
    'defoliation_1',
]

# All weather files replaced by cotton_weather_features.xlsx
FORBIDDEN_FILES = [
    '2026_cotton_with_weather.csv',   # replaced by cotton_weather_features.xlsx
    'cottons3_2025.csv',              # no longer needed
    'weathers3_2025.csv',             # no longer needed
    'cottonCSs3_2025.csv',
    'cottonCS2s3_2025.csv',
    'weatherCSs3_2025.csv',
]

ACTIVE_DATA_FILES = [
    'vw_cotton_lineage_and_quality_june_fg_all_cols.csv',
    'vw_cotton_qlty_rslt_all_cols.csv',
    'cotton_weather_features.xlsx',
]

def assert_clean_features(feature_list, model_name):
    for f in FORBIDDEN_FEATURES:
        assert f not in feature_list, f'FORBIDDEN: {f} in {model_name}'
```

**Never:**
- Use `Variety` as a model feature (keep in dataframe for reference only).
- Use `WG_Current` or `WG_Initial_7Day_Cnt` as model features — these lab test results are not available until all testing is complete, making them unusable for real-time prediction.
- Use `NAWF` features anywhere.
- Use `CT_Initial` without first running the 14-day validity rule.
- Load any file other than the 3 active data files (lineage, quality_results, cotton_weather_features.xlsx).
- Attempt to process raw daily weather rows — `cotton_weather_features.xlsx` is already aggregated.
- Randomise train/test splits — always temporal on `SEASON_YR`.
- Train across `Stage` values without Stage as stratification variable.
- Impute `CT_Current` — it is the target; exclude rows without it.
- Treat M6 as optional — it is the primary business deliverable.
- Use `st.pyplot` — all charts use Plotly + `apply_theme()`.
- Hardcode data paths — use `settings.BASE_DATA_PATH`.
- Cache prediction endpoints with `st.cache_data`.
- Add a Variety dropdown to any sidebar — use rm slider/radio instead.
- Build or demo predictions inside a notebook — the Streamlit app is the deliverable.

---

# 11. Session Start Checklist

When starting a new Claude Code session:

1. Read this CLAUDE.md fully before taking any action.
2. Confirm `Data/raw/` contains exactly **3 files**: lineage CSV, quality_results CSV, cotton_weather_features.xlsx.
3. Confirm NO other weather or s3 files are referenced anywhere in open code.
4. Confirm `models/` directory exists.
5. Run `assert_clean_features()` before every `model.fit()` — verify WG_Current and WG_Initial_7Day_Cnt are absent.
6. Build M6 always — even when iterating on other models.
7. Ask the user which task before opening any file.
8. Never modify raw data files — copy to a working variable.
9. All predictions and dashboards must live in the Streamlit app — not in notebooks.

---

*Version 5 — May 2026 | Single source of truth for modeling, dashboard, and deployment*
