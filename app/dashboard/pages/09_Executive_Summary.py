"""Page 09 — Executive Summary (CLAUDE.md §7 spec)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from components.theme import inject_css, COLORS, FONTS
from api_client import client

st.set_page_config(page_title="Executive Summary", layout="wide")
inject_css()

st.markdown('<div class="section-header">09 — Executive Summary</div>', unsafe_allow_html=True)
st.markdown("Plain-English overview for stakeholders.")

# ── Section A: What Was Built ──────────────────────────────────────
st.markdown("### What Was Built")
pages = [
    ("01", "Portfolio Overview", "Portfolio-wide quality status; which regions are at risk?",
     COLORS["accent_green"]),
    ("02", "Quality Deep Dive", "CT distribution, false-pass analysis, physical quality metrics.",
     COLORS["accent_teal"]),
    ("03", "Field Intelligence", "How does irrigation and heat accumulation affect CT?",
     COLORS["accent_blue"]),
    ("04", "Shelf-Life Predictor", "How many seasons until this lot's CT drops below 60%?  [PRIMARY]",
     COLORS["accent_blue"]),
    ("05", "Pipeline Stage Analysis", "At which stage is quality gained or lost most?",
     COLORS["accent_amber"]),
    ("06", "rm-Band Profiler", "Which Relative Maturity bands perform best? GDD curves.",
     COLORS["accent_teal"]),
    ("07", "Full EDA", "Complete data exploration — every distribution, every chart.",
     COLORS["text_secondary"]),
    ("08", "Batch Predictions", "Score entire inventory at once; download predictions CSV.",
     COLORS["accent_orange"]),
]

cols = st.columns(4)
for i, (num, name, desc, color) in enumerate(pages):
    border = f"2px solid {COLORS['accent_blue']}" if num == "04" else f"1px solid {COLORS['border_subtle']}"
    glow   = f"box-shadow: 0 0 18px {COLORS['accent_blue']}30;" if num == "04" else ""
    with cols[i % 4]:
        st.markdown(f"""
        <div style="background:{COLORS['bg_card']};border:{border};border-radius:10px;
                    padding:16px;margin-bottom:12px;{glow}">
            <div style="font-family:{FONTS['display']};color:{color};font-size:0.9rem">
                {num} — {name}
            </div>
            <div style="color:{COLORS['text_secondary']};font-size:0.8rem;margin-top:6px">
                {desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Section B: Headline Findings ───────────────────────────────────
st.markdown("---")
st.markdown("### Headline Findings")

try:
    bq = client.inventory_business_questions()
    ov = client.eda_overview()
    gap = client.eda_wg_ct_gap()
    stage_data = client.eda_stage_analysis()
except Exception:
    bq = ov = gap = {}
    stage_data = []

findings = [
    (f"{ov.get('degraded_pct', 18.7):.1f}% of lots are currently degraded",
     "Not the crisis it may seem — the larger concern is the At Risk band at ~61%."),
    (f"Arizona is highest-risk region at ~29.4%",
     "Despite producing the most seed. Requires focused investigation."),
    (f"{ov.get('false_pass_count', 8516):,} lots are False-Pass",
     "Pass warm germination, fail cool test. Undetected degradation risk in the field."),
    ("Quality improves dramatically through pipeline",
     "~70% degraded at intake (Stage 1) to <6% at final product (Stage 5)."),
    ("2017 and 2019 best seasons; 2018 and 2022 worst",
     "Driven by heat accumulation and soil moisture — not processing error."),
    ("Median lot has 2-3 growing seasons of viable shelf-life remaining",
     "M6 (Weibull AFT) provides per-lot precision — critical for sell-by scheduling."),
]

for i, (headline, detail) in enumerate(findings, 1):
    st.markdown(f"""
    <div style="background:{COLORS['bg_card']};border-left:4px solid {COLORS['accent_green']};
                padding:14px 18px;margin-bottom:8px;border-radius:0 8px 8px 0">
        <div style="font-family:{FONTS['display']};font-size:1rem;color:{COLORS['text_primary']}">
            {i}. {headline}
        </div>
        <div style="color:{COLORS['text_secondary']};font-size:0.85rem;margin-top:4px">{detail}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Section C: Models in Plain English ────────────────────────────
st.markdown("---")
st.markdown("### The Six Models — Plain English")

models_info = [
    ("M1 — Degradation Classifier",
     "Looks at seed quality measurements and weather data and answers: is this lot degraded right now?",
     "Probability 0-1 that CT score is below 60.", "AUC ≥ 0.80 — correctly ranks lots by risk."),
    ("M2 — CT Score Regressor",
     "Predicts the actual CT number rather than just pass/fail.",
     "Predicted CT score (0-100).", "RMSE < 10 points — within one lab-test error."),
    ("M3 — 3-Class Classifier",
     "Puts each lot in one of three buckets: Degraded, At Risk, or High Quality.",
     "One of three quality tiers.", "Macro-F1 ≥ 0.72 — good across all three classes."),
    ("M4 — Stage 1 Screen",
     "Uses only intake measurements (no lab required) to flag likely rejects at harvest intake.",
     "Reject / Accept recommendation for Stage 1 lots.", "AUC ≥ 0.75 — useful early warning."),
    ("M5 — rm-Band GDD Profiler",
     "Groups cotton lots by Relative Maturity instead of variety name. "
     "Works for new varieties never seen before.",
     "CT score predicted from heat accumulation per maturity group.",
     "RMSE < 12 points per rm band."),
    ("M6 — Shelf-Life Predictor [PRIMARY]",
     "The main business deliverable. Asks: how many growing seasons before this specific lot "
     "becomes too degraded to sell? Uses survival analysis — the same math hospitals use to "
     "predict patient outcomes.",
     "Median seasons-until-degradation, with 95% confidence interval.",
     "C-index ≥ 0.70 — correctly ranks lots by shelf-life remaining."),
]

for name, what, output, metric in models_info:
    is_primary = "M6" in name
    border = f"2px solid {COLORS['accent_blue']}" if is_primary else f"1px solid {COLORS['border_subtle']}"
    with st.expander(name, expanded=is_primary):
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**What it does:** {what}")
        c2.markdown(f"**Output:** {output}")
        c3.markdown(f"**Success metric:** {metric}")

# ── Section D: Challenges ─────────────────────────────────────────
st.markdown("---")
st.markdown("### Challenges Faced")

challenges = [
    ("Cleanout_PCT corrupted", "Values ranged from -92,618 to 3,512. Column dropped entirely."),
    ("915 CT > WG anomalies", "Biologically impossible. Excluded from training as data capture failures."),
    ("New varieties every season", "Solved by replacing Variety with rm (Relative Maturity) — "
                                    "stable across seasons and available for any variety including new ones."),
    ("2026 field operations data", "Not yet available for all lots. Pipeline runs cleanly; "
                                    "weather join logs coverage at startup. No waiting required."),
    ("Sparse FFA coverage (7.3%)", "Used as supplementary signal only, not primary feature."),
    ("Class imbalance", "Handled with scale_pos_weight in XGBoost models."),
]
for title, desc in challenges:
    st.markdown(f"- **{title}:** {desc}")

# ── Section E: Recommendations ────────────────────────────────────
st.markdown("---")
st.markdown("### Recommendations")

recs = {
    "Immediate": [
        "Make rm mandatory at intake — record from seed bag label.",
        "Add range validation to data entry for CT, WG, Moisture.",
        "Investigate Arizona degradation rate — highest risk despite highest volume.",
        "Implement Stage 1 early CT testing to catch intake rejects earlier.",
    ],
    "Short-term": [
        "Deploy M6 (this page) as primary inventory management tool.",
        "Review high-risk variety contracts using regional risk map.",
        "When 2026 field operations data arrives, re-run weather enrichment step.",
    ],
    "Medium-term": [
        "Collect repeat CT measurements over time to build true time-series survival data.",
        "Build variety-specific CT thresholds for premium genetics.",
        "Integrate IoT storage sensors for real-time degradation monitoring.",
    ],
}

for horizon, items in recs.items():
    color = COLORS["degraded"] if horizon == "Immediate" else \
            COLORS["at_risk"] if horizon == "Short-term" else COLORS["accent_teal"]
    st.markdown(f"**{horizon}**")
    for item in items:
        st.markdown(f"  - {item}")

# ── Section F: Glossary ───────────────────────────────────────────
st.markdown("---")
st.markdown("### Glossary")
search = st.text_input("Search glossary", placeholder="Type a term...")

glossary = {
    "AFT Model": "Accelerated Failure Time — predicts actual days/seasons until degradation event.",
    "AUC-ROC": "Area Under Curve 0-1; 1.0=perfect, 0.5=random guess.",
    "C-Index": "Survival model accuracy metric; 0.7+ is acceptable.",
    "Censored": "A lot that has not yet experienced degradation at last observation.",
    "CT (Cool Test)": "Lab test of seed vigor at ~65°F; reveals stress tolerance.",
    "Cox PH Model": "Cox Proportional Hazards — outputs hazard ratios showing which factors accelerate degradation.",
    "DD60": "Degree Days above 60°F — daily heat accumulation.",
    "Degraded": "A lot with CT < 60.",
    "False-Pass Lot": "WG ≥ 80 but CT < 60 — fails under field stress despite passing lab test.",
    "FFA": "Free Fatty Acids — early rancidity signal.",
    "GDD": "Growing Degree Days; same as DD60.",
    "Hazard Ratio": "Cox PH output; >1 increases risk, <1 protective.",
    "Kaplan-Meier Curve": "Survival probability over time.",
    "LightGBM": "Fast gradient boosting; used for M2, M5.",
    "Lineage Dataset": "Primary quality database with 70,000+ records.",
    "Lot": "Single trackable batch of seed.",
    "rm (Relative Maturity)": "Agronomic rating (e.g. 110-day). PRIMARY variety proxy in all models.",
    "rm Band": "Grouping by rm range: Ultra-Early, Early, Mid-Early, Mid-Full, Full.",
    "RMSE": "Root Mean Squared Error — average prediction error.",
    "Season Age": "Years since harvest = 2026 − SEASON_YR.",
    "SHAP": "Shapley values explaining individual model predictions.",
    "Stage 1-5": "Pipeline position; Stage 1 = intake (highest degradation), Stage 5 = final product.",
    "Survival Analysis": "Statistical method for time-to-event modeling.",
    "Temporal Split": "Time-ordered train/test split to prevent leakage.",
    "Threshold (CT=60)": "Quality boundary — below this = Degraded.",
    "Variety": "Cultivar name. NOT used in models — use rm instead.",
    "WG (Warm Germination)": "Lab test at ~85°F under ideal conditions; basic viability.",
    "Weibull AFT": "Survival model variant predicting direct shelf-life per lot.",
    "XGBoost": "Gradient boosting algorithm; used for M1, M3, M4.",
}

filtered = {k: v for k, v in glossary.items()
            if not search or search.lower() in k.lower() or search.lower() in v.lower()}

for term, definition in filtered.items():
    st.markdown(f"**{term}** — {definition}")
