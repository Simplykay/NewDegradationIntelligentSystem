"""
Cotton Seed Quality Intelligence System — Model Training Script
CLAUDE.md v5: lineage + cotton_weather_features.xlsx only; rm replaces Variety; M6 is primary.
Run: python train_models.py
"""
import os, json, pickle, warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, f1_score,
    root_mean_squared_error,
)
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────
ROOT                   = Path(__file__).parent
DATA_PATH              = ROOT / "Data" / "raw"
MODEL_PATH             = ROOT / "models"
CT_THRESHOLD           = 60
RANDOM_STATE           = 42
CURRENT_YEAR           = 2026
CT_INITIAL_RETEST_DAYS = 14

TRAIN_SEASONS = [2017, 2019, 2020, 2021]
VAL_SEASONS   = [2022]
TEST_SEASONS  = [2023, 2024]
HOLDOUT       = [2025]

RM_BANDS = {
    "Ultra-Early": (0,   100),
    "Early":       (100, 110),
    "Mid-Early":   (110, 120),
    "Mid-Full":    (120, 130),
    "Full":        (130, 999),
}

CORE_FEATURES = [
    # WG_Current REMOVED — not available at prediction time
    # WG_Initial_7Day_Cnt REMOVED — not available at prediction time
    "CT_Initial", "Moisture", "Mechanical_Damage", "Actual_Seed_Per_LB", "rm",
    "Stage", "season_age",
    "RR_Lateral_Strip_PCT", "Cry1Ac_Bollgard_Strip_Test", "Cry2Ab_Bollgard_Strip_Test",
]
CAT_FEATURES = ["Origin_Region", "Grower_Region"]

WEATHER_FEATURES = [
    # Core thermal / moisture
    "cumulated_dd60", "avg_soil_moisture", "dd_60", "irrigation_type", "maczone",
    # Full-season environmental signals
    "total_precipitation", "total_solar_radiation", "season_heat_stress_days", "season_avg_vpd",
    # Post-planting 14-day window (pre-engineered in xlsx)
    "pp14_cum_dd60", "pp14_total_precip",
    # Boll fill period
    "boll_fill_cum_dd60", "boll_fill_total_precip", "boll_fill_heat_stress_days",
    # Pre-harvest 30 days
    "pre_harvest30_cum_dd60", "pre_harvest30_total_precip",
    # Post-defoliation period
    "post_defol_cumulated_dd60", "post_defol_avg_soilmoisture", "post_defol_total_precip",
]

ENGINEERED_FEATURES = [
    "season_age", "irrigation_is_dryland", "ct_distance_to_threshold",
    "cumulated_soil_moisture", "rm_band",
    # pp14_cum_dd60 / pp14_total_precip come pre-engineered from cotton_weather_features.xlsx
]

M1_FEATURES = list(dict.fromkeys(CORE_FEATURES + WEATHER_FEATURES + ENGINEERED_FEATURES))
M4_FEATURES = ["Moisture", "Mechanical_Damage", "Actual_Seed_Per_LB", "rm", "CT_Initial"]
M4_CAT      = ["Origin_Region"]
M4_THRESHOLD = 0.40

M5_FEATURES = ["rm", "cumulated_dd60", "avg_soil_moisture",
               "irrigation_is_dryland", "season_age"]
M5_CAT      = ["Origin_Region", "irrigation_type"]
M5_MIN_RECORDS_PER_BAND = 30

SURVIVAL_FEATURES = [
    "season_age",
    # WG_Current REMOVED — not available at prediction time
    "CT_Initial", "Moisture", "Mechanical_Damage",
    "rm", "Stage", "cumulated_dd60", "avg_soil_moisture",
    "irrigation_type", "ct_distance_to_threshold",
    "season_heat_stress_days", "pp14_cum_dd60",
]

FORBIDDEN = [
    "Variety", "WG_Current", "WG_Initial_7Day_Cnt",
    "nodes_above_white_flower_1", "nodes_above_white_flower_2",
    "nodes_above_white_flower_3", "defoliation_1",
]


# ── Helpers ────────────────────────────────────────────────────────

def assert_clean_features(feature_list, model_name):
    for f in FORBIDDEN:
        assert f not in feature_list, f"FORBIDDEN feature '{f}' in {model_name}"


def get_rm_band(rm):
    if pd.isna(rm):
        return "Unknown"
    for name, (lo, hi) in RM_BANDS.items():
        if lo <= rm < hi:
            return name
    return "Unknown"


def extract_rm_from_variety(variety):
    """Extract rm from variety name: last 2 digits of first 4-digit number (e.g. DP1646 → 46)."""
    if pd.isna(variety):
        return np.nan
    m = re.search(r"\d{4}", str(variety))
    return int(m.group(0)[-2:]) if m else np.nan


def log_join_coverage(df, col, step):
    n   = df[col].notna().sum()
    pct = n / len(df) * 100
    tag = "LOW" if pct < 10 else "OK"
    print(f"  [{tag}] JOIN [{step}]: {n:,}/{len(df):,} ({pct:.1f}%)")


def prep_for_model(df, features, cat_cols, target, imputer=None, le_dict=None, fit=False):
    all_cols = list(dict.fromkeys(features + cat_cols))
    X = df[[c for c in all_cols if c in df.columns]].copy()
    y = df[target].copy()
    if le_dict is None:
        le_dict = {}
    # Encode ALL string/object columns (rm_band, irrigation_type, maczone, Origin_Region, etc.)
    str_cols = [c for c in X.columns
                if X[c].dtype == object or str(X[c].dtype).startswith("category")]
    for col in str_cols:
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("Unknown"))
            le_dict[col] = le
        else:
            le = le_dict.get(col)
            if le is not None:
                X[col] = X[col].astype(str).fillna("Unknown").map(
                    lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
                )
            else:
                X[col] = -1
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X) if fit else imputer.transform(X)
    return X_imp, y, imputer, le_dict


# ── STEP 1: Load data ──────────────────────────────────────────────

def load_data():
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    lin  = pd.read_csv(DATA_PATH / "vw_cotton_lineage_and_quality_june_fg_all_cols.csv",
                       low_memory=False)
    qlty = pd.read_csv(DATA_PATH / "vw_cotton_qlty_rslt_all_cols.csv", low_memory=False)
    wm   = pd.read_excel(DATA_PATH / "cotton_weather_features.xlsx")

    print(f"  lineage:        {lin.shape[0]:>7,} rows  {lin.shape[1]:>4} cols")
    print(f"  quality_results:{qlty.shape[0]:>7,} rows  {qlty.shape[1]:>4} cols")
    print(f"  weather (xlsx): {wm.shape[0]:>7,} rows  {wm.shape[1]:>4} cols  (pre-aggregated, expect ~1,716)")
    return lin, qlty, wm


# ── STEP 2: Data quality rules (CLAUDE.md §4.3) ────────────────────

def apply_data_quality_rules(df):
    print("\nSTEP 2: Applying data quality rules")

    mask = (df["CT_Current"].notna() & df["WG_Current"].notna() &
            (df["CT_Current"] > df["WG_Current"]))
    df = df[~mask].copy()
    print(f"  Removed {mask.sum()} CT>WG anomalies")

    df = df.drop(columns=["Cleanout_PCT"], errors="ignore")

    if "FFA" in df.columns:
        df["FFA"] = df["FFA"].clip(upper=5.0)

    if "Seed_Temperature" in df.columns:
        df["Seed_Temperature"] = df["Seed_Temperature"].where(
            df["Seed_Temperature"].between(50, 110), np.nan)

    drop_nawf = ["nodes_above_white_flower_1", "nodes_above_white_flower_2",
                 "nodes_above_white_flower_3", "defoliation_1", "seeding_rate"]
    df = df.drop(columns=[c for c in drop_nawf if c in df.columns])

    dt_cols = ["CT_Initial_DT", "CT_Current_DT", "CT_Initial"]
    if all(c in df.columns for c in dt_cols):
        df["CT_Initial_DT"] = pd.to_datetime(df["CT_Initial_DT"], errors="coerce")
        df["CT_Current_DT"] = pd.to_datetime(df["CT_Current_DT"], errors="coerce")
        gap     = (df["CT_Current_DT"] - df["CT_Initial_DT"]).dt.days
        invalid = gap.notna() & gap.between(0, CT_INITIAL_RETEST_DAYS)
        df.loc[invalid, "CT_Initial"] = np.nan
        print(f"  CT_Initial nulled for {invalid.sum()} retest-within-14-day records")

    df["degraded_binary"] = (df["CT_Current"] < CT_THRESHOLD).astype("Int64")
    df["quality_class"]   = pd.cut(df["CT_Current"],
                                    bins=[-0.01, 60, 80, 100], labels=[0, 1, 2])
    df["season_age"]      = CURRENT_YEAR - df["SEASON_YR"]

    print(f"  Base rows: {len(df):,}  |  CT coverage: {df['CT_Current'].notna().mean()*100:.1f}%"
          f"  |  Degradation rate: {df['degraded_binary'].mean()*100:.1f}%")
    return df


# ── STEP 3: Weather (pre-aggregated xlsx — no aggregation needed) ──

def load_weather(wm):
    """cotton_weather_features.xlsx is already one row per variety+state+pa_year."""
    print("\nSTEP 3: Weather data (pre-aggregated — no daily-row processing needed)")
    group_cols = [c for c in ["variety", "state", "pa_year"] if c in wm.columns]
    if group_cols:
        print(f"  Join key columns present: {group_cols}")
    else:
        print("  WARNING: expected join keys (variety, state, pa_year) not found in weather file")
    present = [c for c in WEATHER_FEATURES if c in wm.columns]
    missing = [c for c in WEATHER_FEATURES if c not in wm.columns]
    print(f"  Weather features present: {len(present)}/{len(WEATHER_FEATURES)}")
    if missing:
        print(f"  Missing (will be NaN after join): {missing}")
    return wm


# ── STEP 4: Join lineage + weather ────────────────────────────────

def join_weather(base, wm):
    print("\nSTEP 4: Joining weather to lineage (variety + state + pa_year)")
    if wm.empty or "Variety" not in base.columns:
        print("  WARNING: skipping weather join — extracting rm from variety names")
        enriched = base.copy()
        enriched["rm"] = enriched["Variety"].apply(extract_rm_from_variety)
        print(f"  rm coverage via name: {enriched['rm'].notna().mean()*100:.1f}%")
        return enriched

    enriched = base.merge(
        wm,
        left_on  = ["Variety", "Origin_Region", "SEASON_YR"],
        right_on = ["variety", "state",          "pa_year"],
        how      = "left",
    )
    drop_dups = [c for c in ["variety", "state", "pa_year"] if c in enriched.columns]
    enriched  = enriched.drop(columns=drop_dups)
    log_join_coverage(enriched, "cumulated_dd60", "weather")

    # Fill rm for rows with no weather match (variety name extraction fallback)
    if "rm" not in enriched.columns:
        enriched["rm"] = np.nan
    missing_rm = enriched["rm"].isna()
    enriched.loc[missing_rm, "rm"] = enriched.loc[missing_rm, "Variety"].apply(extract_rm_from_variety)
    rm_cov = enriched["rm"].notna().mean() * 100
    print(f"  rm coverage (join + name extraction): {rm_cov:.1f}%")
    return enriched


# ── STEP 5: Feature engineering ───────────────────────────────────

def engineer_features(df):
    print("\nSTEP 5: Feature engineering")
    df = df.copy()

    df["rm_band"] = df["rm"].apply(get_rm_band)

    dryland_types = ["FurrowSurge", "Flood", "FurrowConventional",
                     "GatedPipePhaucet", "Furrow", "GatedPipeFaucet"]
    if "irrigation_type" in df.columns:
        df["irrigation_is_dryland"] = df["irrigation_type"].isin(dryland_types).astype(int)
    else:
        df["irrigation_is_dryland"] = np.nan

    df["ct_distance_to_threshold"] = df["CT_Current"] - CT_THRESHOLD
    df["season_age"] = CURRENT_YEAR - df["SEASON_YR"]

    # cumulated_soil_moisture: full-season aggregate proxy when not already in weather file
    if "cumulated_soil_moisture" not in df.columns:
        df["cumulated_soil_moisture"] = df.get("avg_soil_moisture", pd.Series(np.nan, index=df.index))

    # pp14_cum_dd60 and all new weather window features come pre-engineered from xlsx — no computation needed

    print(f"  rm_band distribution: {df['rm_band'].value_counts().to_dict()}")
    return df


# ── STEP 6: Temporal split ─────────────────────────────────────────

def temporal_split(df):
    print("\nSTEP 6: Temporal split")
    base_ct = df[df["CT_Current"].notna()].copy()
    train = base_ct[base_ct["SEASON_YR"].isin(TRAIN_SEASONS)].copy()
    val   = base_ct[base_ct["SEASON_YR"].isin(VAL_SEASONS)].copy()
    test  = base_ct[base_ct["SEASON_YR"].isin(TEST_SEASONS)].copy()
    print(f"  Train {TRAIN_SEASONS}: {len(train):,}")
    print(f"  Val   {VAL_SEASONS}:        {len(val):,}")
    print(f"  Test  {TEST_SEASONS}:    {len(test):,}")
    return train, val, test


# ── M1: Binary Degradation Classifier ─────────────────────────────

def train_m1(train, val, test):
    print("\n" + "=" * 60)
    print("LotGuard — Risk Scorer (XGBoost)")
    print("=" * 60)

    assert_clean_features(M1_FEATURES, "M1")

    le_m1 = {}
    Xtr, ytr, imp_m1, le_m1 = prep_for_model(train, M1_FEATURES, CAT_FEATURES, "degraded_binary", fit=True)
    Xva, yva, _, _           = prep_for_model(val,   M1_FEATURES, CAT_FEATURES, "degraded_binary", imp_m1, le_m1)
    Xte, yte, _, _           = prep_for_model(test,  M1_FEATURES, CAT_FEATURES, "degraded_binary", imp_m1, le_m1)
    ytr, yva, yte = ytr.astype(int), yva.astype(int), yte.astype(int)

    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    scale_pos = neg / max(pos, 1)

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 400),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight":   scale_pos,
            "eval_metric":        "auc",
            "early_stopping_rounds": 20,
            "random_state": RANDOM_STATE, "n_jobs": -1,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best = {**study.best_params, "scale_pos_weight": scale_pos, "eval_metric": "auc",
            "early_stopping_rounds": 20, "random_state": RANDOM_STATE, "n_jobs": -1}
    m1 = xgb.XGBClassifier(**best)
    m1.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    proba = m1.predict_proba(Xte)[:, 1]
    auc   = roc_auc_score(yte, proba)
    pr_auc = average_precision_score(yte, proba)
    status = "PASS" if auc >= 0.80 else "FAIL"
    print(f"  [{status}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  (target: >=0.80)")

    return m1, imp_m1, le_m1, {"m1_auc": auc, "m1_pr_auc": pr_auc}


# ── M2: CT Score Regressor ─────────────────────────────────────────

def train_m2(train, val, test):
    print("\n" + "=" * 60)
    print("QualityScope — CT Score Predictor (LightGBM)")
    print("=" * 60)

    le_m2 = {}
    Xtr, ytr, imp_m2, le_m2 = prep_for_model(train, M1_FEATURES, CAT_FEATURES, "CT_Current", fit=True)
    Xva, yva, _, _           = prep_for_model(val,   M1_FEATURES, CAT_FEATURES, "CT_Current", imp_m2, le_m2)
    Xte, yte, _, _           = prep_for_model(test,  M1_FEATURES, CAT_FEATURES, "CT_Current", imp_m2, le_m2)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 100),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        return -root_mean_squared_error(yva, m.predict(Xva))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    m2 = lgb.LGBMRegressor(**{**study.best_params, "random_state": RANDOM_STATE,
                                "n_jobs": -1, "verbosity": -1})
    m2.fit(Xtr, ytr, eval_set=[(Xva, yva)],
           callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

    pred = m2.predict(Xte)
    rmse = root_mean_squared_error(yte, pred)
    mae  = mean_absolute_error(yte, pred)
    r2   = r2_score(yte, pred)
    status = "PASS" if rmse < 10.0 else "FAIL"
    print(f"  [{status}] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}  (target: RMSE<10)")

    return m2, imp_m2, le_m2, {"m2_rmse": rmse, "m2_mae": mae, "m2_r2": r2}


# ── M3: 3-Class Quality Classifier ────────────────────────────────

def train_m3(train, val, test):
    print("\n" + "=" * 60)
    print("GradeView — Quality Grader (XGBoost + SHAP)")
    print("=" * 60)

    le_m3 = {}
    Xtr, ytr, imp_m3, le_m3 = prep_for_model(train, M1_FEATURES, CAT_FEATURES, "quality_class", fit=True)
    Xva, yva, _, _           = prep_for_model(val,   M1_FEATURES, CAT_FEATURES, "quality_class", imp_m3, le_m3)
    Xte, yte, _, _           = prep_for_model(test,  M1_FEATURES, CAT_FEATURES, "quality_class", imp_m3, le_m3)
    ytr, yva, yte = ytr.astype(int), yva.astype(int), yte.astype(int)

    m3 = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softmax", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=25,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    m3.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    pred = m3.predict(Xte)
    f1_macro = f1_score(yte, pred, average="macro")
    status = "PASS" if f1_macro >= 0.72 else "FAIL"
    print(f"  [{status}] Macro-F1={f1_macro:.4f}  (target: >=0.72)")

    return m3, imp_m3, le_m3, {"m3_f1_macro": f1_macro}


# ── M4: Stage 1 Screen ─────────────────────────────────────────────

def train_m4(train, val, test):
    print("\n" + "=" * 60)
    print("M4 — Stage 1 Early Reject Screen (XGBoost, intake-only features)")
    print("=" * 60)

    assert_clean_features(M4_FEATURES + M4_CAT, "M4")

    s1_tr = train[train["Stage"] == 1].copy()
    s1_va = val[val["Stage"] == 1].copy()
    s1_te = test[test["Stage"] == 1].copy()
    print(f"  Stage-1 rows: train={len(s1_tr):,}  val={len(s1_va):,}  test={len(s1_te):,}")

    le_m4 = {}
    Xtr, ytr, imp_m4, le_m4 = prep_for_model(s1_tr, M4_FEATURES, M4_CAT, "degraded_binary", fit=True)
    Xva, yva, _, _           = prep_for_model(s1_va, M4_FEATURES, M4_CAT, "degraded_binary", imp_m4, le_m4)
    Xte, yte, _, _           = prep_for_model(s1_te, M4_FEATURES, M4_CAT, "degraded_binary", imp_m4, le_m4)
    ytr, yva, yte = ytr.astype(int), yva.astype(int), yte.astype(int)

    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    m4 = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="auc", early_stopping_rounds=20,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    m4.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    proba = m4.predict_proba(Xte)[:, 1]
    auc   = roc_auc_score(yte, proba)
    status = "PASS" if auc >= 0.75 else "FAIL"
    print(f"  [{status}] AUC={auc:.4f}  (target: >=0.75)")

    return m4, imp_m4, le_m4, {"m4_auc": auc}


# ── M5: rm-Band GDD Profiler ───────────────────────────────────────

def train_m5(enriched):
    print("\n" + "=" * 60)
    print("M5 — rm-Band GDD Profiler (LightGBM per rm band)")
    print("=" * 60)

    assert_clean_features(M5_FEATURES + M5_CAT, "M5")

    fe_ct    = enriched[enriched["CT_Current"].notna()].copy()
    fe_train = fe_ct[fe_ct["SEASON_YR"].isin(TRAIN_SEASONS)]
    fe_test  = fe_ct[fe_ct["SEASON_YR"].isin(TEST_SEASONS)]

    print(f"  rm_band train distribution: {fe_train['rm_band'].value_counts().to_dict()}")

    m5_band_models = {}
    for band_name in RM_BANDS:
        tr = fe_train[fe_train["rm_band"] == band_name]
        te = fe_test[fe_test["rm_band"]  == band_name]
        if len(tr) < M5_MIN_RECORDS_PER_BAND:
            print(f"  SKIP {band_name}: {len(tr)} train records (min={M5_MIN_RECORDS_PER_BAND})")
            continue

        le5 = {}
        Xtr = tr[[c for c in M5_FEATURES + M5_CAT if c in tr.columns]].copy()
        Xte = te[[c for c in M5_FEATURES + M5_CAT if c in te.columns]].copy() if len(te) > 0 else Xtr.iloc[:0].copy()
        for col in M5_CAT:
            if col in Xtr.columns:
                le = LabelEncoder()
                Xtr[col] = le.fit_transform(Xtr[col].astype(str).fillna("Unknown"))
                if len(Xte) > 0:
                    Xte[col] = Xte[col].astype(str).fillna("Unknown").map(
                        lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
                    )
                le5[col] = le

        imp5 = SimpleImputer(strategy="median")
        Xtr_imp = imp5.fit_transform(Xtr)
        Xte_imp = imp5.transform(Xte) if len(te) > 0 else np.empty((0, Xtr_imp.shape[1]))

        m5 = lgb.LGBMRegressor(n_estimators=300, num_leaves=40, learning_rate=0.08,
                                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                                random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
        m5.fit(Xtr_imp, tr["CT_Current"])

        rmse = float(root_mean_squared_error(te["CT_Current"], m5.predict(Xte_imp))) \
               if len(te) > 0 else float("nan")
        status = "PASS" if (np.isnan(rmse) or rmse < 12.0) else "FAIL"
        print(f"  [{status}] {band_name:<15} train={len(tr):,}  test={len(te):,}  RMSE={rmse:.2f}")
        m5_band_models[band_name] = {
            "model": m5, "imputer": imp5, "label_encoders": le5,
            "features": M5_FEATURES, "cat_features": M5_CAT,
        }

    print(f"  M5 bands trained: {list(m5_band_models.keys())}")
    return m5_band_models


# ── M6: Survival Model (Cox PH + Weibull AFT) ─────────────────────

def train_m6(enriched):
    print("\n" + "=" * 60)
    print("ShelfSight — Shelf-Life Predictor (Cox PH + Weibull AFT) [PRIMARY]")
    print("=" * 60)

    surv = enriched[enriched["CT_Current"].notna()].copy()
    surv["event"]    = (surv["CT_Current"] < CT_THRESHOLD).astype(bool)
    surv["duration"] = surv["season_age"].clip(lower=0)
    surv.loc[(surv["duration"] == 0) & surv["event"], "duration"] = 0.5

    cox_cols = ["duration", "event"] + [c for c in SURVIVAL_FEATURES if c != "ct_distance_to_threshold"]
    cox_df   = surv[[c for c in cox_cols if c in surv.columns]].copy()

    # Dummy encode Origin_Region
    if "Origin_Region" in cox_df.columns:
        cox_df = pd.get_dummies(cox_df, columns=["Origin_Region"], drop_first=True)
    if "irrigation_type" in cox_df.columns:
        cox_df = pd.get_dummies(cox_df, columns=["irrigation_type"], drop_first=True)
    if "maczone" in cox_df.columns:
        cox_df = cox_df.drop(columns=["maczone"])

    # Impute
    num_cols = [c for c in cox_df.columns if c not in ["duration", "event"]]
    imp_cox  = SimpleImputer(strategy="median")
    cox_df[num_cols] = imp_cox.fit_transform(cox_df[num_cols])

    cox_df = cox_df[cox_df["duration"] > 0].copy()
    print(f"  Survival dataset: {len(cox_df):,} rows  |  event rate: {cox_df['event'].mean()*100:.1f}%")

    # Cox PH
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)
    c_index = concordance_index(cox_df["duration"], -cph.predict_partial_hazard(cox_df), cox_df["event"])
    status = "PASS" if c_index >= 0.70 else "FAIL"
    print(f"  [{status}] Cox PH C-index: {c_index:.4f}  (target: >=0.70)")

    # Weibull AFT
    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)
    print(f"  Weibull AFT fitted")

    return cph, aft, imp_cox, {"m6_c_index": c_index}


# ── Save all artifacts ─────────────────────────────────────────────

def save_artifacts(m1, imp_m1, le_m1,
                   m2, imp_m2, le_m2,
                   m3, imp_m3, le_m3,
                   m4, imp_m4, le_m4,
                   m5_band_models,
                   cph, aft, imp_cox,
                   metrics):
    print("\n" + "=" * 60)
    print("Saving model artifacts")
    print("=" * 60)

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "m1_binary_classifier": m1,
        "m1_imputer":           imp_m1,
        "m1_label_encoders":    le_m1,
        "m2_ct_regressor":      m2,
        "m2_imputer":           imp_m2,
        "m2_label_encoders":    le_m2,
        "m3_3class_classifier": m3,
        "m3_imputer":           imp_m3,
        "m3_label_encoders":    le_m3,
        "m4_stage1_screen":     m4,
        "m4_imputer":           imp_m4,
        "m4_label_encoders":    le_m4,
        "m5_rmband_profiler":   m5_band_models,
        "m6_cox_ph":            cph,
        "m6_aft_weibull":       aft,
        "cox_imputer":          imp_cox,
    }

    for name, obj in artifacts.items():
        path = MODEL_PATH / f"{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        print(f"  Saved: {name}.pkl")

    metadata = {
        "version":                "5",
        "CT_THRESHOLD":           CT_THRESHOLD,
        "RANDOM_STATE":           RANDOM_STATE,
        "CURRENT_YEAR":           CURRENT_YEAR,
        "CT_INITIAL_RETEST_DAYS": CT_INITIAL_RETEST_DAYS,
        "TRAIN_SEASONS":          TRAIN_SEASONS,
        "VAL_SEASONS":            VAL_SEASONS,
        "TEST_SEASONS":           TEST_SEASONS,
        "CORE_FEATURES":          CORE_FEATURES,
        "CAT_FEATURES":           CAT_FEATURES,
        "WEATHER_FEATURES":       WEATHER_FEATURES,
        "ENGINEERED_FEATURES":    ENGINEERED_FEATURES,
        "M1_FEATURES":            M1_FEATURES,
        "M4_FEATURES":            M4_FEATURES,
        "M4_CAT":                 M4_CAT,
        "M5_FEATURES":            M5_FEATURES,
        "M5_CAT":                 M5_CAT,
        "rm_bands":               list(RM_BANDS.keys()),
        "SURVIVAL_FEATURES":      SURVIVAL_FEATURES,
        "survival_event_col":     "event",
        "survival_duration_col":  "duration",
        "survival_time_unit":     "growing_seasons",
        "metrics":                metrics,
    }

    with open(MODEL_PATH / "model_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print("  Saved: model_metadata.json")

    # Compliance checks (CLAUDE.md v5)
    assert "rm_bands" in metadata
    assert "Variety"             not in str(metadata["M1_FEATURES"])
    assert "WG_Current"          not in str(metadata["M1_FEATURES"])
    assert "WG_Initial_7Day_Cnt" not in str(metadata["M1_FEATURES"])
    assert "m5_rmband_profiler"  in artifacts
    print("\n  CLAUDE.md v5 compliance checks passed")
    print(f"  Artifacts saved to: {MODEL_PATH}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("Cotton Seed Quality Intelligence System — Model Training")
    print("CLAUDE.md v5 | rm replaces Variety | ShelfSight is primary deliverable")
    print("=" * 60)

    lin, qlty, wm = load_data()
    base          = apply_data_quality_rules(lin)
    wm_ready      = load_weather(wm)       # already aggregated — no daily-row processing
    enriched      = join_weather(base, wm_ready)
    enriched      = engineer_features(enriched)
    train, val, test = temporal_split(enriched)

    m1, imp_m1, le_m1, met1 = train_m1(train, val, test)
    m2, imp_m2, le_m2, met2 = train_m2(train, val, test)
    m3, imp_m3, le_m3, met3 = train_m3(train, val, test)
    m4, imp_m4, le_m4, met4 = train_m4(train, val, test)
    m5_band_models           = train_m5(enriched)
    cph, aft, imp_cox, met6  = train_m6(enriched)

    metrics = {**met1, **met2, **met3, **met4, **met6}

    save_artifacts(m1, imp_m1, le_m1,
                   m2, imp_m2, le_m2,
                   m3, imp_m3, le_m3,
                   m4, imp_m4, le_m4,
                   m5_band_models,
                   cph, aft, imp_cox,
                   metrics)

    print("\n" + "=" * 60)
    print("Training complete. Summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
