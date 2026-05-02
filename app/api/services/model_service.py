"""Loads trained models and provides inference methods."""
from __future__ import annotations
import pickle, json, numpy as np, pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Optional, List

from ..config import settings
from ..schemas import LotInput, SinglePredictionResponse

CT_THRESHOLD = settings.CT_THRESHOLD
CURRENT_YEAR = settings.CURRENT_YEAR

# rm values encoded as 2-digit suffix of variety number (e.g. DP1646 → rm=46)
RM_BANDS = {
    "Ultra-Early": (0,  20),
    "Early":       (20, 30),
    "Mid-Early":   (30, 40),
    "Mid-Full":    (40, 50),
    "Full":        (50, 100),
}
DRYLAND = ["FurrowSurge", "Flood", "FurrowConventional",
           "GatedPipePhaucet", "Furrow", "GatedPipeFaucet"]

M1_FEATURES = [
    "WG_Current", "CT_Initial", "WG_Initial_7Day_Cnt", "Moisture",
    "Mechanical_Damage", "Actual_Seed_Per_LB", "rm",
    "Stage", "season_age",
    "RR_Lateral_Strip_PCT", "Cry1Ac_Bollgard_Strip_Test", "Cry2Ab_Bollgard_Strip_Test",
    "cumulated_dd60", "avg_soil_moisture", "dd_60", "irrigation_type", "maczone",
    "season_age", "irrigation_is_dryland", "ct_distance_to_threshold",
    "pp_day5_cum_dd60", "pp_day10_cum_dd60",
    "pp_day5_avg_soilmoisture", "pp_day10_avg_soilmoisture",
    "cumulated_soil_moisture", "rm_band",
]
CAT_FEATURES = ["Origin_Region", "Grower_Region"]
QUALITY_LABELS = {0: "Degraded", 1: "At Risk", 2: "High Quality"}
RISK_LEVELS    = {0: "High", 1: "Medium", 2: "Low"}


def _load_pkl(name: str):
    path = settings.model_dir / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_models() -> dict:
    names = [
        "m1_binary_classifier", "m1_imputer", "m1_label_encoders",
        "m2_ct_regressor",      "m2_imputer", "m2_label_encoders",
        "m3_3class_classifier", "m3_imputer",
        "m4_stage1_screen",     "m4_imputer",
        "m5_rmband_profiler",
        "m6_cox_ph",            "m6_aft_weibull", "cox_imputer",
    ]
    return {n: _load_pkl(n) for n in names}


def models_loaded() -> bool:
    m = load_models()
    return m.get("m1_binary_classifier") is not None


def get_model_metrics() -> dict:
    meta_path = settings.model_dir / "model_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("metrics", {})


def get_rm_band(rm) -> str:
    if rm is None or (isinstance(rm, float) and np.isnan(rm)):
        return "Unknown"
    for name, (lo, hi) in RM_BANDS.items():
        if lo <= rm < hi:
            return name
    return "Unknown"


def _lot_to_row(lot: LotInput) -> pd.DataFrame:
    season_age = CURRENT_YEAR - lot.season_yr
    irr_dryland = int(lot.irrigation_type in DRYLAND) if lot.irrigation_type else np.nan
    dd60 = lot.cumulated_dd60 or np.nan
    sm   = lot.avg_soil_moisture or np.nan

    row = {
        "WG_Current":              lot.wg_current,
        "CT_Initial":              lot.ct_initial,
        "WG_Initial_7Day_Cnt":     np.nan,
        "Moisture":                lot.moisture,
        "Mechanical_Damage":       lot.mechanical_damage,
        "Actual_Seed_Per_LB":      lot.actual_seed_per_lb,
        "rm":                      lot.rm,
        "Stage":                   lot.stage,
        "season_age":              season_age,
        "RR_Lateral_Strip_PCT":    np.nan,
        "Cry1Ac_Bollgard_Strip_Test": np.nan,
        "Cry2Ab_Bollgard_Strip_Test": np.nan,
        "Origin_Region":           lot.origin_region,
        "Grower_Region":           np.nan,
        "cumulated_dd60":          dd60,
        "avg_soil_moisture":       sm,
        "dd_60":                   np.nan,
        "irrigation_type":         lot.irrigation_type,
        "maczone":                 lot.maczone,
        "irrigation_is_dryland":   irr_dryland,
        "ct_distance_to_threshold": np.nan,
        "pp_day5_cum_dd60":        dd60 * 0.015 if not np.isnan(dd60) else np.nan,
        "pp_day10_cum_dd60":       dd60 * 0.03  if not np.isnan(dd60) else np.nan,
        "pp_day5_avg_soilmoisture":  sm,
        "pp_day10_avg_soilmoisture": sm,
        "cumulated_soil_moisture":   sm,
        "rm_band":                 get_rm_band(lot.rm),
    }
    return pd.DataFrame([row])


def predict_single(lot: LotInput, lot_id: Optional[str] = None) -> SinglePredictionResponse:
    m = load_models()
    row = _lot_to_row(lot)

    def _encode_and_impute(row_df, features, cat_cols, imputer, le_dict):
        all_cols = list(dict.fromkeys(features + cat_cols))
        X = row_df[[c for c in all_cols if c in row_df.columns]].copy()
        str_cols = [c for c in X.columns
                    if X[c].dtype == object or str(X[c].dtype).startswith("category")]
        for col in str_cols:
            le = (le_dict or {}).get(col)
            if le is not None:
                X[col] = X[col].astype(str).fillna("Unknown").map(
                    lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
                )
            else:
                X[col] = -1
        if imputer is None:
            return X
        # Align columns to exactly what the imputer expects
        if hasattr(imputer, "feature_names_in_"):
            X = X.reindex(columns=imputer.feature_names_in_)
        out = imputer.transform(X)
        out_cols = list(imputer.get_feature_names_out()) if hasattr(imputer, "get_feature_names_out") else list(X.columns)
        return pd.DataFrame(out, columns=out_cols)

    # M1 — degradation probability
    deg_prob = 0.5
    if m["m1_binary_classifier"] and m["m1_imputer"]:
        X1 = _encode_and_impute(row, M1_FEATURES, CAT_FEATURES, m["m1_imputer"], m["m1_label_encoders"])
        deg_prob = float(m["m1_binary_classifier"].predict_proba(X1)[0, 1])

    # M2 — CT score
    ct_pred = float(CT_THRESHOLD - 5)
    if m["m2_ct_regressor"] and m["m2_imputer"]:
        X2 = _encode_and_impute(row, M1_FEATURES, CAT_FEATURES, m["m2_imputer"], m["m2_label_encoders"])
        ct_pred = float(m["m2_ct_regressor"].predict(X2)[0])

    # M3 — quality class
    q_class = 0
    if m["m3_3class_classifier"] and m["m3_imputer"]:
        X3 = _encode_and_impute(row, M1_FEATURES, CAT_FEATURES, m["m3_imputer"], {})
        q_class = int(m["m3_3class_classifier"].predict(X3)[0])

    # M6 — shelf-life (AFT Weibull)
    shelf_life = None
    if m["m6_aft_weibull"] and m["cox_imputer"]:
        try:
            aft     = m["m6_aft_weibull"]
            imp_cox = m["cox_imputer"]
            covariates = list(dict.fromkeys(
                c for c in aft.params_.index.get_level_values("covariate")
                if c != "Intercept"
            ))
            season_age_val = CURRENT_YEAR - lot.season_yr
            row_dict = {c: np.nan for c in covariates}
            for k, v in {
                "season_age": season_age_val, "WG_Current": lot.wg_current,
                "CT_Initial": lot.ct_initial, "Moisture": lot.moisture,
                "Mechanical_Damage": lot.mechanical_damage, "rm": lot.rm,
                "Stage": lot.stage, "cumulated_dd60": lot.cumulated_dd60,
                "avg_soil_moisture": lot.avg_soil_moisture,
            }.items():
                if k in row_dict and v is not None:
                    row_dict[k] = float(v)
            if lot.irrigation_type:
                key = f"irrigation_type_{lot.irrigation_type}"
                if key in row_dict:
                    row_dict[key] = 1.0
                    for c in covariates:
                        if c.startswith("irrigation_type_") and c != key:
                            row_dict[c] = 0.0
            aft_row = pd.DataFrame([row_dict])
            imp_cols = (list(imp_cox.feature_names_in_)
                        if hasattr(imp_cox, "feature_names_in_") else covariates)
            imp_vals = imp_cox.transform(aft_row.reindex(columns=imp_cols))
            for i, col in enumerate(imp_cols):
                if col in aft_row.columns:
                    aft_row[col] = imp_vals[0, i]
            aft_row = aft_row.reindex(columns=covariates)
            shelf_life = float(aft.predict_median(aft_row).iloc[0])
            shelf_life = max(0.0, min(shelf_life, 20.0))
        except Exception as _e:
            import traceback; traceback.print_exc()
            shelf_life = None

    rm_band = get_rm_band(lot.rm)

    if deg_prob >= 0.7:
        risk = "High"
        rec  = "Prioritise for immediate sale or processing. High degradation risk."
    elif deg_prob >= 0.4:
        risk = "Medium"
        rec  = "Monitor closely. Schedule re-test within 60 days."
    else:
        risk = "Low"
        rec  = "Suitable for storage. Review again next season."

    # Top-3 SHAP features
    shap_feats = None
    if m["m1_binary_classifier"] and m["m1_imputer"]:
        try:
            import shap
            X1 = _encode_and_impute(row, M1_FEATURES, CAT_FEATURES, m["m1_imputer"], m["m1_label_encoders"])
            explainer = shap.TreeExplainer(m["m1_binary_classifier"])
            sv = explainer.shap_values(X1)
            vals = sv[0] if isinstance(sv, list) else sv[0]
            feat_names = M1_FEATURES + [c for c in CAT_FEATURES if c not in M1_FEATURES]
            top3 = sorted(zip(feat_names[:len(vals)], vals), key=lambda x: abs(x[1]), reverse=True)[:3]
            shap_feats = [{"feature": f, "shap_value": float(v)} for f, v in top3]
        except Exception:
            pass

    return SinglePredictionResponse(
        lot_id=lot_id,
        degradation_prob=round(deg_prob, 4),
        predicted_ct_score=round(ct_pred, 2),
        quality_class=QUALITY_LABELS.get(q_class, "Unknown"),
        risk_level=risk,
        predicted_shelf_life_seasons=round(shelf_life, 2) if shelf_life else None,
        rm_band=rm_band,
        recommendation=rec,
        shap_top_features=shap_feats,
    )


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    m = load_models()
    for _, row in df.iterrows():
        lot = LotInput(
            rm=row.get("rm"),
            origin_region=str(row.get("origin_region", row.get("Origin_Region", "AZ"))),
            season_yr=int(row.get("season_yr", row.get("SEASON_YR", 2023))),
            stage=row.get("stage", row.get("Stage")),
            wg_current=row.get("wg_current", row.get("WG_Current")),
            ct_initial=row.get("ct_initial", row.get("CT_Initial")),
            moisture=row.get("moisture", row.get("Moisture")),
            mechanical_damage=row.get("mechanical_damage", row.get("Mechanical_Damage")),
            actual_seed_per_lb=row.get("actual_seed_per_lb", row.get("Actual_Seed_Per_LB")),
            cumulated_dd60=row.get("cumulated_dd60"),
            avg_soil_moisture=row.get("avg_soil_moisture"),
            irrigation_type=row.get("irrigation_type"),
        )
        pred = predict_single(lot, lot_id=str(row.get("lot_id", "")))
        results.append({
            "lot_id":                       pred.lot_id,
            "degradation_prob":             pred.degradation_prob,
            "predicted_ct_score":           pred.predicted_ct_score,
            "quality_class":                pred.quality_class,
            "risk_level":                   pred.risk_level,
            "predicted_shelf_life_seasons": pred.predicted_shelf_life_seasons,
            "rm_band":                      pred.rm_band,
        })
    return pd.DataFrame(results)
