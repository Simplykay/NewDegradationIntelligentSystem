"""Survival analysis service — KM curves, Cox hazard ratios, log-rank tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from functools import lru_cache
from .data_loader import get_df
from .model_service import load_models
from ..config import settings

CT_THRESHOLD = settings.CT_THRESHOLD
CURRENT_YEAR = settings.CURRENT_YEAR


def _surv_df() -> pd.DataFrame:
    df = get_df()
    s  = df[df["CT_Current"].notna()].copy()
    s["event"]    = (s["CT_Current"] < CT_THRESHOLD).astype(bool)
    s["duration"] = (CURRENT_YEAR - s["SEASON_YR"]).clip(lower=0.5)
    return s


def _km_to_records(kmf) -> list[dict]:
    tl = kmf.timeline
    sf = kmf.survival_function_["KM_estimate"]
    ci = kmf.confidence_interval_
    points = []
    for t in tl:
        lo = float(ci.loc[t].iloc[0]) if t in ci.index else None
        hi = float(ci.loc[t].iloc[1]) if t in ci.index else None
        points.append({"time": float(t), "survival": float(sf.loc[t]),
                       "lower_ci": lo, "upper_ci": hi})
    return points


def km_overall() -> list[dict]:
    try:
        from lifelines import KaplanMeierFitter
        s = _surv_df()
        kmf = KaplanMeierFitter()
        kmf.fit(s["duration"], event_observed=s["event"])
        return _km_to_records(kmf)
    except Exception:
        return []


def km_by_group(group_col: str) -> dict[str, list[dict]]:
    try:
        from lifelines import KaplanMeierFitter
        s = _surv_df()
        if group_col not in s.columns:
            return {}
        result = {}
        for group, sub in s.groupby(group_col):
            if len(sub) < 10:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["duration"], event_observed=sub["event"], label=str(group))
            result[str(group)] = _km_to_records(kmf)
        return result
    except Exception:
        return {}


def km_by_region() -> dict:
    return km_by_group("Origin_Region")


def km_by_stage() -> dict:
    return km_by_group("Stage")


def km_by_rm_band() -> dict:
    return km_by_group("rm_band")


def cox_hazard_ratios() -> list[dict]:
    m = load_models()
    cph = m.get("m6_cox_ph")
    if cph is None:
        return []
    try:
        summary = cph.summary
        result = []
        for coef_name in summary.index:
            row = summary.loc[coef_name]
            result.append({
                "feature":      coef_name,
                "hazard_ratio": round(float(np.exp(row.get("coef", 0))), 4),
                "coef":         round(float(row.get("coef", 0)), 4),
                "p_value":      round(float(row.get("p", 1.0)), 4),
                "lower_ci":     round(float(row.get("exp(coef) lower 95%", np.nan)), 4),
                "upper_ci":     round(float(row.get("exp(coef) upper 95%", np.nan)), 4),
            })
        return sorted(result, key=lambda x: abs(x["coef"]), reverse=True)
    except Exception:
        return []


def logrank_test_results() -> list[dict]:
    try:
        from lifelines.statistics import multivariate_logrank_test
        s = _surv_df()
        results = []
        for col in ["Origin_Region", "Stage", "rm_band"]:
            if col not in s.columns:
                continue
            test = multivariate_logrank_test(s["duration"], s[col], s["event"])
            results.append({
                "group_by": col,
                "test_statistic": round(float(test.test_statistic), 4),
                "p_value": round(float(test.p_value), 4),
                "significant": bool(test.p_value < 0.05),
            })
        return results
    except Exception:
        return []


def median_shelflife() -> dict:
    try:
        from lifelines import KaplanMeierFitter
        s = _surv_df()
        kmf = KaplanMeierFitter()
        kmf.fit(s["duration"], event_observed=s["event"])
        raw_median = kmf.median_survival_time_
        # KM median is inf when fewer than 50% of lots ever degrade
        median = None if (raw_median is None or (isinstance(raw_median, float) and not np.isfinite(raw_median))) else round(float(raw_median), 2)
        return {
            "median_seasons": median,
            "p25_seasons":    round(float(s["duration"].quantile(0.25)), 2),
            "p75_seasons":    round(float(s["duration"].quantile(0.75)), 2),
            "event_rate_pct": round(s["event"].mean() * 100, 2),
        }
    except Exception:
        return {}


def predict_lot_survival(lot_row: dict) -> dict:
    m = load_models()
    aft = m.get("m6_aft_weibull")
    imp = m.get("cox_imputer")
    if aft is None or imp is None:
        return {"error": "M6 model not loaded"}
    try:
        row = pd.DataFrame([lot_row])
        aft_params = [c for c in aft.params_.index.get_level_values(1)
                      if c not in ("Intercept",)]
        row = row.reindex(columns=aft_params)
        num_cols = [c for c in aft_params if c in row.columns]
        row[num_cols] = imp.transform(row[num_cols])
        median_life = float(aft.predict_median(row).iloc[0])
        return {
            "median_shelf_life_seasons": round(median_life, 2),
            "sell_by_season":           int(CURRENT_YEAR + median_life),
        }
    except Exception as e:
        return {"error": str(e)}
