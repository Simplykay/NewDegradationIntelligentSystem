"""Computes EDA statistics from the processed dataset."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data_loader import get_df, get_weather_df
from ..config import settings

CT_THRESHOLD = settings.CT_THRESHOLD


def overview() -> dict:
    df = get_df()
    ct = df[df["CT_Current"].notna()]
    return {
        "total_lots":         int(len(df)),
        "ct_tested_lots":     int(len(ct)),
        "degraded_count":     int((ct["CT_Current"] < CT_THRESHOLD).sum()),
        "degraded_pct":       round((ct["CT_Current"] < CT_THRESHOLD).mean() * 100, 2),
        "at_risk_pct":        round(ct["CT_Current"].between(CT_THRESHOLD, 80).mean() * 100, 2),
        "high_quality_pct":   round((ct["CT_Current"] >= 80).mean() * 100, 2),
        "seasons":            sorted(df["SEASON_YR"].dropna().unique().tolist()),
        "regions":            sorted(df["Origin_Region"].dropna().unique().tolist()),
        "stages":             sorted(df["Stage"].dropna().unique().astype(int).tolist()),
        "ct_mean":            round(float(ct["CT_Current"].mean()), 2),
        "ct_std":             round(float(ct["CT_Current"].std()), 2),
        "wg_mean":            round(float(df["WG_Current"].mean()), 2) if "WG_Current" in df else None,
        "false_pass_count":   int(_false_pass_count(df)),
        "rm_null_pct":        round(df["rm"].isna().mean() * 100, 2) if "rm" in df else None,
        "weather_join_pct":   round(df["cumulated_dd60"].notna().mean() * 100, 2)
                              if "cumulated_dd60" in df else 0.0,
    }


def _false_pass_count(df: pd.DataFrame) -> int:
    mask = (df["WG_Current"].notna() & df["CT_Current"].notna() &
            (df["WG_Current"] >= 80) & (df["CT_Current"] < CT_THRESHOLD))
    return mask.sum()


def ct_distribution() -> dict:
    df = get_df()
    ct = df["CT_Current"].dropna()
    hist, edges = np.histogram(ct, bins=40)
    return {
        "bins":      [round(float(e), 2) for e in edges[:-1]],
        "counts":    [int(c) for c in hist],
        "threshold": CT_THRESHOLD,
        "degraded_count":    int((ct < CT_THRESHOLD).sum()),
        "at_risk_count":     int(ct.between(CT_THRESHOLD, 80).sum()),
        "high_quality_count": int((ct >= 80).sum()),
    }


def seasonal_trends() -> list[dict]:
    df = get_df()
    ct = df[df["CT_Current"].notna()]
    result = (ct.groupby("SEASON_YR")
                .agg(mean_ct=("CT_Current", "mean"),
                     degraded_pct=("CT_Current", lambda x: (x < CT_THRESHOLD).mean() * 100),
                     lot_count=("CT_Current", "count"))
                .reset_index())
    return result.rename(columns={"SEASON_YR": "season"}).round(2).to_dict(orient="records")


def regional_stats() -> list[dict]:
    df = get_df()
    ct = df[df["CT_Current"].notna()]
    result = (ct.groupby("Origin_Region")
                .agg(mean_ct=("CT_Current", "mean"),
                     degraded_pct=("CT_Current", lambda x: (x < CT_THRESHOLD).mean() * 100),
                     lot_count=("CT_Current", "count"))
                .reset_index())
    return result.rename(columns={"Origin_Region": "region"}).round(2).to_dict(orient="records")


def stage_analysis() -> list[dict]:
    df = get_df()
    ct = df[df["CT_Current"].notna() & df["Stage"].notna()]
    result = (ct.groupby("Stage")
                .agg(mean_ct=("CT_Current", "mean"),
                     degraded_pct=("CT_Current", lambda x: (x < CT_THRESHOLD).mean() * 100),
                     lot_count=("CT_Current", "count"))
                .reset_index())
    result["Stage"] = result["Stage"].astype(int)
    return result.round(2).to_dict(orient="records")


def rm_rankings(top_n: int = 20) -> list[dict]:
    df = get_df()
    if "rm_band" not in df.columns:
        return []
    ct = df[df["CT_Current"].notna()]
    result = (ct.groupby("rm_band")
                .agg(mean_ct=("CT_Current", "mean"),
                     degraded_pct=("CT_Current", lambda x: (x < CT_THRESHOLD).mean() * 100),
                     lot_count=("CT_Current", "count"))
                .reset_index()
                .sort_values("degraded_pct"))
    return result.round(2).to_dict(orient="records")


def wg_ct_gap() -> dict:
    df = get_df()
    both = df[df["WG_Current"].notna() & df["CT_Current"].notna()].copy()
    both["vigor_gap"] = both["WG_Current"] - both["CT_Current"]
    false_pass = both[(both["WG_Current"] >= 80) & (both["CT_Current"] < CT_THRESHOLD)]
    return {
        "mean_vigor_gap": round(float(both["vigor_gap"].mean()), 2),
        "false_pass_count": int(len(false_pass)),
        "false_pass_pct":   round(len(false_pass) / max(len(both), 1) * 100, 2),
        "wg_mean":          round(float(both["WG_Current"].mean()), 2),
        "ct_mean":          round(float(both["CT_Current"].mean()), 2),
    }


def physical_quality() -> dict:
    df = get_df()
    result = {}
    for col in ["Moisture", "Mechanical_Damage", "FFA", "Actual_Seed_Per_LB"]:
        if col in df.columns:
            s = df[col].dropna()
            result[col] = {
                "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "std": round(float(s.std()), 3),
                "p5": round(float(s.quantile(0.05)), 3),
                "p95": round(float(s.quantile(0.95)), 3),
                "null_pct": round(df[col].isna().mean() * 100, 2),
            }
    return result


def correlation_matrix() -> dict:
    df = get_df()
    candidate_cols = [
        "CT_Current", "WG_Current", "Moisture", "Mechanical_Damage",
        "rm", "season_age", "cumulated_dd60", "avg_soil_moisture",
        "season_heat_stress_days", "pp14_cum_dd60", "total_precipitation",
        "total_solar_radiation", "boll_fill_cum_dd60",
    ]
    num_cols = [c for c in candidate_cols if c in df.columns]
    corr = df[num_cols].corr().round(3)
    matrix = [[None if (v != v) else v for v in row] for row in corr.values.tolist()]
    return {"columns": corr.columns.tolist(), "matrix": matrix}


def weather_summary() -> dict:
    try:
        wm = get_weather_df()
        result = {}
        summary_cols = [
            "cumulated_dd60", "avg_soil_moisture", "dd_60",
            "total_precipitation", "total_solar_radiation",
            "season_heat_stress_days", "season_avg_vpd",
            "pp14_cum_dd60", "pp14_total_precip",
            "boll_fill_cum_dd60", "boll_fill_total_precip", "boll_fill_heat_stress_days",
            "pre_harvest30_cum_dd60", "pre_harvest30_total_precip",
            "post_defol_cumulated_dd60", "post_defol_avg_soilmoisture", "post_defol_total_precip",
        ]
        for col in summary_cols:
            if col in wm.columns:
                s = wm[col].dropna()
                if len(s) > 0:
                    result[col] = {
                        "mean":     round(float(s.mean()), 3),
                        "max":      round(float(s.max()),  3),
                        "min":      round(float(s.min()),  3),
                        "null_pct": round(wm[col].isna().mean() * 100, 2),
                    }
        if "irrigation_type" in wm.columns:
            result["irrigation_breakdown"] = wm["irrigation_type"].value_counts().to_dict()
        return result
    except Exception:
        return {}


def irrigation_breakdown() -> list[dict]:
    try:
        wm = get_weather_df()
        if "irrigation_type" not in wm.columns:
            return []
        counts = wm["irrigation_type"].value_counts().reset_index()
        counts.columns = ["irrigation_type", "field_count"]
        counts["pct"] = (counts["field_count"] / counts["field_count"].sum() * 100).round(2)
        return counts.to_dict(orient="records")
    except Exception:
        return []
