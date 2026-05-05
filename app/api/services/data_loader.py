"""Loads, cleans, joins, and enriches the cotton seed dataset. Cached singleton."""
from __future__ import annotations
import re, numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from ..config import settings

CT_THRESHOLD           = settings.CT_THRESHOLD
CURRENT_YEAR           = settings.CURRENT_YEAR
CT_INITIAL_RETEST_DAYS = settings.CT_INITIAL_RETEST_DAYS

RM_BANDS = {
    "Ultra-Early": (0,   100),
    "Early":       (100, 110),
    "Mid-Early":   (110, 120),
    "Mid-Full":    (120, 130),
    "Full":        (130, 999),
}

DRYLAND = ["FurrowSurge", "Flood", "FurrowConventional",
           "GatedPipePhaucet", "Furrow", "GatedPipeFaucet"]


def _extract_rm_from_variety(variety) -> float:
    if pd.isna(variety):
        return np.nan
    m = re.search(r"\d{4}", str(variety))
    return float(int(m.group(0)[-2:])) if m else np.nan


def get_rm_band(rm):
    if pd.isna(rm):
        return "Unknown"
    for name, (lo, hi) in RM_BANDS.items():
        if lo <= rm < hi:
            return name
    return "Unknown"


def _apply_quality_rules(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["CT_Current"].notna() & df["WG_Current"].notna() &
            (df["CT_Current"] > df["WG_Current"]))
    df = df[~mask].copy()
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

    df["degraded_binary"] = (df["CT_Current"] < CT_THRESHOLD).astype("Int64")
    df["quality_class"]   = pd.cut(df["CT_Current"],
                                    bins=[-0.01, 60, 80, 100], labels=[0, 1, 2])
    df["season_age"]      = CURRENT_YEAR - df["SEASON_YR"]
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rm_band"] = df["rm"].apply(get_rm_band)

    if "irrigation_type" in df.columns:
        df["irrigation_is_dryland"] = df["irrigation_type"].isin(DRYLAND).astype(int)
    else:
        df["irrigation_is_dryland"] = np.nan

    df["ct_distance_to_threshold"] = df["CT_Current"] - CT_THRESHOLD
    df["season_age"]               = CURRENT_YEAR - df["SEASON_YR"]

    # cumulated_soil_moisture: full-season aggregate proxy when not already in weather file
    if "cumulated_soil_moisture" not in df.columns:
        df["cumulated_soil_moisture"] = df.get(
            "avg_soil_moisture", pd.Series(np.nan, index=df.index))

    # pp14_cum_dd60 and all new weather window features come pre-engineered from xlsx
    return df


@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    data_dir = settings.data_dir

    lin = pd.read_csv(data_dir / "vw_cotton_lineage_and_quality_june_fg_all_cols.csv",
                      low_memory=False)
    wm  = pd.read_excel(data_dir / "cotton_weather_features.xlsx")

    base = _apply_quality_rules(lin)

    if not wm.empty and "Variety" in base.columns:
        enriched = base.merge(
            wm,
            left_on  = ["Variety", "Origin_Region", "SEASON_YR"],
            right_on = ["variety", "state",          "pa_year"],
            how      = "left",
        )
        enriched = enriched.drop(
            columns=[c for c in ["variety", "state", "pa_year"] if c in enriched.columns],
            errors="ignore",
        )
    else:
        enriched = base.copy()

    # Fill rm from variety name for rows with no weather match
    if "rm" not in enriched.columns:
        enriched["rm"] = np.nan
    missing_rm = enriched["rm"].isna()
    enriched.loc[missing_rm, "rm"] = (
        enriched.loc[missing_rm, "Variety"].apply(_extract_rm_from_variety)
    )

    return _engineer_features(enriched)


def get_weather_df() -> pd.DataFrame:
    return pd.read_excel(settings.data_dir / "cotton_weather_features.xlsx")
