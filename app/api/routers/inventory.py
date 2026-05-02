from fastapi import APIRouter, Query
import pandas as pd
from ..services.data_loader import get_df
from ..services.model_service import load_models
from ..config import settings

CT_THRESHOLD = settings.CT_THRESHOLD
CURRENT_YEAR = settings.CURRENT_YEAR

router = APIRouter(prefix="/inventory", tags=["inventory"])


def _add_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "degraded_binary" not in df.columns:
        df["degraded_binary"] = (df["CT_Current"] < CT_THRESHOLD).astype("Int64")
    return df


@router.get("/risk-summary")
def risk_summary():
    df = get_df()
    ct = df[df["CT_Current"].notna()].copy()
    ct["risk_tier"] = pd.cut(ct["CT_Current"],
                              bins=[-0.01, CT_THRESHOLD, 80, 100],
                              labels=["Degraded", "At Risk", "High Quality"])
    counts = ct["risk_tier"].value_counts()
    total  = len(ct)
    return [
        {"tier": str(tier), "lot_count": int(counts.get(tier, 0)),
         "percentage": round(counts.get(tier, 0) / max(total, 1) * 100, 2)}
        for tier in ["Degraded", "At Risk", "High Quality"]
    ]


@router.get("/high-risk-lots")
def high_risk_lots(threshold: float = Query(0.7, ge=0.0, le=1.0), limit: int = Query(100)):
    df = get_df()
    ct = df[df["CT_Current"].notna()].copy()
    ct["deg_prob"] = (ct["CT_Current"] < CT_THRESHOLD).astype(float)
    high_risk = ct[ct["deg_prob"] >= threshold].head(limit)
    cols = [c for c in ["INSPCT_LOT_NBR", "Variety", "Origin_Region", "Stage",
                         "CT_Current", "WG_Current", "SEASON_YR", "rm_band", "deg_prob"]
            if c in high_risk.columns]
    return high_risk[cols].round(3).to_dict(orient="records")


@router.get("/false-pass-lots")
def false_pass_lots(limit: int = Query(100)):
    df = get_df()
    both = df[df["WG_Current"].notna() & df["CT_Current"].notna()].copy()
    fp   = both[(both["WG_Current"] >= 80) & (both["CT_Current"] < CT_THRESHOLD)].head(limit)
    cols = [c for c in ["INSPCT_LOT_NBR", "Variety", "Origin_Region", "Stage",
                         "CT_Current", "WG_Current", "SEASON_YR", "rm_band"]
            if c in fp.columns]
    return fp[cols].round(3).to_dict(orient="records")


@router.get("/expiring-soon")
def expiring_soon(seasons_threshold: float = Query(1.5)):
    df = get_df()
    ct = df[df["CT_Current"].notna() & df["season_age"].notna()].copy()
    expiring = ct[ct["season_age"] >= seasons_threshold].copy()
    expiring = expiring.sort_values("season_age", ascending=False).head(200)
    cols = [c for c in ["INSPCT_LOT_NBR", "Variety", "Origin_Region", "Stage",
                         "CT_Current", "SEASON_YR", "season_age", "rm_band"]
            if c in expiring.columns]
    return expiring[cols].round(3).to_dict(orient="records")


@router.get("/regional-risk-map")
def regional_risk_map():
    df = get_df()
    ct = df[df["CT_Current"].notna()]
    result = (ct.groupby("Origin_Region")
                .agg(lot_count=("CT_Current", "count"),
                     degraded_pct=("CT_Current", lambda x: (x < CT_THRESHOLD).mean() * 100),
                     mean_ct=("CT_Current", "mean"))
                .reset_index()
                .rename(columns={"Origin_Region": "region"})
                .round(2))
    return result.to_dict(orient="records")


@router.get("/rm-risk-table")
def rm_risk_table():
    df = get_df()
    if "rm_band" not in df.columns:
        return []
    ct = df[df["CT_Current"].notna()]
    result = (ct.groupby("rm_band")
                .agg(lot_count=("CT_Current", "count"),
                     degraded_pct=("CT_Current", lambda x: (x < CT_THRESHOLD).mean() * 100),
                     mean_ct=("CT_Current", "mean"),
                     mean_season_age=("season_age", "mean"))
                .reset_index()
                .round(2))
    return result.to_dict(orient="records")


@router.get("/stage1-reject-screen")
def stage1_reject_screen():
    df   = get_df()
    s1   = df[(df["Stage"] == 1) & df["CT_Current"].notna()].copy()
    s1["predicted_reject"] = (s1["CT_Current"] < CT_THRESHOLD).astype(int)
    cols = [c for c in ["INSPCT_LOT_NBR", "Variety", "Origin_Region",
                         "CT_Current", "WG_Current", "SEASON_YR", "rm", "predicted_reject"]
            if c in s1.columns]
    return s1[cols].head(200).round(3).to_dict(orient="records")


@router.get("/business-questions")
def business_questions():
    df   = get_df()
    ct   = df[df["CT_Current"].notna()]
    both = df[df["WG_Current"].notna() & df["CT_Current"].notna()]
    fp   = both[(both["WG_Current"] >= 80) & (both["CT_Current"] < CT_THRESHOLD)]

    stage_deg = {}
    if "Stage" in ct.columns:
        stage_deg = {int(k): round(float(v) * 100, 2)
                     for k, v in ct.groupby("Stage")["CT_Current"]
                     .apply(lambda x: (x < CT_THRESHOLD).mean()).items()}

    return {
        "portfolio_degraded_pct":    round((ct["CT_Current"] < CT_THRESHOLD).mean() * 100, 2),
        "portfolio_at_risk_pct":     round(ct["CT_Current"].between(CT_THRESHOLD, 80).mean() * 100, 2),
        "false_pass_count":          int(len(fp)),
        "worst_region":              ct.groupby("Origin_Region")["CT_Current"]
                                       .apply(lambda x: (x < CT_THRESHOLD).mean())
                                       .idxmax() if "Origin_Region" in ct.columns else None,
        "degraded_pct_by_stage":     stage_deg,
        "total_lots":                int(len(df)),
        "ct_tested_lots":            int(len(ct)),
    }
