"""HTTP client for the FastAPI backend, with automatic local-service fallback.

When the API is not reachable (e.g. Streamlit Cloud deployment without a
separate FastAPI server), all calls are served directly from the service
modules, so the dashboard works as a standalone Streamlit app.
"""
from __future__ import annotations
import os, sys, io
import httpx
import pandas as pd

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 30.0


# ── Ensure repo root is on sys.path so service imports work in local mode ──
_here      = os.path.dirname(os.path.abspath(__file__))           # app/dashboard
_repo_root = os.path.dirname(os.path.dirname(_here))              # repo root
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _api_reachable(base_url: str) -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            c.get(f"{base_url}/health")
        return True
    except Exception:
        return False


# ── HTTP client (used when FastAPI is running) ─────────────────────────────

class APIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base = base_url.rstrip("/")

    def _get(self, path: str, params: dict = None) -> dict | list:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.get(f"{self.base}{path}", params=params)
            r.raise_for_status()
            return r.json()

    def _post(self, path: str, json: dict = None) -> dict | list:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post(f"{self.base}{path}", json=json)
            r.raise_for_status()
            return r.json()

    def health(self) -> dict:
        return self._get("/health")

    # EDA
    def eda_overview(self) -> dict:           return self._get("/eda/overview")
    def eda_ct_distribution(self) -> dict:    return self._get("/eda/ct-distribution")
    def eda_seasonal_trends(self) -> list:    return self._get("/eda/seasonal-trends")
    def eda_regional(self) -> list:           return self._get("/eda/regional")
    def eda_stage_analysis(self) -> list:     return self._get("/eda/stage-analysis")
    def eda_rm_rankings(self) -> list:        return self._get("/eda/rm-rankings")
    def eda_wg_ct_gap(self) -> dict:          return self._get("/eda/wg-ct-gap")
    def eda_physical_quality(self) -> dict:   return self._get("/eda/physical-quality")
    def eda_correlation_matrix(self) -> dict: return self._get("/eda/correlation-matrix")
    def eda_weather_summary(self) -> dict:    return self._get("/eda/weather-summary")
    def eda_irrigation_breakdown(self) -> list: return self._get("/eda/irrigation-breakdown")

    # Predictions
    def predict_single(self, payload: dict) -> dict:
        return self._post("/predict/single", json=payload)
    def m1_feature_importance(self) -> list:  return self._get("/predict/m1/feature-importance")
    def model_metrics(self) -> dict:          return self._get("/predict/model-metrics")

    # Survival
    def survival_km_overall(self) -> list:       return self._get("/survival/km-overall")
    def survival_km_by_region(self) -> dict:     return self._get("/survival/km-by-region")
    def survival_km_by_stage(self) -> dict:      return self._get("/survival/km-by-stage")
    def survival_km_by_rm_band(self) -> dict:    return self._get("/survival/km-by-rm-band")
    def survival_cox_hazard_ratios(self) -> list: return self._get("/survival/cox-hazard-ratios")
    def survival_logrank_test(self) -> list:     return self._get("/survival/logrank-test")
    def survival_median_shelflife(self) -> dict: return self._get("/survival/median-shelflife")

    # Inventory
    def inventory_risk_summary(self) -> list:     return self._get("/inventory/risk-summary")
    def inventory_high_risk_lots(self) -> list:   return self._get("/inventory/high-risk-lots")
    def inventory_false_pass_lots(self) -> list:  return self._get("/inventory/false-pass-lots")
    def inventory_regional_risk_map(self) -> list: return self._get("/inventory/regional-risk-map")
    def inventory_rm_risk_table(self) -> list:    return self._get("/inventory/rm-risk-table")
    def inventory_business_questions(self) -> dict: return self._get("/inventory/business-questions")

    def predict_batch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        csv_bytes = df.to_csv(index=False).encode()
        with httpx.Client(timeout=120.0) as c:
            r = c.post(f"{self.base}/predict/batch",
                       files={"file": ("upload.csv", io.BytesIO(csv_bytes), "text/csv")})
            r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))


# ── Local client (calls services directly — no FastAPI needed) ─────────────

class LocalAPIClient:
    """Identical interface to APIClient; calls service modules directly."""

    def health(self) -> dict:
        try:
            from app.api.services.data_loader import get_df
            from app.api.services.model_service import models_loaded
            get_df()
            return {"status": "healthy", "data_loaded": True, "models_loaded": models_loaded()}
        except Exception:
            return {"status": "degraded", "data_loaded": False, "models_loaded": False}

    # ── EDA ──────────────────────────────────────────────────────────────
    def eda_overview(self) -> dict:
        from app.api.services import eda_service
        return eda_service.overview()

    def eda_ct_distribution(self) -> dict:
        from app.api.services import eda_service
        return eda_service.ct_distribution()

    def eda_seasonal_trends(self) -> list:
        from app.api.services import eda_service
        return eda_service.seasonal_trends()

    def eda_regional(self) -> list:
        from app.api.services import eda_service
        return eda_service.regional_stats()

    def eda_stage_analysis(self) -> list:
        from app.api.services import eda_service
        return eda_service.stage_analysis()

    def eda_rm_rankings(self) -> list:
        from app.api.services import eda_service
        return eda_service.rm_rankings()

    def eda_wg_ct_gap(self) -> dict:
        from app.api.services import eda_service
        return eda_service.wg_ct_gap()

    def eda_physical_quality(self) -> dict:
        from app.api.services import eda_service
        return eda_service.physical_quality()

    def eda_correlation_matrix(self) -> dict:
        from app.api.services import eda_service
        return eda_service.correlation_matrix()

    def eda_weather_summary(self) -> dict:
        from app.api.services import eda_service
        return eda_service.weather_summary()

    def eda_irrigation_breakdown(self) -> list:
        from app.api.services import eda_service
        return eda_service.irrigation_breakdown()

    # ── Predictions ───────────────────────────────────────────────────────
    def predict_single(self, payload: dict) -> dict:
        from app.api.services.model_service import predict_single as _predict
        from app.api.schemas import LotInput
        lot = LotInput(**payload)
        result = _predict(lot, lot_id=payload.get("lot_id"))
        return result.model_dump()

    def m1_feature_importance(self) -> list:
        return []

    def model_metrics(self) -> dict:
        from app.api.services.model_service import get_model_metrics
        return get_model_metrics()

    # ── Survival ─────────────────────────────────────────────────────────
    def survival_km_overall(self) -> list:
        from app.api.services import survival_service
        return survival_service.km_overall()

    def survival_km_by_region(self) -> dict:
        from app.api.services import survival_service
        return survival_service.km_by_region()

    def survival_km_by_stage(self) -> dict:
        from app.api.services import survival_service
        return survival_service.km_by_stage()

    def survival_km_by_rm_band(self) -> dict:
        from app.api.services import survival_service
        return survival_service.km_by_rm_band()

    def survival_cox_hazard_ratios(self) -> list:
        from app.api.services import survival_service
        return survival_service.cox_hazard_ratios()

    def survival_logrank_test(self) -> list:
        from app.api.services import survival_service
        return survival_service.logrank_test_results()

    def survival_median_shelflife(self) -> dict:
        from app.api.services import survival_service
        return survival_service.median_shelflife()

    # ── Inventory ─────────────────────────────────────────────────────────
    def inventory_risk_summary(self) -> list:
        from app.api.services.data_loader import get_df
        CT = 60.0
        df = get_df()
        ct = df[df["CT_Current"].notna()].copy()
        import pandas as pd
        ct["risk_tier"] = pd.cut(ct["CT_Current"], bins=[-0.01, CT, 80, 100],
                                  labels=["Degraded", "At Risk", "High Quality"])
        counts = ct["risk_tier"].value_counts()
        total  = len(ct)
        return [{"tier": str(t), "lot_count": int(counts.get(t, 0)),
                 "percentage": round(counts.get(t, 0) / max(total, 1) * 100, 2)}
                for t in ["Degraded", "At Risk", "High Quality"]]

    def inventory_high_risk_lots(self) -> list:
        from app.api.services.data_loader import get_df
        CT = 60.0
        df = get_df()
        ct = df[df["CT_Current"].notna()].copy()
        ct["deg_prob"] = (ct["CT_Current"] < CT).astype(float)
        hr  = ct[ct["deg_prob"] >= 0.7].head(100)
        cols = [c for c in ["INSPCT_LOT_NBR","Variety","Origin_Region","Stage",
                             "CT_Current","WG_Current","SEASON_YR","rm_band","deg_prob"]
                if c in hr.columns]
        return hr[cols].round(3).to_dict(orient="records")

    def inventory_false_pass_lots(self) -> list:
        from app.api.services.data_loader import get_df
        CT = 60.0
        df = get_df()
        both = df[df["WG_Current"].notna() & df["CT_Current"].notna()].copy()
        fp   = both[(both["WG_Current"] >= 80) & (both["CT_Current"] < CT)].head(100)
        cols = [c for c in ["INSPCT_LOT_NBR","Variety","Origin_Region","Stage",
                             "CT_Current","WG_Current","SEASON_YR","rm_band"]
                if c in fp.columns]
        return fp[cols].round(3).to_dict(orient="records")

    def inventory_regional_risk_map(self) -> list:
        from app.api.services.data_loader import get_df
        CT = 60.0
        df = get_df()
        ct = df[df["CT_Current"].notna()]
        result = (ct.groupby("Origin_Region")
                    .agg(lot_count=("CT_Current","count"),
                         degraded_pct=("CT_Current", lambda x: (x < CT).mean() * 100),
                         mean_ct=("CT_Current","mean"))
                    .reset_index().rename(columns={"Origin_Region":"region"}).round(2))
        return result.to_dict(orient="records")

    def inventory_rm_risk_table(self) -> list:
        from app.api.services.data_loader import get_df
        CT = 60.0
        df = get_df()
        if "rm_band" not in df.columns:
            return []
        ct = df[df["CT_Current"].notna()]
        result = (ct.groupby("rm_band")
                    .agg(lot_count=("CT_Current","count"),
                         degraded_pct=("CT_Current", lambda x: (x < CT).mean() * 100),
                         mean_ct=("CT_Current","mean"),
                         mean_season_age=("season_age","mean"))
                    .reset_index().round(2))
        return result.to_dict(orient="records")

    def inventory_business_questions(self) -> dict:
        from app.api.services.data_loader import get_df
        CT = 60.0
        df   = get_df()
        ct   = df[df["CT_Current"].notna()]
        both = df[df["WG_Current"].notna() & df["CT_Current"].notna()]
        fp   = both[(both["WG_Current"] >= 80) & (both["CT_Current"] < CT)]
        stage_deg = {}
        if "Stage" in ct.columns:
            stage_deg = {int(k): round(float(v) * 100, 2)
                         for k, v in ct.groupby("Stage")["CT_Current"]
                         .apply(lambda x: (x < CT).mean()).items()}
        return {
            "portfolio_degraded_pct":  round((ct["CT_Current"] < CT).mean() * 100, 2),
            "portfolio_at_risk_pct":   round(ct["CT_Current"].between(CT, 80).mean() * 100, 2),
            "false_pass_count":        int(len(fp)),
            "worst_region":            ct.groupby("Origin_Region")["CT_Current"]
                                         .apply(lambda x: (x < CT).mean()).idxmax()
                                         if "Origin_Region" in ct.columns else None,
            "degraded_pct_by_stage":   stage_deg,
            "total_lots":              int(len(df)),
            "ct_tested_lots":          int(len(ct)),
        }

    def predict_batch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        from app.api.services.model_service import predict_batch
        return predict_batch(df)


# ── Auto-detect: use API if reachable, else fall back to local services ────
_USE_API = os.getenv("USE_API", "auto").lower()
if _USE_API == "false" or _USE_API == "0":
    client = LocalAPIClient()
elif _USE_API == "true" or _USE_API == "1":
    client = APIClient()
else:
    client = APIClient() if _api_reachable(API_BASE_URL) else LocalAPIClient()
