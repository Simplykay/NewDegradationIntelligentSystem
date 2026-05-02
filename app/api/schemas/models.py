from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class LotInput(BaseModel):
    rm:                 Optional[float] = Field(None, description="Relative Maturity (80-145)")
    origin_region:      str             = Field(..., description="US state code: AZ, TX, CA, MS, AR, NM")
    season_yr:          int             = Field(..., description="Harvest season year")
    stage:              Optional[int]   = Field(None, description="Pipeline stage 1-5")
    wg_current:         Optional[float] = Field(None, description="Warm germination %")
    ct_initial:         Optional[float] = Field(None, description="Initial cool test %")
    moisture:           Optional[float] = Field(None, description="Seed moisture %")
    mechanical_damage:  Optional[float] = Field(None, description="Mechanical damage %")
    actual_seed_per_lb: Optional[float] = Field(None, description="Seeds per pound")
    cumulated_dd60:     Optional[float] = Field(None, description="Cumulative degree days above 60F")
    avg_soil_moisture:  Optional[float] = Field(None, description="Average soil moisture fraction 0-1")
    irrigation_type:    Optional[str]   = Field(None, description="Irrigation system type")
    maczone:            Optional[str]   = Field(None, description="Management zone 1-8")


class SinglePredictionResponse(BaseModel):
    lot_id:                       Optional[str]        = None
    degradation_prob:             float
    predicted_ct_score:           float
    quality_class:                str
    risk_level:                   str
    predicted_shelf_life_seasons: Optional[float]      = None
    rm_band:                      Optional[str]        = None
    recommendation:               str
    shap_top_features:            Optional[List[dict]] = None


class BatchPredictionRow(BaseModel):
    lot_id:                       Optional[str] = None
    degradation_prob:             float
    predicted_ct_score:           float
    quality_class:                str
    risk_level:                   str
    predicted_shelf_life_seasons: Optional[float] = None
    rm_band:                      Optional[str]   = None


class SurvivalKMPoint(BaseModel):
    time:        float
    survival:    float
    lower_ci:    Optional[float] = None
    upper_ci:    Optional[float] = None
    at_risk:     Optional[int]   = None
    events:      Optional[int]   = None


class RiskSummary(BaseModel):
    tier:              str
    lot_count:         int
    percentage:        float
    avg_degradation_prob: Optional[float] = None


class ModelMetrics(BaseModel):
    m1_auc:      Optional[float] = None
    m1_pr_auc:   Optional[float] = None
    m2_rmse:     Optional[float] = None
    m2_r2:       Optional[float] = None
    m3_f1_macro: Optional[float] = None
    m4_auc:      Optional[float] = None
    m6_c_index:  Optional[float] = None
