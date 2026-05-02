from fastapi import APIRouter, Query
from ..services import eda_service as svc

router = APIRouter(prefix="/eda", tags=["eda"])


@router.get("/overview")
def overview():
    return svc.overview()


@router.get("/ct-distribution")
def ct_distribution():
    return svc.ct_distribution()


@router.get("/seasonal-trends")
def seasonal_trends():
    return svc.seasonal_trends()


@router.get("/regional")
def regional():
    return svc.regional_stats()


@router.get("/stage-analysis")
def stage_analysis():
    return svc.stage_analysis()


@router.get("/rm-rankings")
def rm_rankings(top_n: int = Query(20, ge=1, le=100)):
    return svc.rm_rankings(top_n)


@router.get("/wg-ct-gap")
def wg_ct_gap():
    return svc.wg_ct_gap()


@router.get("/physical-quality")
def physical_quality():
    return svc.physical_quality()


@router.get("/correlation-matrix")
def correlation_matrix():
    return svc.correlation_matrix()


@router.get("/weather-summary")
def weather_summary():
    return svc.weather_summary()


@router.get("/irrigation-breakdown")
def irrigation_breakdown():
    return svc.irrigation_breakdown()
