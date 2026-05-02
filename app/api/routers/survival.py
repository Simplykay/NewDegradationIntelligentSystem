from fastapi import APIRouter
from ..services import survival_service as svc

router = APIRouter(prefix="/survival", tags=["survival"])


@router.get("/km-overall")
def km_overall():
    return svc.km_overall()


@router.get("/km-by-region")
def km_by_region():
    return svc.km_by_region()


@router.get("/km-by-stage")
def km_by_stage():
    return svc.km_by_stage()


@router.get("/km-by-rm-band")
def km_by_rm_band():
    return svc.km_by_rm_band()


@router.post("/predict-lot")
def predict_lot(lot_row: dict):
    return svc.predict_lot_survival(lot_row)


@router.get("/cox-hazard-ratios")
def cox_hazard_ratios():
    return svc.cox_hazard_ratios()


@router.get("/logrank-test")
def logrank_test():
    return svc.logrank_test_results()


@router.get("/median-shelflife")
def median_shelflife():
    return svc.median_shelflife()
