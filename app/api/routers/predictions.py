import io
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse
import pandas as pd

from ..schemas import LotInput, SinglePredictionResponse
from ..services.model_service import predict_single, predict_batch, get_model_metrics, load_models

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/single", response_model=SinglePredictionResponse)
def predict_single_lot(lot: LotInput, lot_id: Optional[str] = Query(None)):
    return predict_single(lot, lot_id=lot_id)


@router.post("/batch")
async def predict_batch_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    results = predict_batch(df)
    output  = io.StringIO()
    results.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@router.get("/m1/feature-importance")
def m1_feature_importance():
    m = load_models()
    model = m.get("m1_binary_classifier")
    if model is None:
        return []
    try:
        fi = model.feature_importances_
        from ..services.model_service import M1_FEATURES, CAT_FEATURES
        names = M1_FEATURES + [c for c in CAT_FEATURES if c not in M1_FEATURES]
        return sorted(
            [{"feature": n, "importance": round(float(v), 6)}
             for n, v in zip(names[:len(fi)], fi)],
            key=lambda x: x["importance"], reverse=True
        )
    except Exception:
        return []


@router.get("/m2/feature-importance")
def m2_feature_importance():
    m = load_models()
    model = m.get("m2_ct_regressor")
    if model is None:
        return []
    try:
        fi = model.feature_importances_
        from ..services.model_service import M1_FEATURES, CAT_FEATURES
        names = M1_FEATURES + [c for c in CAT_FEATURES if c not in M1_FEATURES]
        return sorted(
            [{"feature": n, "importance": round(float(v), 6)}
             for n, v in zip(names[:len(fi)], fi)],
            key=lambda x: x["importance"], reverse=True
        )
    except Exception:
        return []


@router.get("/model-metrics")
def model_metrics():
    return get_model_metrics()
