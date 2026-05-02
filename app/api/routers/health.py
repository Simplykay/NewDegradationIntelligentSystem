from fastapi import APIRouter
from ..services.model_service import models_loaded
from ..services.data_loader import get_df

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    try:
        df_loaded = len(get_df()) > 0
    except Exception:
        df_loaded = False
    return {
        "status":        "healthy" if df_loaded and models_loaded() else "degraded",
        "data_loaded":   df_loaded,
        "models_loaded": models_loaded(),
    }
