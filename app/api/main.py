"""Cotton Seed Quality Intelligence System — FastAPI backend"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, eda, predictions, survival, inventory
from .services.data_loader import get_df
from .services.model_service import load_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm caches on startup so first request is fast
    try:
        get_df()
        print("[startup] Dataset loaded and cached")
    except Exception as e:
        print(f"[startup] WARNING: data load failed — {e}")
    try:
        load_models()
        print("[startup] Models loaded and cached")
    except Exception as e:
        print(f"[startup] WARNING: model load failed — {e}")
    yield


app = FastAPI(
    title="Cotton Seed Quality Intelligence System",
    version="3.0",
    description="Degradation prediction and shelf-life analysis for cotton seed lots.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(eda.router)
app.include_router(predictions.router)
app.include_router(survival.router)
app.include_router(inventory.router)
