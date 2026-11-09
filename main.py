import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from ai.chute_model import CHUTE_MODEL
from ai.crack_criticality_model import CRITICALITY_MODEL
from ai.crack_model import CRACK_MODEL
from ai.device import get_best_device
from ai.humidity_model import SAM_MODEL
from ai.spalling_model import SPALLING_MODEL
from routes.chute_routes import router as chute_router
from routes.crack_routes import router as crack_router
from routes.humidity_routes import router as humidity_router
from routes.spalling_routes import router as spalling_router

MODELS = [CRACK_MODEL, CRITICALITY_MODEL, SAM_MODEL, CHUTE_MODEL, SPALLING_MODEL]

PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "1") == "1"


def _preload():
    for m in MODELS:
        try:
            m.get()
        except Exception as e:
            # Un modèle manquant ne bloque pas l'API : seule sa route renverra une erreur
            print(f" [startup] {m.name} non chargé : {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_best_device()
    if PRELOAD_MODELS:
        await run_in_threadpool(_preload)
    yield


app = FastAPI(title="AI Inspection Inference", lifespan=lifespan)

app.include_router(crack_router,    prefix="/crack-detection",    tags=["Crack Detection"])
app.include_router(humidity_router, prefix="/humidity-detection", tags=["Humidity Detection"])
app.include_router(chute_router,    prefix="/chute-detection",    tags=["Chute Detection"])
app.include_router(spalling_router, prefix="/spalling-detection", tags=["Spalling Detection"])


@app.get("/")
def home():
    return {"message": "AI API running"}


@app.get("/health")
def health():
    return {
        "device": get_best_device(),
        "models": {m.name: m.status for m in MODELS},
    }
