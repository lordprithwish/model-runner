from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers.models import router
from app.services.runner import registry


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load(settings.models_config_path)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")


@app.get("/health")
def health():
    return {"status": "ok"}
