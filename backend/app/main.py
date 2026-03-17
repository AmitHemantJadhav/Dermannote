from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

from app.api import annotations, export, images, predictions
from app.models.database import init_db

UPLOAD_DIR = "uploads"
EXPORT_DIR = "exports"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    await init_db()

    # Pre-warm the real classifier at startup so the first request isn't slow.
    # Skipped in mock mode because MockClassifier.__init__ is instant.
    use_mock = os.getenv("USE_MOCK", "true").strip().lower() == "true"
    if not use_mock:
        from app.api.predictions import load_classifier
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_classifier)

    yield


app = FastAPI(
    title="DermAnnotate API",
    description="ML-assisted annotation for autoimmune skin conditions",
    version="1.0.0",
    lifespan=lifespan,
)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.include_router(images.router, prefix="/api/images", tags=["images"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(annotations.router, prefix="/api/annotations", tags=["annotations"])
app.include_router(export.router, prefix="/api/export", tags=["export"])


@app.get("/")
async def health() -> dict:
    return {"status": "ok", "service": "DermAnnotate API"}
