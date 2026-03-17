"""
Predictions router — wraps the ML pipeline.

Key design decisions:
  - Classifier loaded lazily behind a threading.Lock (thread-safe double-checked locking).
  - Inference runs in a dedicated ThreadPoolExecutor (max_workers=1) so it never blocks
    the asyncio event loop, even during long PyTorch forward passes.
  - Errors from the classifier surface as structured HTTP responses instead of raw 500s.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Image as ImageModel
from app.models.database import Prediction as PredictionModel
from app.models.database import get_db
from app.schemas.schemas import PredictionOut

logger = logging.getLogger(__name__)

router = APIRouter()

# -------------------------------------------------------------------
# Classifier singleton — thread-safe, lazy-loaded
# -------------------------------------------------------------------

_classifier = None
_classifier_lock = threading.Lock()

# Single-worker pool keeps GPU memory usage predictable and prevents
# concurrent inference (which would OOM on small GPUs).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml_worker")


def _get_classifier():
    """Double-checked locking: safe under concurrent first requests."""
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                from ml.classifier import create_classifier
                logger.info("Initialising ML classifier…")
                _classifier = create_classifier()
                logger.info("ML classifier ready: %s", type(_classifier).__name__)
    return _classifier


def load_classifier() -> None:
    """
    Eagerly initialise the classifier (blocking).
    Call from lifespan startup when USE_MOCK=false to avoid a slow first request.
    """
    _get_classifier()


async def _run_inference(file_path: str) -> list[dict]:
    """Run classifier.predict() in the thread-pool so the event loop stays free."""
    classifier = _get_classifier()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, classifier.predict, file_path)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@router.post("/{image_id}/classify", response_model=list[PredictionOut])
async def classify_image(
    image_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[PredictionOut]:
    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    # Wipe previous predictions so a re-classify always returns fresh results
    await db.execute(delete(PredictionModel).where(PredictionModel.image_id == image_id))

    # Run inference off the event loop
    try:
        raw_predictions = await _run_inference(image.file_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Image file missing from disk: {exc}",
        )
    except ValueError as exc:
        # Corrupt or unreadable image
        raise HTTPException(
            status_code=422,
            detail=f"Image processing error: {exc}",
        )
    except Exception as exc:
        logger.exception("Inference failed for image_id=%s", image_id)
        raise HTTPException(
            status_code=500,
            detail=f"Classifier error: {exc}",
        )

    db_predictions: list[PredictionModel] = []
    for pred in raw_predictions:
        db_pred = PredictionModel(
            image_id=image_id,
            model_name=pred["model_name"],
            label=pred["label"],
            confidence=pred["confidence"],
            rank=pred["rank"],
        )
        db.add(db_pred)
        db_predictions.append(db_pred)

    image.status = "predicted"
    await db.commit()

    for pred in db_predictions:
        await db.refresh(pred)

    return db_predictions  # type: ignore[return-value]


@router.get("/{image_id}", response_model=list[PredictionOut])
async def get_predictions(
    image_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[PredictionOut]:
    predictions = (
        await db.execute(
            select(PredictionModel)
            .where(PredictionModel.image_id == image_id)
            .order_by(PredictionModel.rank)
        )
    ).scalars().all()

    return predictions  # type: ignore[return-value]
