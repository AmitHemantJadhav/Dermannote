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
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Image as ImageModel
from app.models.database import Prediction as PredictionModel
from app.models.database import ReviewQueue as ReviewQueueModel
from app.models.database import get_db
from app.schemas.schemas import PredictionOut
from app.services.uncertainty import compute_uncertainty

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

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


def _generate_and_save_gradcam(file_path: str, output_path: str) -> bool:
    """Generate GradCAM and save PNG. Returns True on success, False if unavailable."""
    from ml.classifier import save_gradcam_png

    classifier = _get_classifier()
    vis = classifier.generate_gradcam(file_path)
    if vis is None:
        return False
    save_gradcam_png(vis, output_path)
    return True


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

    # Invalidate cached GradCAM so next request regenerates with new predictions
    gradcam_path = os.path.join(UPLOAD_DIR, f"gradcam_{image_id}.png")
    if os.path.exists(gradcam_path):
        os.remove(gradcam_path)
        logger.info("Removed stale GradCAM cache: %s", gradcam_path)

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

    # ── Auto-enqueue for active learning review ──
    uncertainty = compute_uncertainty(raw_predictions)
    existing_queue = (
        await db.execute(
            select(ReviewQueueModel).where(ReviewQueueModel.image_id == image_id)
        )
    ).scalar_one_or_none()
    if existing_queue:
        existing_queue.max_confidence = uncertainty["max_confidence"]
        existing_queue.margin = uncertainty["margin"]
        existing_queue.entropy = uncertainty["entropy"]
        existing_queue.uncertainty_score = uncertainty["uncertainty_score"]
        existing_queue.review_status = "pending"
        existing_queue.reviewed_at = None
    else:
        db.add(ReviewQueueModel(
            image_id=image_id,
            max_confidence=uncertainty["max_confidence"],
            margin=uncertainty["margin"],
            entropy=uncertainty["entropy"],
            uncertainty_score=uncertainty["uncertainty_score"],
        ))

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


@router.post("/{image_id}/gradcam")
async def generate_gradcam(
    image_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Generate (or return cached) GradCAM heatmap for a classified image."""
    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    gradcam_filename = f"gradcam_{image_id}.png"
    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)

    # Return cached if available
    if os.path.exists(gradcam_path):
        return {"gradcam_url": f"/uploads/{gradcam_filename}"}

    # Generate in thread pool
    try:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            _executor, _generate_and_save_gradcam, image.file_path, gradcam_path
        )
    except Exception as exc:
        logger.exception("GradCAM generation failed for image_id=%s", image_id)
        raise HTTPException(status_code=500, detail=f"GradCAM error: {exc}")

    if not success:
        raise HTTPException(
            status_code=400,
            detail="GradCAM is not available for the current classifier (mock mode).",
        )

    return {"gradcam_url": f"/uploads/{gradcam_filename}"}
