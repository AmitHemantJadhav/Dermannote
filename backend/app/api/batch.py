"""
Batch classification endpoint — classify multiple images in one request.

Processes images sequentially through the single-worker inference pool
to avoid GPU contention. Returns per-image results and aggregate stats.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Image as ImageModel,
    Prediction as PredictionModel,
    ReviewQueue as ReviewQueueModel,
    get_db,
)
from app.schemas.schemas import BatchItemResult, BatchOut
from app.services.uncertainty import compute_uncertainty

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

logger = logging.getLogger(__name__)

router = APIRouter()


class BatchRequest(BaseModel):
    image_ids: Optional[list[int]] = None  # None = classify all pending


@router.post("/classify", response_model=BatchOut)
async def batch_classify(
    body: BatchRequest = BatchRequest(),
    db: AsyncSession = Depends(get_db),
) -> BatchOut:
    """
    Classify multiple images in a single request.

    If image_ids is omitted, classifies all images with status 'pending'.
    """
    from app.api.predictions import _get_classifier, _executor

    if body.image_ids:
        rows = (
            await db.execute(
                select(ImageModel).where(ImageModel.id.in_(body.image_ids))
            )
        ).scalars().all()
    else:
        rows = (
            await db.execute(
                select(ImageModel).where(ImageModel.status == "pending")
            )
        ).scalars().all()

    if not rows:
        return BatchOut(total=0, succeeded=0, failed=0, results=[])

    classifier = _get_classifier()
    loop = asyncio.get_event_loop()
    results: list[BatchItemResult] = []

    for image in rows:
        try:
            raw_preds = await loop.run_in_executor(
                _executor, classifier.predict, image.file_path,
            )

            # Wipe old predictions
            await db.execute(
                delete(PredictionModel).where(PredictionModel.image_id == image.id)
            )

            for pred in raw_preds:
                db.add(PredictionModel(
                    image_id=image.id,
                    model_name=pred["model_name"],
                    label=pred["label"],
                    confidence=pred["confidence"],
                    rank=pred["rank"],
                ))

            image.status = "predicted"

            # Enqueue for active learning review
            uncertainty = compute_uncertainty(raw_preds)
            existing = (
                await db.execute(
                    select(ReviewQueueModel).where(
                        ReviewQueueModel.image_id == image.id
                    )
                )
            ).scalar_one_or_none()
            if existing:
                existing.max_confidence = uncertainty["max_confidence"]
                existing.margin = uncertainty["margin"]
                existing.entropy = uncertainty["entropy"]
                existing.uncertainty_score = uncertainty["uncertainty_score"]
                existing.review_status = "pending"
            else:
                db.add(ReviewQueueModel(
                    image_id=image.id,
                    max_confidence=uncertainty["max_confidence"],
                    margin=uncertainty["margin"],
                    entropy=uncertainty["entropy"],
                    uncertainty_score=uncertainty["uncertainty_score"],
                ))

            top = raw_preds[0] if raw_preds else None
            results.append(BatchItemResult(
                image_id=image.id,
                filename=image.original_name,
                success=True,
                top_label=top["label"] if top else None,
                top_confidence=top["confidence"] if top else None,
            ))

        except Exception as exc:
            logger.warning("Batch classify failed for image %d: %s", image.id, exc)
            results.append(BatchItemResult(
                image_id=image.id,
                filename=image.original_name,
                success=False,
                error=str(exc),
            ))

    await db.commit()

    succeeded = sum(1 for r in results if r.success)
    return BatchOut(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )
