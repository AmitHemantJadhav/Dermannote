"""Active Learning Queue router — surfaces uncertain predictions for expert review."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Image as ImageModel
from app.models.database import Prediction as PredictionModel
from app.models.database import ReviewQueue as ReviewQueueModel
from app.models.database import get_db
from app.schemas.schemas import QueueItemOut, QueueStatsOut, ReviewCreate

router = APIRouter()


@router.get("/", response_model=list[QueueItemOut])
async def list_queue(
    status: Optional[str] = Query(None, description="Filter by review_status"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[QueueItemOut]:
    """List queue items sorted by uncertainty (most uncertain first)."""
    query = select(ReviewQueueModel)
    if status:
        query = query.where(ReviewQueueModel.review_status == status)
    query = query.order_by(ReviewQueueModel.uncertainty_score.desc()).limit(limit)

    rows = (await db.execute(query)).scalars().all()

    result: list[QueueItemOut] = []
    for row in rows:
        # Fetch denormalized image + top prediction info
        image = (
            await db.execute(select(ImageModel).where(ImageModel.id == row.image_id))
        ).scalar_one_or_none()

        top_pred = (
            await db.execute(
                select(PredictionModel)
                .where(PredictionModel.image_id == row.image_id)
                .order_by(PredictionModel.rank)
                .limit(1)
            )
        ).scalar_one_or_none()

        item = QueueItemOut(
            id=row.id,
            image_id=row.image_id,
            max_confidence=row.max_confidence,
            margin=row.margin,
            entropy=row.entropy,
            uncertainty_score=row.uncertainty_score,
            review_status=row.review_status,
            reviewer=row.reviewer,
            corrected_label=row.corrected_label,
            review_notes=row.review_notes,
            reviewed_at=row.reviewed_at,
            created_at=row.created_at,
            filename=image.filename if image else None,
            original_name=image.original_name if image else None,
            image_status=image.status if image else None,
            top_label=top_pred.label if top_pred else None,
            top_confidence=top_pred.confidence if top_pred else None,
        )
        result.append(item)

    return result


@router.get("/stats", response_model=QueueStatsOut)
async def queue_stats(
    db: AsyncSession = Depends(get_db),
) -> QueueStatsOut:
    """Summary counts per review status + average uncertainty."""
    rows = (await db.execute(select(ReviewQueueModel))).scalars().all()

    if not rows:
        return QueueStatsOut()

    counts = {"pending": 0, "confirmed": 0, "corrected": 0, "skipped": 0}
    total_uncertainty = 0.0
    for r in rows:
        counts[r.review_status] = counts.get(r.review_status, 0) + 1
        total_uncertainty += r.uncertainty_score

    return QueueStatsOut(
        total=len(rows),
        pending=counts.get("pending", 0),
        confirmed=counts.get("confirmed", 0),
        corrected=counts.get("corrected", 0),
        skipped=counts.get("skipped", 0),
        avg_uncertainty=round(total_uncertainty / len(rows), 4) if rows else 0.0,
    )


@router.post("/{image_id}/review", response_model=QueueItemOut)
async def review_image(
    image_id: int,
    body: ReviewCreate,
    db: AsyncSession = Depends(get_db),
) -> QueueItemOut:
    """Submit a review decision for a queued image."""
    row = (
        await db.execute(
            select(ReviewQueueModel).where(ReviewQueueModel.image_id == image_id)
        )
    ).scalar_one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Image not in review queue.")

    if body.status not in ("confirmed", "corrected", "skipped"):
        raise HTTPException(
            status_code=400,
            detail="status must be confirmed, corrected, or skipped.",
        )

    if body.status == "corrected" and not body.corrected_label:
        raise HTTPException(
            status_code=400,
            detail="corrected_label is required when status is 'corrected'.",
        )

    row.review_status = body.status
    row.reviewer = body.reviewer
    row.corrected_label = body.corrected_label
    row.review_notes = body.review_notes
    row.reviewed_at = datetime.now(timezone.utc)

    # Update image status to "reviewed"
    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()
    if image:
        image.status = "reviewed"

    await db.commit()
    await db.refresh(row)

    # Fetch denormalized info for response
    top_pred = (
        await db.execute(
            select(PredictionModel)
            .where(PredictionModel.image_id == image_id)
            .order_by(PredictionModel.rank)
            .limit(1)
        )
    ).scalar_one_or_none()

    return QueueItemOut(
        id=row.id,
        image_id=row.image_id,
        max_confidence=row.max_confidence,
        margin=row.margin,
        entropy=row.entropy,
        uncertainty_score=row.uncertainty_score,
        review_status=row.review_status,
        reviewer=row.reviewer,
        corrected_label=row.corrected_label,
        review_notes=row.review_notes,
        reviewed_at=row.reviewed_at,
        created_at=row.created_at,
        filename=image.filename if image else None,
        original_name=image.original_name if image else None,
        image_status=image.status if image else None,
        top_label=top_pred.label if top_pred else None,
        top_confidence=top_pred.confidence if top_pred else None,
    )
