"""
Segmentation router — 1-click SAM auto-segmentation.

Accepts a click point in display coordinates, runs MobileSAM inference,
and returns polygon points in display coordinates (ready for Fabric.js).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Image as ImageModel
from app.models.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# -------------------------------------------------------------------
# SAM singleton — thread-safe, lazy-loaded (same pattern as predictions.py)
# -------------------------------------------------------------------

_sam = None
_sam_lock = threading.Lock()
_sam_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam_worker")


def _get_sam():
    global _sam
    if _sam is None:
        with _sam_lock:
            if _sam is None:
                from ml.sam_model import create_sam_model
                logger.info("Initialising MobileSAM…")
                _sam = create_sam_model()
                logger.info("MobileSAM ready.")
    return _sam


# -------------------------------------------------------------------
# Request / response schemas
# -------------------------------------------------------------------

class SegmentRequest(BaseModel):
    x: float
    y: float
    display_width: int
    display_height: int


class PointOut(BaseModel):
    x: float
    y: float


class SegmentResponse(BaseModel):
    points: list[PointOut]


# -------------------------------------------------------------------
# Route
# -------------------------------------------------------------------

@router.post("/{image_id}", response_model=SegmentResponse)
async def segment_image(
    image_id: int,
    body: SegmentRequest,
    db: AsyncSession = Depends(get_db),
) -> SegmentResponse:
    """Accept a click point (display coords), run SAM, return polygon."""
    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    if not os.path.exists(image.file_path):
        raise HTTPException(status_code=404, detail="Image file missing from disk.")

    # Convert display coords → original image coords
    scale_x = image.width / body.display_width
    scale_y = image.height / body.display_height
    img_x = body.x * scale_x
    img_y = body.y * scale_y

    def _run():
        sam = _get_sam()
        return sam.segment(
            image_path=image.file_path,
            point_x=img_x,
            point_y=img_y,
            img_width=image.width,
            img_height=image.height,
        )

    try:
        loop = asyncio.get_event_loop()
        raw_points = await loop.run_in_executor(_sam_executor, _run)
    except Exception as exc:
        logger.exception("SAM segmentation failed for image_id=%s", image_id)
        raise HTTPException(status_code=500, detail=f"Segmentation error: {exc}")

    if not raw_points:
        raise HTTPException(status_code=422, detail="No contour found at click point.")

    # Convert image coords → display coords
    inv_scale_x = body.display_width / image.width
    inv_scale_y = body.display_height / image.height
    display_points = [
        PointOut(x=round(pt["x"] * inv_scale_x, 1), y=round(pt["y"] * inv_scale_y, 1))
        for pt in raw_points
    ]

    return SegmentResponse(points=display_points)
