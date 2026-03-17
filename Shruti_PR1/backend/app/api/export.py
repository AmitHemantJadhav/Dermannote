from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import aiofiles
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Annotation as AnnotationModel
from app.models.database import Image as ImageModel
from app.models.database import get_db
from app.schemas.schemas import ExportCreate, ExportOut

router = APIRouter()

EXPORT_DIR = "exports"


@router.post("/", response_model=ExportOut)
async def export_dataset(
    data: ExportCreate,
    db: AsyncSession = Depends(get_db),
) -> ExportOut:
    query = select(ImageModel)
    if data.image_ids:
        query = query.where(ImageModel.id.in_(data.image_ids))
    images = (await db.execute(query)).scalars().all()

    image_ids = [img.id for img in images]
    annotations = (
        await db.execute(
            select(AnnotationModel).where(AnnotationModel.image_id.in_(image_ids))
        )
    ).scalars().all()

    # Build COCO category list from unique labels used in annotations
    all_labels = sorted({ann.label for ann in annotations})
    categories = [
        {"id": i + 1, "name": label, "supercategory": "skin_condition"}
        for i, label in enumerate(all_labels)
    ]
    label_to_cat_id = {cat["name"]: cat["id"] for cat in categories}

    coco_images = [
        {
            "id": img.id,
            "file_name": img.original_name,
            "width": img.width,
            "height": img.height,
        }
        for img in images
    ]

    coco_annotations = []
    for i, ann in enumerate(annotations):
        segmentation: list[list[float]] = []
        bbox: list[float] = [0.0, 0.0, 0.0, 0.0]
        area: float = 0.0

        if ann.annotation_type == "segmentation" and ann.geometry:
            points = ann.geometry.get("points", [])
            if points:
                flat = [coord for p in points for coord in (p["x"], p["y"])]
                segmentation = [flat]
                xs = [p["x"] for p in points]
                ys = [p["y"] for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w, h = x_max - x_min, y_max - y_min
                bbox = [x_min, y_min, w, h]
                area = w * h

        coco_annotations.append(
            {
                "id": i + 1,
                "image_id": ann.image_id,
                "category_id": label_to_cat_id.get(ann.label, 0),
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
            }
        )

    coco_export = {
        "info": {
            "description": "DermAnnotate Export — Autoimmune Skin Conditions",
            "version": "1.0",
            "contributor": "DermAnnotate Team — PFW",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    filename = f"dermannotate_export_{uuid.uuid4().hex[:8]}.json"
    filepath = os.path.join(EXPORT_DIR, filename)

    # Use aiofiles so the JSON serialisation + disk write doesn't block the event loop
    async with aiofiles.open(filepath, "w") as f:
        await f.write(json.dumps(coco_export, indent=2))

    return ExportOut(
        download_url=f"/api/export/download/{filename}",
        num_images=len(images),
        num_annotations=len(annotations),
        filename=filename,
    )


@router.get("/download/{filename}")
async def download_export(filename: str) -> FileResponse:
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Export file not found.")

    return FileResponse(filepath, filename=filename, media_type="application/json")
