from __future__ import annotations

import os
import uuid
from io import BytesIO

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image as PILImage, UnidentifiedImageError
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Annotation as AnnotationModel
from app.models.database import Image as ImageModel
from app.models.database import Prediction as PredictionModel
from app.models.database import get_db
from app.schemas.schemas import ImageDetailOut, ImageListOut, ImageOut

router = APIRouter()

UPLOAD_DIR = "uploads"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/upload", response_model=ImageOut)
async def upload_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> ImageOut:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG files are allowed.")

    content = await file.read()

    # --- File size guard ---
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # --- Content validation: confirm the bytes decode as a valid image ---
    # Done before writing to disk so we never persist a corrupt file.
    try:
        img = PILImage.open(BytesIO(content))
        img.load()  # force full decode — catches truncated / corrupt files
        width, height = img.size
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot decode image content: {exc}",
        )

    ext = ".jpg" if file.content_type == "image/jpeg" else ".png"
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    db_image = ImageModel(
        filename=filename,
        original_name=file.filename or filename,
        file_path=file_path,
        width=width,
        height=height,
    )
    db.add(db_image)
    await db.commit()
    await db.refresh(db_image)

    return db_image  # type: ignore[return-value]


@router.get("/", response_model=list[ImageListOut])
async def list_images(
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[ImageListOut]:
    # Single LEFT JOIN — avoids one extra query per image (N+1 problem)
    query = (
        select(ImageModel, PredictionModel)
        .outerjoin(
            PredictionModel,
            and_(
                PredictionModel.image_id == ImageModel.id,
                PredictionModel.rank == 1,
            ),
        )
        .order_by(ImageModel.uploaded_at.desc())
    )
    if status:
        query = query.where(ImageModel.status == status)

    rows = (await db.execute(query)).all()

    return [
        ImageListOut(
            id=image.id,
            filename=image.filename,
            original_name=image.original_name,
            status=image.status,
            top_label=pred.label if pred else None,
            top_confidence=pred.confidence if pred else None,
            uploaded_at=image.uploaded_at,
        )
        for image, pred in rows
    ]


@router.get("/{image_id}", response_model=ImageDetailOut)
async def get_image(image_id: int, db: AsyncSession = Depends(get_db)) -> ImageDetailOut:
    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    predictions = (
        await db.execute(
            select(PredictionModel)
            .where(PredictionModel.image_id == image_id)
            .order_by(PredictionModel.rank)
        )
    ).scalars().all()

    annotations = (
        await db.execute(
            select(AnnotationModel)
            .where(AnnotationModel.image_id == image_id)
            .order_by(AnnotationModel.created_at.desc())
        )
    ).scalars().all()

    return ImageDetailOut(
        id=image.id,
        filename=image.filename,
        original_name=image.original_name,
        width=image.width,
        height=image.height,
        uploaded_at=image.uploaded_at,
        status=image.status,
        predictions=list(predictions),
        annotations=list(annotations),
    )


@router.delete("/{image_id}")
async def delete_image(image_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    if os.path.exists(image.file_path):
        os.remove(image.file_path)

    await db.delete(image)
    await db.commit()

    return {"message": "Image deleted."}
