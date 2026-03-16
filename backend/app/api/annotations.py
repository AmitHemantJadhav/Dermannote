from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Annotation as AnnotationModel
from app.models.database import Image as ImageModel
from app.models.database import get_db
from app.schemas.schemas import AnnotationCreate, AnnotationOut

router = APIRouter()


@router.post("/{image_id}", response_model=AnnotationOut)
async def create_annotation(
    image_id: int,
    data: AnnotationCreate,
    db: AsyncSession = Depends(get_db),
) -> AnnotationOut:
    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    annotation = AnnotationModel(
        image_id=image_id,
        label=data.label,
        annotation_type=data.annotation_type,
        geometry=data.geometry,
        notes=data.notes,
        annotator=data.annotator or "expert",
    )
    db.add(annotation)

    image.status = "annotated"
    await db.commit()
    await db.refresh(annotation)

    return annotation  # type: ignore[return-value]


@router.get("/{image_id}", response_model=list[AnnotationOut])
async def list_annotations(
    image_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[AnnotationOut]:
    annotations = (
        await db.execute(
            select(AnnotationModel)
            .where(AnnotationModel.image_id == image_id)
            .order_by(AnnotationModel.created_at.desc())
        )
    ).scalars().all()

    return annotations  # type: ignore[return-value]


@router.delete("/{annotation_id}")
async def delete_annotation(
    annotation_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    annotation = (
        await db.execute(
            select(AnnotationModel).where(AnnotationModel.id == annotation_id)
        )
    ).scalar_one_or_none()

    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found.")

    await db.delete(annotation)
    await db.commit()

    return {"message": "Annotation deleted."}
