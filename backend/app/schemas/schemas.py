from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ImageOut(BaseModel):
    id: int
    filename: str
    original_name: str
    width: int
    height: int
    uploaded_at: datetime
    status: str

    model_config = {"from_attributes": True}


class ImageListOut(BaseModel):
    id: int
    filename: str
    original_name: str
    status: str
    top_label: Optional[str] = None
    top_confidence: Optional[float] = None
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class PredictionOut(BaseModel):
    id: int
    image_id: int
    model_name: str
    label: str
    confidence: float
    rank: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AnnotationCreate(BaseModel):
    label: str
    annotation_type: str = "classification"
    geometry: Optional[dict] = None
    notes: Optional[str] = None
    annotator: Optional[str] = "expert"


class AnnotationOut(BaseModel):
    id: int
    image_id: int
    annotator: str
    label: str
    annotation_type: str
    geometry: Optional[dict] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ImageDetailOut(BaseModel):
    """Full image response including eager-loaded predictions and annotations."""

    id: int
    filename: str
    original_name: str
    width: int
    height: int
    uploaded_at: datetime
    status: str
    predictions: list[PredictionOut] = []
    annotations: list[AnnotationOut] = []

    model_config = {"from_attributes": True}


class ExportCreate(BaseModel):
    format: str = "coco"
    image_ids: Optional[list[int]] = None


class ExportOut(BaseModel):
    download_url: str
    num_images: int
    num_annotations: int
    filename: str
