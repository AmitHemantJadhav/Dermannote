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


# ── Active Learning Queue ──


class QueueItemOut(BaseModel):
    id: int
    image_id: int
    max_confidence: float
    margin: float
    entropy: float
    uncertainty_score: float
    review_status: str
    reviewer: Optional[str] = None
    corrected_label: Optional[str] = None
    review_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    # Denormalized image info
    filename: Optional[str] = None
    original_name: Optional[str] = None
    image_status: Optional[str] = None
    top_label: Optional[str] = None
    top_confidence: Optional[float] = None

    model_config = {"from_attributes": True}


class ReviewCreate(BaseModel):
    status: str  # confirmed | corrected | skipped
    corrected_label: Optional[str] = None
    review_notes: Optional[str] = None
    reviewer: str = "expert"


class QueueStatsOut(BaseModel):
    total: int = 0
    pending: int = 0
    confirmed: int = 0
    corrected: int = 0
    skipped: int = 0
    avg_uncertainty: float = 0.0


class ExportCreate(BaseModel):
    format: str = "coco"
    image_ids: Optional[list[int]] = None


class ExportOut(BaseModel):
    download_url: str
    num_images: int
    num_annotations: int
    filename: str


# ── Clinical Report ──


# ── Batch Processing ──


class BatchItemResult(BaseModel):
    image_id: int
    filename: str
    success: bool
    top_label: Optional[str] = None
    top_confidence: Optional[float] = None
    error: Optional[str] = None


class BatchOut(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchItemResult]


# ── Embedding Visualization ──


class EmbeddingPoint(BaseModel):
    image_id: int
    x: float
    y: float
    label: str
    filename: str


class TSNEOut(BaseModel):
    num_images: int
    perplexity: float
    points: list[EmbeddingPoint]


# ── Clinical Report ──


class ReportSection(BaseModel):
    title: str
    content: str


class ReportOut(BaseModel):
    image_id: int
    image_name: str
    generated_at: str
    primary_diagnosis: str
    confidence: float
    risk_level: str
    model_name: str
    sections: list[ReportSection]
