"""
Clinical report generation from ML predictions, annotations, and review data.

Builds a structured report using template sections populated with real data
from the database. No external API keys required — all logic is local.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Annotation as AnnotationModel,
    Image as ImageModel,
    Prediction as PredictionModel,
    ReviewQueue as ReviewQueueModel,
    get_db,
)
from app.schemas.schemas import ReportOut

logger = logging.getLogger(__name__)

router = APIRouter()

# Clinical reference data for the 7 HAM10000 conditions.
# Each entry: (risk_level, icd10_code, description, follow_up)
CLINICAL_REFERENCE: dict[str, dict] = {
    "Melanocytic Nevi": {
        "risk": "Low",
        "icd10": "D22.9",
        "description": (
            "Melanocytic nevi (moles) are benign proliferations of melanocytes. "
            "They are extremely common and typically appear as small, evenly colored "
            "brown or tan spots with well-defined borders."
        ),
        "follow_up": (
            "Routine monitoring recommended. Advise patient to watch for ABCDE "
            "changes (Asymmetry, Border irregularity, Color variation, Diameter >6mm, "
            "Evolution). Dermoscopic follow-up at next annual exam."
        ),
    },
    "Melanoma": {
        "risk": "High",
        "icd10": "C43.9",
        "description": (
            "Melanoma is a malignant neoplasm arising from melanocytes. It is the most "
            "dangerous form of skin cancer with potential for metastasis. Early detection "
            "is critical for favorable prognosis."
        ),
        "follow_up": (
            "URGENT: Refer to dermatology/oncology for biopsy and staging. Full-body "
            "skin examination recommended. Consider sentinel lymph node biopsy if "
            "confirmed. Breslow thickness measurement required for staging."
        ),
    },
    "Benign Keratosis": {
        "risk": "Low",
        "icd10": "L82.1",
        "description": (
            "Benign keratoses include seborrheic keratoses and solar lentigines. "
            "These are non-cancerous growths that appear as waxy, scaly, or slightly "
            "elevated lesions, most common in older adults."
        ),
        "follow_up": (
            "No treatment necessary unless symptomatic or cosmetically concerning. "
            "Cryotherapy or curettage available if removal desired. Reassess if "
            "lesion changes in appearance."
        ),
    },
    "Basal Cell Carcinoma": {
        "risk": "Moderate",
        "icd10": "C44.91",
        "description": (
            "Basal cell carcinoma (BCC) is the most common form of skin cancer. "
            "It typically presents as a pearly or waxy bump, often with visible blood "
            "vessels. While rarely metastatic, it can cause significant local tissue damage."
        ),
        "follow_up": (
            "Refer to dermatology for biopsy confirmation. Treatment options include "
            "Mohs micrographic surgery, excision, or topical therapy depending on size "
            "and location. Follow-up every 6 months for 2 years post-treatment."
        ),
    },
    "Actinic Keratosis": {
        "risk": "Moderate",
        "icd10": "L57.0",
        "description": (
            "Actinic keratoses are rough, scaly patches caused by prolonged UV exposure. "
            "They are considered pre-malignant lesions with a small but real risk of "
            "progressing to squamous cell carcinoma (estimated 5-10% over 10 years)."
        ),
        "follow_up": (
            "Treatment recommended to prevent potential SCC progression. Options include "
            "cryotherapy, topical 5-fluorouracil, imiquimod, or photodynamic therapy. "
            "Counsel on sun protection. Re-evaluate in 3 months."
        ),
    },
    "Vascular Lesion": {
        "risk": "Low",
        "icd10": "D18.01",
        "description": (
            "Vascular lesions include hemangiomas, angiomas, and pyogenic granulomas. "
            "These are benign proliferations of blood vessels that may appear as red, "
            "purple, or blue spots or raised nodules."
        ),
        "follow_up": (
            "Typically benign — observation is usually sufficient. Consider treatment "
            "if bleeding, rapid growth, or cosmetic concern. Laser therapy or excision "
            "available. Biopsy if diagnosis uncertain."
        ),
    },
    "Dermatofibroma": {
        "risk": "Low",
        "icd10": "D23.9",
        "description": (
            "Dermatofibromas are common benign fibrous nodules of the skin. They are "
            "firm to palpation, usually less than 1cm, and often appear on the lower "
            "extremities. The 'dimple sign' (central depression on lateral compression) "
            "is a characteristic finding."
        ),
        "follow_up": (
            "No treatment necessary. Reassure patient of benign nature. Excision "
            "available if symptomatic or cosmetically bothersome, though recurrence "
            "is possible. Re-evaluate if rapid growth or color change."
        ),
    },
}

RISK_PRIORITY = {"High": 0, "Moderate": 1, "Low": 2}


def _confidence_assessment(confidence: float) -> str:
    """Translate a raw confidence score into a clinical interpretation."""
    if confidence >= 0.85:
        return "very high confidence — model strongly favors this diagnosis"
    if confidence >= 0.70:
        return "high confidence — diagnosis is well-supported by learned features"
    if confidence >= 0.50:
        return "moderate confidence — consider differential diagnoses"
    if confidence >= 0.30:
        return "low confidence — multiple diagnoses plausible, clinical correlation needed"
    return "very low confidence — model is uncertain, recommend manual assessment"


def _uncertainty_summary(score: float, margin: float, entropy: float) -> str:
    """Build a human-readable uncertainty interpretation."""
    parts = []

    if score >= 0.6:
        parts.append(
            "The model exhibits HIGH uncertainty for this image. "
            "Expert review is strongly recommended before acting on these results."
        )
    elif score >= 0.35:
        parts.append(
            "The model shows MODERATE uncertainty. The prediction may be reliable "
            "but clinical correlation is advised."
        )
    else:
        parts.append(
            "The model has LOW uncertainty, suggesting a confident prediction. "
            "Results are likely reliable for this image."
        )

    if margin < 0.15:
        parts.append(
            f"The margin between top predictions is narrow ({margin:.1%}), "
            "indicating the model found multiple plausible diagnoses."
        )

    if entropy > 0.5:
        parts.append(
            f"Shannon entropy is elevated ({entropy:.3f}), reflecting spread "
            "across several diagnostic categories."
        )

    return " ".join(parts)


def _build_report(
    image: ImageModel,
    predictions: list[PredictionModel],
    annotations: list[AnnotationModel],
    review: ReviewQueueModel | None,
) -> dict:
    """Assemble the full report structure from database records."""

    now = datetime.now(timezone.utc)
    top = predictions[0] if predictions else None
    top_label = top.label if top else "N/A"
    top_conf = top.confidence if top else 0.0
    ref = CLINICAL_REFERENCE.get(top_label, {})

    # Header section
    sections = []

    sections.append({
        "title": "Image Information",
        "content": (
            f"File: {image.original_name}\n"
            f"Dimensions: {image.width} x {image.height} px\n"
            f"Uploaded: {image.uploaded_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Current Status: {image.status.upper()}"
        ),
    })

    # Classification results
    pred_lines = []
    for p in predictions:
        marker = " <<<" if p.rank == 1 else ""
        pred_lines.append(
            f"  #{p.rank}  {p.label}  —  {p.confidence:.1%} confidence{marker}"
        )

    assessment = _confidence_assessment(top_conf) if top else "No predictions available."
    pred_block = "\n".join(pred_lines) if pred_lines else "No classification has been performed."

    sections.append({
        "title": "ML Classification Results",
        "content": (
            f"Model: {top.model_name if top else 'N/A'}\n"
            f"Primary Diagnosis: {top_label}\n"
            f"Confidence: {top_conf:.1%} ({assessment})\n\n"
            f"Top Predictions:\n{pred_block}"
        ),
    })

    # Clinical reference for the top prediction
    if ref:
        risk = ref.get("risk", "Unknown")
        sections.append({
            "title": "Clinical Reference",
            "content": (
                f"Condition: {top_label}\n"
                f"ICD-10 Code: {ref.get('icd10', 'N/A')}\n"
                f"Risk Level: {risk}\n\n"
                f"Description:\n{ref.get('description', '')}\n\n"
                f"Recommended Follow-Up:\n{ref.get('follow_up', '')}"
            ),
        })

    # Uncertainty analysis (if we have review queue data)
    if review:
        unc_text = _uncertainty_summary(
            review.uncertainty_score, review.margin, review.entropy,
        )
        sections.append({
            "title": "Model Uncertainty Analysis",
            "content": (
                f"Uncertainty Score: {review.uncertainty_score:.3f}\n"
                f"Max Confidence: {review.max_confidence:.1%}\n"
                f"Prediction Margin: {review.margin:.1%}\n"
                f"Shannon Entropy: {review.entropy:.4f}\n\n"
                f"Interpretation:\n{unc_text}"
            ),
        })

    # Expert review status
    if review and review.review_status != "pending":
        review_lines = [
            f"Status: {review.review_status.upper()}",
            f"Reviewer: {review.reviewer or 'N/A'}",
        ]
        if review.corrected_label:
            review_lines.append(f"Corrected Diagnosis: {review.corrected_label}")
        if review.review_notes:
            review_lines.append(f"Notes: {review.review_notes}")
        if review.reviewed_at:
            review_lines.append(
                f"Reviewed: {review.reviewed_at.strftime('%Y-%m-%d %H:%M UTC')}"
            )
        sections.append({
            "title": "Expert Review",
            "content": "\n".join(review_lines),
        })

    # Annotations summary
    if annotations:
        ann_lines = []
        for ann in annotations:
            geo_info = ""
            if ann.geometry and ann.geometry.get("points"):
                n_points = len(ann.geometry["points"])
                geo_info = f"  ({n_points}-point polygon)"

            ann_lines.append(
                f"  - {ann.label} [{ann.annotation_type}] by {ann.annotator}{geo_info}"
                + (f"  Note: {ann.notes}" if ann.notes else "")
            )

        sections.append({
            "title": "Annotations",
            "content": (
                f"Total annotations: {len(annotations)}\n\n"
                + "\n".join(ann_lines)
            ),
        })

    # Differential diagnosis (suggest second/third predictions as differentials)
    if len(predictions) > 1:
        diff_lines = []
        for p in predictions[1:]:
            p_ref = CLINICAL_REFERENCE.get(p.label, {})
            risk = p_ref.get("risk", "Unknown")
            diff_lines.append(
                f"  - {p.label} ({p.confidence:.1%}) — Risk: {risk}"
            )
        sections.append({
            "title": "Differential Diagnoses",
            "content": (
                "The following conditions were also considered by the model:\n\n"
                + "\n".join(diff_lines)
                + "\n\nClinical correlation is recommended to rule out these alternatives."
            ),
        })

    # Disclaimer
    sections.append({
        "title": "Disclaimer",
        "content": (
            "This report is based on predictions from a ViT-Base-16 classifier "
            "trained on the HAM10000 dermatoscopic image dataset. Results are intended "
            "to assist clinical decision-making and should NOT be used as a sole basis "
            "for diagnosis or treatment. All findings should be confirmed by a qualified "
            "dermatologist through clinical examination and, where appropriate, histopathological "
            "analysis."
        ),
    })

    # Determine overall risk level from the top prediction
    risk_level = ref.get("risk", "Unknown") if ref else "Unknown"

    return {
        "image_id": image.id,
        "image_name": image.original_name,
        "generated_at": now.isoformat(),
        "primary_diagnosis": top_label,
        "confidence": round(top_conf, 4),
        "risk_level": risk_level,
        "model_name": top.model_name if top else "N/A",
        "sections": sections,
    }


@router.post("/{image_id}", response_model=ReportOut)
async def generate_report(
    image_id: int,
    db: AsyncSession = Depends(get_db),
) -> ReportOut:
    """Generate a clinical report for the given image."""

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

    if not predictions:
        raise HTTPException(
            status_code=400,
            detail="No predictions found — classify the image first.",
        )

    annotations = (
        await db.execute(
            select(AnnotationModel).where(AnnotationModel.image_id == image_id)
        )
    ).scalars().all()

    review = (
        await db.execute(
            select(ReviewQueueModel).where(ReviewQueueModel.image_id == image_id)
        )
    ).scalar_one_or_none()

    report = _build_report(image, list(predictions), list(annotations), review)
    return ReportOut(**report)
