"""
Attention head visualization — extract and render ViT self-attention patterns.

Produces heatmap overlays showing which image regions each attention head
focuses on at each transformer layer. Cached as PNG files.
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Image as ImageModel, get_db

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

logger = logging.getLogger(__name__)

router = APIRouter()


class AttentionRequest(BaseModel):
    layer: int = 11
    head: int | None = None  # None = average across all heads


def _generate_attention_png(file_path: str, output_path: str, layer: int, head: int | None) -> bool:
    """Generate attention heatmap overlay and save as PNG. Returns True on success."""
    from ml.classifier import save_gradcam_png

    from app.api.predictions import _get_classifier

    classifier = _get_classifier()
    vis = classifier.extract_attention(file_path, layer=layer, head=head)
    if vis is None:
        return False
    save_gradcam_png(vis, output_path)
    return True


@router.post("/{image_id}")
async def generate_attention_map(
    image_id: int,
    body: AttentionRequest = AttentionRequest(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Generate (or return cached) attention head visualization for an image."""
    from app.api.predictions import _executor

    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    head_str = "avg" if body.head is None else str(body.head)
    filename = f"attn_{image_id}_L{body.layer}_H{head_str}.png"
    output_path = os.path.join(UPLOAD_DIR, filename)

    # Return cached if available
    if os.path.exists(output_path):
        return {
            "attention_url": f"/uploads/{filename}",
            "layer": body.layer,
            "head": body.head,
        }

    loop = asyncio.get_event_loop()

    try:
        success = await loop.run_in_executor(
            _executor,
            _generate_attention_png,
            image.file_path,
            output_path,
            body.layer,
            body.head,
        )
    except Exception as exc:
        logger.exception("Attention map generation failed for image %d", image_id)
        raise HTTPException(status_code=500, detail=f"Attention map error: {exc}")

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Attention visualization is not available (mock mode or unsupported architecture).",
        )

    return {
        "attention_url": f"/uploads/{filename}",
        "layer": body.layer,
        "head": body.head,
    }


@router.get("/{image_id}/info")
async def get_attention_info(
    image_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return model config info needed by the frontend (number of layers and heads)."""
    image = (
        await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    ).scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    from app.api.predictions import _get_classifier

    classifier = _get_classifier()
    if hasattr(classifier, "model") and hasattr(classifier.model, "config"):
        config = classifier.model.config
        return {
            "num_layers": getattr(config, "num_hidden_layers", 12),
            "num_heads": getattr(config, "num_attention_heads", 12),
            "architecture": getattr(config, "model_type", "unknown"),
        }

    return {"num_layers": 0, "num_heads": 0, "architecture": "mock"}
