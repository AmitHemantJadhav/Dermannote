"""
Embedding extraction and t-SNE dimensionality reduction.

Extracts CLS token embeddings from the ViT backbone for each classified image,
then projects them to 2D via t-SNE for interactive visualization.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Image as ImageModel,
    Prediction as PredictionModel,
    get_db,
)
from app.schemas.schemas import EmbeddingPoint, TSNEOut

logger = logging.getLogger(__name__)

router = APIRouter()


class TSNERequest(BaseModel):
    perplexity: float = 30.0
    image_ids: Optional[list[int]] = None  # None = all classified images


def _compute_tsne(embeddings, labels, image_ids, filenames, perplexity):
    """
    Run t-SNE on the embedding matrix. Returns list of EmbeddingPoint dicts.

    Uses scikit-learn's TSNE with Barnes-Hut approximation.
    Falls back to perplexity = n_samples - 1 when the dataset is tiny.
    """
    import numpy as np
    from sklearn.manifold import TSNE

    n = len(embeddings)
    if n < 2:
        # Can't run t-SNE with fewer than 2 points
        return [
            {
                "image_id": image_ids[0],
                "x": 0.0,
                "y": 0.0,
                "label": labels[0],
                "filename": filenames[0],
            }
        ] if n == 1 else []

    # Perplexity must be less than the number of samples
    effective_perplexity = min(perplexity, n - 1)

    embedding_matrix = np.array(embeddings, dtype=np.float32)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="pca" if n > 3 else "random",
    )
    coords = tsne.fit_transform(embedding_matrix)

    # Normalize to [-1, 1] for consistent frontend rendering
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    points = []
    for i in range(n):
        points.append({
            "image_id": image_ids[i],
            "x": float((coords[i, 0] - x_min) / x_range * 2 - 1),
            "y": float((coords[i, 1] - y_min) / y_range * 2 - 1),
            "label": labels[i],
            "filename": filenames[i],
        })

    return points


@router.post("/tsne", response_model=TSNEOut)
async def compute_tsne_embeddings(
    body: TSNERequest = TSNERequest(),
    db: AsyncSession = Depends(get_db),
) -> TSNEOut:
    """
    Extract ViT embeddings for classified images and project via t-SNE.

    Returns 2D coordinates colored by predicted label for scatter-plot rendering.
    """
    from app.api.predictions import _get_classifier, _executor

    # Fetch images that have predictions
    if body.image_ids:
        images = (
            await db.execute(
                select(ImageModel).where(
                    ImageModel.id.in_(body.image_ids),
                    ImageModel.status != "pending",
                )
            )
        ).scalars().all()
    else:
        images = (
            await db.execute(
                select(ImageModel).where(ImageModel.status != "pending")
            )
        ).scalars().all()

    if not images:
        raise HTTPException(
            status_code=400,
            detail="No classified images found. Run classification first.",
        )

    # Fetch the top prediction label for each image
    labels_map: dict[int, str] = {}
    for img in images:
        top_pred = (
            await db.execute(
                select(PredictionModel)
                .where(PredictionModel.image_id == img.id)
                .order_by(PredictionModel.rank)
                .limit(1)
            )
        ).scalar_one_or_none()
        if top_pred:
            labels_map[img.id] = top_pred.label

    # Filter to images that actually have predictions
    images = [img for img in images if img.id in labels_map]
    if not images:
        raise HTTPException(status_code=400, detail="No predictions available.")

    classifier = _get_classifier()
    loop = asyncio.get_event_loop()

    # Extract embeddings sequentially (single-worker executor)
    embeddings = []
    image_ids = []
    labels = []
    filenames = []

    for img in images:
        try:
            emb = await loop.run_in_executor(
                _executor, classifier.extract_embeddings, img.file_path,
            )
            embeddings.append(emb)
            image_ids.append(img.id)
            labels.append(labels_map[img.id])
            filenames.append(img.original_name)
        except Exception as exc:
            logger.warning("Embedding extraction failed for image %d: %s", img.id, exc)

    if len(embeddings) < 1:
        raise HTTPException(status_code=500, detail="Failed to extract any embeddings.")

    # Run t-SNE in the thread pool (CPU-bound sklearn work)
    points = await loop.run_in_executor(
        None,
        _compute_tsne,
        embeddings,
        labels,
        image_ids,
        filenames,
        body.perplexity,
    )

    return TSNEOut(
        num_images=len(points),
        perplexity=min(body.perplexity, len(points) - 1) if len(points) > 1 else 0,
        points=[EmbeddingPoint(**p) for p in points],
    )
