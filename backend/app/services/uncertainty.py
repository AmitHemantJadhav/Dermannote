"""Uncertainty scoring for active learning queue.

Pure function: takes top-k predictions → returns uncertainty metrics.
Composite score: 0.4*(1-top1) + 0.3*(1-margin) + 0.3*normalized_entropy
"""

from __future__ import annotations

import math


def compute_uncertainty(predictions: list[dict]) -> dict:
    """Compute uncertainty metrics from ranked predictions.

    Args:
        predictions: list of dicts with at least {"confidence": float},
                     sorted by rank (highest confidence first).

    Returns:
        {max_confidence, margin, entropy, uncertainty_score}
    """
    if not predictions:
        return {
            "max_confidence": 0.0,
            "margin": 0.0,
            "entropy": 0.0,
            "uncertainty_score": 1.0,
        }

    confidences = [p["confidence"] for p in predictions]
    top1 = confidences[0]
    top2 = confidences[1] if len(confidences) > 1 else 0.0
    margin = top1 - top2

    # Shannon entropy over the prediction distribution
    entropy = 0.0
    for c in confidences:
        if c > 0:
            entropy -= c * math.log2(c)

    # Normalize entropy: max entropy for k classes = log2(k)
    k = len(confidences)
    max_entropy = math.log2(k) if k > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Composite uncertainty score (higher = more uncertain)
    uncertainty_score = 0.4 * (1 - top1) + 0.3 * (1 - margin) + 0.3 * norm_entropy

    return {
        "max_confidence": round(top1, 6),
        "margin": round(margin, 6),
        "entropy": round(entropy, 6),
        "uncertainty_score": round(uncertainty_score, 6),
    }
