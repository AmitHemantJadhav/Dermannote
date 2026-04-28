"""
MobileSAM auto-segmentation for DermAnnotate.

Provides 1-click point-prompt segmentation: user clicks on a lesion,
SAM returns a polygon mask. Uses the same lazy-load + thread-safe
singleton pattern as ml/classifier.py.
"""

from __future__ import annotations

import logging
import os
import urllib.request

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "mobile_sam.pt")
CHECKPOINT_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"


def _mps_available(torch) -> bool:  # noqa: ANN001
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def _ensure_checkpoint() -> str:
    """Download MobileSAM checkpoint if not present."""
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info("Downloading MobileSAM checkpoint to %s …", CHECKPOINT_PATH)
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    logger.info("MobileSAM checkpoint downloaded.")
    return CHECKPOINT_PATH


class SAMSegmenter:
    """MobileSAM point-prompt segmenter."""

    def __init__(self) -> None:
        import torch
        from mobile_sam import SamPredictor, sam_model_registry

        checkpoint = _ensure_checkpoint()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if _mps_available(torch) else "cpu"

        logger.info("Loading MobileSAM on device '%s'…", self.device)
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        logger.info("MobileSAM ready.")

    def segment(
        self,
        image_path: str,
        point_x: float,
        point_y: float,
        img_width: int,
        img_height: int,
    ) -> list[dict[str, float]]:
        """
        Run point-prompt segmentation.

        Args:
            image_path: Path to the original image on disk.
            point_x, point_y: Click coordinates in original image pixel space.
            img_width, img_height: Original image dimensions (for validation).

        Returns:
            List of {x, y} polygon points in original image coordinates.
        """
        import torch

        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        self.predictor.set_image(img_array)

        input_point = np.array([[point_x, point_y]])
        input_label = np.array([1])  # 1 = foreground

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Pick the mask with the highest score
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)

        # Convert binary mask to polygon contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify polygon to reduce point count
        epsilon = 2.0
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        points = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]
        logger.debug("SAM segment: %d contour points → %d simplified points", len(contour), len(points))
        return points


def create_sam_model() -> SAMSegmenter:
    """Factory function — matches the pattern from ml/classifier.py."""
    return SAMSegmenter()
