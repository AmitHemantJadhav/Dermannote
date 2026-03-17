"""
ML classifier pipeline for DermAnnotate.

Two modes controlled by USE_MOCK in backend/.env:
  - MockClassifier (default): deterministic, color-stats based — no GPU, no downloads
  - RealClassifier: EfficientNet fine-tuned on HAM10000 (marmal88/skin_cancer on HuggingFace)
"""

from __future__ import annotations

import logging
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Label definitions
# -------------------------------------------------------------------

# Exactly the 7 HAM10000 classes — consistent with what marmal88/skin_cancer
# was trained on. Mock and real classifiers now predict the same label set.
CONDITION_LABELS = [
    "Melanocytic Nevi",
    "Melanoma",
    "Benign Keratosis",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Vascular Lesion",
    "Dermatofibroma",
]

# Comprehensive case-insensitive map covering:
#   • HAM10000 short codes (nv, mel, …)
#   • Full English class names the HuggingFace model may emit
#   • Our canonical display names (identity pass-through)
_HAM_LABEL_NORMALIZE: dict[str, str] = {
    # Short codes
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma",
    # Full English variants found in the wild across HuggingFace configs
    "melanocytic nevi": "Melanocytic Nevi",
    "melanocytic nevus": "Melanocytic Nevi",
    "melanoma": "Melanoma",
    "benign keratosis-like lesions": "Benign Keratosis",
    "benign keratosis": "Benign Keratosis",
    "basal cell carcinoma": "Basal Cell Carcinoma",
    "actinic keratoses": "Actinic Keratosis",
    "actinic keratosis": "Actinic Keratosis",
    "intraepithelial carcinoma": "Actinic Keratosis",
    "actinic keratoses and intraepithelial carcinoma": "Actinic Keratosis",
    "vascular lesions": "Vascular Lesion",
    "vascular lesion": "Vascular Lesion",
    "dermatofibroma": "Dermatofibroma",
    # gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer label format
    "actinic-keratoses": "Actinic Keratosis",
    "basal-cell-carcinoma": "Basal Cell Carcinoma",
    "benign-keratosis-like-lesions": "Benign Keratosis",
    "melanocytic-nevi": "Melanocytic Nevi",
    "melanoma, nos": "Melanoma",
    "vascular-lesions": "Vascular Lesion",
}


def _map_ham_label(raw: str) -> str:
    """Normalise a raw model label to our canonical display name."""
    return (
        _HAM_LABEL_NORMALIZE.get(raw)
        or _HAM_LABEL_NORMALIZE.get(raw.lower())
        or raw  # fall back to whatever the model returns
    )


def _mps_available(torch) -> bool:  # noqa: ANN001
    """Return True if Apple Silicon MPS backend is available."""
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


# -------------------------------------------------------------------
# Image loading helper (shared by both classifiers)
# -------------------------------------------------------------------

_MAX_IMAGE_SIDE = 2048  # resize images larger than this before inference


def _load_rgb(image_path: str, max_side: int = _MAX_IMAGE_SIDE) -> Image.Image:
    """
    Open, validate, decode, and return an RGB PIL image.

    Raises:
        FileNotFoundError: image_path does not exist.
        ValueError: file is corrupt or cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = Image.open(image_path)
        img.load()  # force decode — catches truncated/corrupt files early
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot decode image '{image_path}': {exc}") from exc

    img = img.convert("RGB")

    if img.width > max_side or img.height > max_side:
        logger.debug(
            "Resizing large image %dx%d → max_side=%d before inference.",
            img.width, img.height, max_side,
        )
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    return img


# -------------------------------------------------------------------
# MockClassifier
# -------------------------------------------------------------------

class MockClassifier:
    """
    Deterministic fake classifier based on image RGB statistics.

    Algorithm:
    1. Compute per-channel means (R, G, B).
    2. Derive a reproducible integer seed from those means.
    3. Sample a Dirichlet probability vector over all 13 labels.
    4. Boost the top-ranked class into a realistic 60–95 % confidence band.
    5. Return top-3 predictions (rank, label, confidence).

    Identical input → identical output every time (no randomness at runtime).
    """

    model_name = "mock-v1"

    def predict(self, image_path: str) -> list[dict]:
        logger.debug("MockClassifier.predict(%s)", image_path)

        img = _load_rgb(image_path)
        arr = np.array(img, dtype=np.float32)

        r_mean = float(arr[:, :, 0].mean())
        g_mean = float(arr[:, :, 1].mean())
        b_mean = float(arr[:, :, 2].mean())

        # Map channel means → a deterministic seed in [0, 2³¹)
        seed = int(r_mean * 1_000 + g_mean * 100 + b_mean * 10) % (2**31)
        rng = np.random.default_rng(seed)

        # Dirichlet sample with varied α for a natural-looking confidence spread
        alpha = rng.uniform(0.3, 3.0, len(CONDITION_LABELS))
        probs = rng.dirichlet(alpha)

        # Boost the top class into a 60–95 % confidence band
        top_idx = int(np.argmax(probs))
        boost = float(rng.uniform(0.60, 0.95))

        remaining = 1.0 - boost
        other = probs.copy()
        other[top_idx] = 0.0
        other = other / other.sum() * remaining
        probs = other
        probs[top_idx] = boost

        top3 = np.argsort(probs)[::-1][:3]

        results = [
            {
                "label": CONDITION_LABELS[idx],
                "confidence": float(probs[idx]),
                "rank": rank,
                "model_name": self.model_name,
            }
            for rank, idx in enumerate(top3, start=1)
        ]

        logger.debug("MockClassifier results: %s", [(r["label"], r["confidence"]) for r in results])
        return results


# -------------------------------------------------------------------
# RealClassifier
# -------------------------------------------------------------------

class RealClassifier:
    """
    EfficientNet fine-tuned on HAM10000, loaded from HuggingFace: marmal88/skin_cancer.

    Requirements:
        pip install transformers torch

    GPU is used automatically if CUDA is available; falls back to CPU.
    Returns top-3 softmax probabilities mapped to canonical label names.
    """

    model_name = "efficientnet-isic"

    def __init__(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError(
                "RealClassifier requires 'transformers' and 'torch'.\n"
                "Install: pip install transformers torch\n"
                f"Original error: {exc}"
            ) from exc

        # Use locally trained model if SKIN_MODEL_PATH is set in .env, otherwise HuggingFace
        model_id = os.getenv("SKIN_MODEL_PATH") or "gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if _mps_available(torch) else "cpu"
        logger.info("Loading RealClassifier from '%s' on device '%s'…", model_id, self.device)

        # AutoImageProcessor supersedes the deprecated AutoFeatureExtractor
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "RealClassifier ready — %d output classes on %s.",
            self.model.config.num_labels,
            self.device,
        )

    def predict(self, image_path: str) -> list[dict]:
        import torch

        logger.debug("RealClassifier.predict(%s)", image_path)

        img = _load_rgb(image_path)
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        k = min(3, len(probs))
        top = torch.topk(probs, k)

        results = []
        for rank, (prob, idx) in enumerate(zip(top.values, top.indices), start=1):
            raw_label = self.model.config.id2label[idx.item()]
            label = _map_ham_label(raw_label)
            results.append(
                {
                    "label": label,
                    "confidence": float(prob),
                    "rank": rank,
                    "model_name": self.model_name,
                }
            )

        logger.debug("RealClassifier results: %s", [(r["label"], r["confidence"]) for r in results])
        return results


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------

def create_classifier() -> MockClassifier | RealClassifier:
    """
    Read USE_MOCK from the environment and return the appropriate classifier.
    Defaults to MockClassifier so the app works without any downloads or GPU.
    """
    use_mock = os.getenv("USE_MOCK", "true").strip().lower() == "true"
    logger.info("Creating classifier — USE_MOCK=%s", use_mock)

    if use_mock:
        return MockClassifier()

    return RealClassifier()
