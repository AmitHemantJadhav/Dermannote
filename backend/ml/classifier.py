"""
ML classifier pipeline for DermAnnotate.

Two modes controlled by USE_MOCK in backend/.env:
  - MockClassifier (default): deterministic, color-stats based — no GPU, no downloads
  - RealClassifier: Vision Transformer fine-tuned on HAM10000 (supports ViT, Swin, DeiT)
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

    def generate_gradcam(self, image_path: str, target_class_idx: int | None = None) -> None:
        """Mock classifier has no model — GradCAM is not available."""
        return None

    def extract_embeddings(self, image_path: str) -> np.ndarray:
        """Return a deterministic pseudo-embedding for mock mode (768-d)."""
        img = _load_rgb(image_path)
        arr = np.array(img, dtype=np.float32)
        seed = int(arr[:, :, 0].mean() * 1000 + arr[:, :, 1].mean() * 100) % (2**31)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(768).astype(np.float32)

    def extract_attention(
        self, image_path: str, layer: int = 11, head: int | None = None,
    ) -> np.ndarray | None:
        """Mock classifier has no real attention — not available."""
        return None

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
    Vision Transformer fine-tuned on HAM10000.

    Supports ViT-Base, Swin-Tiny, and DeiT-Small architectures.
    Loads from local path (SKIN_MODEL_PATH) or HuggingFace fallback.
    Returns top-3 softmax probabilities mapped to canonical label names.
    """

    model_name = "vit-base-ham10000"

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
        model_id = os.getenv("SKIN_MODEL_PATH") or "google/vit-base-patch16-224"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if _mps_available(torch) else "cpu"
        logger.info("Loading RealClassifier from '%s' on device '%s'…", model_id, self.device)

        # AutoImageProcessor supersedes the deprecated AutoFeatureExtractor
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        # Use "eager" attention so we can extract per-head attention weights
        # for the attention visualization feature. SDPA fuses the attention
        # computation in a single kernel and does not expose the weight matrix.
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id, attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "RealClassifier ready — %d output classes on %s.",
            self.model.config.num_labels,
            self.device,
        )

    def generate_gradcam(self, image_path: str, target_class_idx: int | None = None) -> np.ndarray | None:
        """
        Generate a GradCAM heatmap overlay for the given image.

        Returns an RGB numpy array matching the original image dimensions (uint8)
        or None on failure. Uses the model's top predicted class when target is None.
        """
        import torch
        from pytorch_grad_cam import EigenGradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        logger.debug("RealClassifier.generate_gradcam(%s, target=%s)", image_path, target_class_idx)

        img = _load_rgb(image_path)
        img_resized = img.resize((224, 224), Image.LANCZOS)
        rgb_array = np.array(img_resized, dtype=np.float32) / 255.0

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Determine target class if not specified
        if target_class_idx is None:
            with torch.no_grad():
                logits = self.model(**inputs).logits
            target_class_idx = int(logits.argmax(dim=-1).item())

        # Auto-detect architecture and configure target layer + reshape.
        # Standard GradCAM relies on spatially localized gradients, which works
        # for CNNs but fails for ViTs because self-attention distributes gradients
        # uniformly across all patch tokens. EigenGradCAM addresses this by applying
        # PCA to the element-wise product of activations and gradients, extracting
        # the dominant spatial pattern regardless of gradient magnitude.
        if hasattr(self.model, "vit"):
            # ViT / DeiT: target the pre-attention LayerNorm in the last block.
            # This preserves per-patch feature variance before attention mixing.
            target_layer = self.model.vit.encoder.layer[-1].layernorm_before

            def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
                b, tokens, c = tensor.shape
                # Drop CLS token (index 0), reshape 196 patch tokens → 14×14 grid
                spatial = tensor[:, 1:, :]
                h = w = int(spatial.shape[1] ** 0.5)
                return spatial.reshape(b, h, w, c).permute(0, 3, 1, 2)

        elif hasattr(self.model, "swin"):
            # Swin-Tiny: last stage output, 7×7 spatial tokens, no CLS token
            target_layer = self.model.swin.layernorm

            def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
                b, tokens, c = tensor.shape
                h = w = int(tokens ** 0.5)
                return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2)
        else:
            logger.warning("Unknown model architecture for GradCAM — cannot determine target layer.")
            return None

        # Wrap model so forward() returns raw logits tensor (not HuggingFace output object)
        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self._model = model

            def forward(self, pixel_values):
                return self._model(pixel_values=pixel_values).logits

        wrapper = _LogitsWrapper(self.model)

        cam = EigenGradCAM(
            model=wrapper,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )

        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # first (only) image in batch

        logger.debug(
            "GradCAM map stats: min=%.4f max=%.4f std=%.4f",
            grayscale_cam.min(), grayscale_cam.max(), grayscale_cam.std(),
        )

        # Blend heatmap onto original image (lower image_weight = more visible heatmap)
        visualization = show_cam_on_image(rgb_array, grayscale_cam, use_rgb=True, image_weight=0.35)

        # Resize to original image dimensions so the overlay aligns correctly
        vis_pil = Image.fromarray(visualization)
        vis_pil = vis_pil.resize((img.width, img.height), Image.LANCZOS)
        return np.array(vis_pil)

    def extract_embeddings(self, image_path: str) -> np.ndarray:
        """
        Extract the CLS token embedding from the penultimate layer.
        Returns a 1-D numpy array (768-d for ViT-Base).
        """
        import torch

        img = _load_rgb(image_path)
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(self.model, "vit"):
                outputs = self.model.vit(pixel_values=inputs["pixel_values"])
            elif hasattr(self.model, "swin"):
                outputs = self.model.swin(pixel_values=inputs["pixel_values"])
            else:
                logger.warning("Unknown architecture for embedding extraction.")
                return np.zeros(768, dtype=np.float32)

            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()

        return cls_embedding

    def extract_attention(
        self, image_path: str, layer: int = 11, head: int | None = None,
    ) -> np.ndarray | None:
        """
        Extract attention weights and produce a heatmap overlay.

        Returns an RGB numpy array (224x224x3, uint8) showing attention
        from the CLS token to spatial patches, blended onto the original image.
        If head is None, averages across all heads in the given layer.
        """
        import cv2
        import torch

        if not hasattr(self.model, "vit"):
            logger.warning("Attention extraction only supported for ViT architectures.")
            return None

        img = _load_rgb(image_path)
        img_resized = img.resize((224, 224), Image.LANCZOS)
        rgb_array = np.array(img_resized, dtype=np.float32) / 255.0

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        num_layers = self.model.config.num_hidden_layers
        if layer < 0 or layer >= num_layers:
            layer = num_layers - 1

        with torch.no_grad():
            outputs = self.model.vit(
                pixel_values=inputs["pixel_values"],
                output_attentions=True,
            )

        if not outputs.attentions or len(outputs.attentions) == 0:
            logger.warning("Model returned no attention weights.")
            return None

        # attentions[layer] shape: (batch, num_heads, seq_len, seq_len)
        attn = outputs.attentions[layer][0]  # (num_heads, 197, 197)

        # CLS token (index 0) attention to the 196 patch tokens
        cls_attn = attn[:, 0, 1:]  # (num_heads, 196)

        if head is not None:
            num_heads = cls_attn.shape[0]
            head = min(head, num_heads - 1)
            attn_map = cls_attn[head].cpu().numpy()  # (196,)
        else:
            attn_map = cls_attn.mean(dim=0).cpu().numpy()  # (196,)

        # Reshape to spatial grid and normalize
        grid_size = int(attn_map.shape[0] ** 0.5)  # 14 for ViT-Base
        attn_map = attn_map.reshape(grid_size, grid_size)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Upscale to image dimensions and apply colormap
        attn_resized = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(
            (attn_resized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO,
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend heatmap with original image
        blended = (0.55 * (rgb_array * 255) + 0.45 * heatmap.astype(np.float32))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return blended

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


def save_gradcam_png(visualization: np.ndarray, output_path: str) -> None:
    """Save a GradCAM RGB numpy array as a PNG file."""
    Image.fromarray(visualization).save(output_path, format="PNG")


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
