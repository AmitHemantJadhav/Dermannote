"""
One-time script to pre-download HuggingFace model weights for offline use.

Run from backend/ directory before switching USE_MOCK=false in .env:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_models() -> None:
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        print(
            "Error: 'transformers' is not installed.\n"
            "Install it with:\n"
            "  pip install transformers torch\n"
            "(See the commented lines at the bottom of requirements.txt)"
        )
        sys.exit(1)

    model_id = "gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer"

    print(f"Downloading image processor from '{model_id}'...")
    AutoImageProcessor.from_pretrained(model_id)

    print(f"Downloading model weights from '{model_id}'...")
    AutoModelForImageClassification.from_pretrained(model_id)

    print("\nDone.")
    print("Set USE_MOCK=false in backend/.env to use the real classifier.")
    print("On first request the model will be loaded into memory (~300 MB on CPU).")


if __name__ == "__main__":
    download_models()
