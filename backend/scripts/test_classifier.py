"""
Smoke test for the DermAnnotate ML classifier pipeline.

Run from backend/ directory:
    python scripts/test_classifier.py                # mock mode only
    python scripts/test_classifier.py --real         # also test RealClassifier
    python scripts/test_classifier.py --image /path/to/skin.jpg
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Ensure ml/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env so USE_MOCK and any other vars are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_synthetic_image() -> str:
    """
    Write a small 224×224 synthetic skin-tone JPEG to a temp file.
    Returns the file path (caller is responsible for cleanup).
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    arr = rng.integers(150, 220, (224, 224, 3), dtype=np.uint8)
    # Shift channels to approximate a reddish skin tone
    arr[:, :, 0] = np.clip(arr[:, :, 0] + 20, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - 20, 0, 255)

    img = Image.fromarray(arr, "RGB")
    path = tempfile.mktemp(suffix=".jpg")
    img.save(path, quality=90)
    return path


def _print_results(results: list[dict]) -> None:
    for pred in results:
        bar_len = int(pred["confidence"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  [{pred['rank']}] {bar} {pred['confidence']:5.1%}  {pred['label']}")


# ── Test functions ─────────────────────────────────────────────────────────────

def test_mock(image_path: str) -> None:
    from ml.classifier import MockClassifier

    print("\n── MockClassifier ────────────────────────────────────────────────────")
    clf = MockClassifier()

    results = clf.predict(image_path)

    assert len(results) == 3, f"Expected 3 predictions, got {len(results)}"
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
    assert results[2]["rank"] == 3

    total = sum(r["confidence"] for r in results)
    assert total <= 1.001, f"Confidences sum to {total:.4f} > 1 (unexpected)"

    for r in results:
        assert 0.0 <= r["confidence"] <= 1.0, f"Confidence out of range: {r}"
        assert r["label"], "Empty label"
        assert r["model_name"] == "mock-v1"

    _print_results(results)

    # Determinism: same image must always produce the same output
    results2 = clf.predict(image_path)
    assert results == results2, "MockClassifier is NOT deterministic — seed logic broken"
    print("  ✓ Determinism check passed (same input → identical output)")
    print("  ✓ MockClassifier OK")


def test_real(image_path: str) -> None:
    print("\n── RealClassifier ────────────────────────────────────────────────────")

    try:
        from ml.classifier import RealClassifier
        clf = RealClassifier()
    except ImportError as exc:
        print(f"  ✗ Cannot load RealClassifier: {exc}")
        print("    Install with: pip install transformers torch")
        sys.exit(1)

    results = clf.predict(image_path)

    assert len(results) == 3, f"Expected 3 predictions, got {len(results)}"

    for r in results:
        assert 0.0 <= r["confidence"] <= 1.0, f"Confidence out of range: {r}"
        assert r["label"], "Empty label"
        assert r["model_name"] == "efficientnet-isic"

    _print_results(results)
    print("  ✓ RealClassifier OK")


def test_error_handling() -> None:
    """Confirm _load_rgb raises the right exceptions for bad inputs."""
    from ml.classifier import _load_rgb

    print("\n── Error handling ────────────────────────────────────────────────────")

    # Non-existent file → FileNotFoundError
    try:
        _load_rgb("/tmp/__dermannotate_nonexistent__.jpg")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  ✓ Missing file → FileNotFoundError")

    # Corrupt file → ValueError
    corrupt = tempfile.mktemp(suffix=".jpg")
    try:
        with open(corrupt, "wb") as f:
            f.write(b"this is not a jpeg")
        try:
            _load_rgb(corrupt)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("  ✓ Corrupt file  → ValueError")
    finally:
        if os.path.exists(corrupt):
            os.remove(corrupt)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DermAnnotate ML pipeline smoke test")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Also run RealClassifier test (requires transformers + torch + downloaded weights)",
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        help="Use an existing image instead of a synthetic one",
    )
    args = parser.parse_args()

    if args.image:
        image_path = args.image
        cleanup = False
        print(f"Using image: {image_path}")
    else:
        print("No --image provided — generating a synthetic 224×224 skin-tone image.")
        image_path = create_synthetic_image()
        cleanup = True
        print(f"Temp image: {image_path}")

    try:
        test_mock(image_path)
        test_error_handling()

        if args.real:
            test_real(image_path)
        else:
            print("\n(Skip --real flag to also test RealClassifier)")

        print("\n✓ All tests passed.\n")
    finally:
        if cleanup and os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    main()
