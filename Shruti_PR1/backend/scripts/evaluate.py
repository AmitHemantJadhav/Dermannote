"""
Evaluation script for the DermAnnotate ML classifier.

Computes top-1 accuracy, top-3 accuracy, and per-class breakdown
against a labelled image dataset.

Input formats (pick one):
  --csv   PATH    CSV with columns: image_path, label
  --dir   PATH    Directory structured as label_name/image.jpg

Usage examples:
  # Evaluate real classifier against a CSV of known images
  python scripts/evaluate.py --csv dataset/labels.csv --real

  # Evaluate mock classifier against an ImageNet-style directory
  python scripts/evaluate.py --dir dataset/ham10000_sample/

  # Use real classifier, save results to a CSV
  python scripts/evaluate.py --csv labels.csv --real --out results.csv

  # Show a confusion matrix too
  python scripts/evaluate.py --dir dataset/ --real --confusion
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ── Label normalisation ────────────────────────────────────────────────────────

# Normalise whatever label the user provides to our canonical label set.
# Handles HAM10000 short codes, full English names, and loose variants.
_LABEL_ALIASES: dict[str, str] = {
    # Short codes
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma",
    # Full English variants
    "melanocytic nevi": "Melanocytic Nevi",
    "melanocytic nevus": "Melanocytic Nevi",
    "melanoma": "Melanoma",
    "benign keratosis-like lesions": "Benign Keratosis",
    "benign keratosis": "Benign Keratosis",
    "basal cell carcinoma": "Basal Cell Carcinoma",
    "actinic keratoses": "Actinic Keratosis",
    "actinic keratosis": "Actinic Keratosis",
    "actinic keratoses and intraepithelial carcinoma": "Actinic Keratosis",
    "intraepithelial carcinoma": "Actinic Keratosis",
    "vascular lesions": "Vascular Lesion",
    "vascular lesion": "Vascular Lesion",
    "dermatofibroma": "Dermatofibroma",
}

def normalise_label(raw: str) -> str:
    return _LABEL_ALIASES.get(raw) or _LABEL_ALIASES.get(raw.lower()) or raw.strip()


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_from_csv(csv_path: str) -> list[tuple[str, str]]:
    """
    Load (image_path, label) pairs from a CSV file.
    Expected columns: image_path, label  (header row required).
    """
    samples = []
    csv_dir = Path(csv_path).parent

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "image_path" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            print("Error: CSV must have 'image_path' and 'label' columns.")
            sys.exit(1)

        for row in reader:
            path = row["image_path"].strip()
            # Resolve relative paths against the CSV's directory
            if not os.path.isabs(path):
                path = str(csv_dir / path)
            label = normalise_label(row["label"].strip())
            samples.append((path, label))

    return samples


def load_from_dir(dir_path: str) -> list[tuple[str, str]]:
    """
    Load (image_path, label) pairs from a directory structured as:
        dir_path/
            Melanoma/
                img1.jpg
                img2.jpg
            Melanocytic Nevi/
                img3.jpg
    """
    samples = []
    base = Path(dir_path)

    for label_dir in sorted(base.iterdir()):
        if not label_dir.is_dir():
            continue
        label = normalise_label(label_dir.name)
        for img_file in sorted(label_dir.iterdir()):
            if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((str(img_file), label))

    return samples


# ── Evaluation core ────────────────────────────────────────────────────────────

def evaluate(
    samples: list[tuple[str, str]],
    classifier,
    show_confusion: bool = False,
    out_csv: str | None = None,
) -> None:
    n = len(samples)
    if n == 0:
        print("No samples found. Check your --csv or --dir path.")
        sys.exit(1)

    print(f"\nEvaluating {n} image(s) with {type(classifier).__name__}...\n")

    top1_correct = 0
    top3_correct = 0
    errors = 0

    # Per-class: {label: {"total": int, "top1": int, "top3": int}}
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})

    # For confusion matrix: {actual: {predicted: count}}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    rows_out = []  # for optional CSV output

    width = len(str(n))
    for i, (image_path, true_label) in enumerate(samples, 1):
        prefix = f"[{i:>{width}}/{n}]"

        if not os.path.exists(image_path):
            print(f"{prefix} SKIP  {image_path!r} — file not found")
            errors += 1
            continue

        try:
            t0 = time.perf_counter()
            preds = classifier.predict(image_path)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"{prefix} ERROR {Path(image_path).name} — {exc}")
            errors += 1
            continue

        pred_labels = [p["label"] for p in preds]
        top1_pred = pred_labels[0]
        top1_conf = preds[0]["confidence"]

        is_top1 = top1_pred == true_label
        is_top3 = true_label in pred_labels

        top1_correct += int(is_top1)
        top3_correct += int(is_top3)

        per_class[true_label]["total"] += 1
        per_class[true_label]["top1"] += int(is_top1)
        per_class[true_label]["top3"] += int(is_top3)

        confusion[true_label][top1_pred] += 1

        mark = "✓" if is_top1 else ("~" if is_top3 else "✗")
        print(
            f"{prefix} {mark}  {Path(image_path).name:<40}"
            f"  truth={true_label:<25}"
            f"  pred={top1_pred:<25}"
            f"  conf={top1_conf:.1%}"
            f"  {elapsed*1000:.0f}ms"
        )

        rows_out.append({
            "image_path": image_path,
            "true_label": true_label,
            "pred_rank1": top1_pred,
            "pred_rank2": pred_labels[1] if len(pred_labels) > 1 else "",
            "pred_rank3": pred_labels[2] if len(pred_labels) > 2 else "",
            "top1_correct": int(is_top1),
            "top3_correct": int(is_top3),
            "confidence": f"{top1_conf:.4f}",
        })

    evaluated = n - errors

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("SUMMARY")
    print("─" * 72)
    print(f"  Images evaluated : {evaluated}")
    print(f"  Errors / skipped : {errors}")
    if evaluated > 0:
        print(f"  Top-1 accuracy   : {top1_correct}/{evaluated} = {top1_correct/evaluated:.1%}")
        print(f"  Top-3 accuracy   : {top3_correct}/{evaluated} = {top3_correct/evaluated:.1%}")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    if per_class:
        print("\n" + "─" * 72)
        print(f"  {'Class':<30} {'Samples':>7} {'Top-1':>8} {'Top-3':>8}")
        print("─" * 72)
        for label in sorted(per_class):
            stats = per_class[label]
            t = stats["total"]
            t1_pct = stats["top1"] / t if t else 0
            t3_pct = stats["top3"] / t if t else 0
            bar = "█" * int(t1_pct * 20) + "░" * (20 - int(t1_pct * 20))
            print(f"  {label:<30} {t:>7}   {bar} {t1_pct:>5.1%}  {t3_pct:>5.1%}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if show_confusion and confusion:
        all_labels = sorted(
            {lbl for lbl in list(confusion.keys()) + [p for preds in confusion.values() for p in preds]}
        )
        col_w = max(len(l) for l in all_labels)

        print("\n" + "─" * 72)
        print("CONFUSION MATRIX  (rows = actual, cols = predicted)")
        print("─" * 72)
        header = f"  {'':>{col_w}} " + "  ".join(f"{l[:6]:>6}" for l in all_labels)
        print(header)
        for actual in all_labels:
            row_vals = "  ".join(
                f"{confusion[actual].get(pred, 0):>6}" for pred in all_labels
            )
            marker = "←" if any(confusion[actual].values()) else ""
            print(f"  {actual:>{col_w}} {row_vals} {marker}")

    # ── Optional CSV output ───────────────────────────────────────────────────
    if out_csv and rows_out:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nResults saved to: {out_csv}")

    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DermAnnotate ML classifier accuracy against labelled images."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv", metavar="PATH",
        help="CSV file with columns: image_path, label",
    )
    source.add_argument(
        "--dir", metavar="PATH",
        help="Directory structured as label_name/image.jpg",
    )

    parser.add_argument(
        "--real", action="store_true",
        help="Use RealClassifier (requires transformers + torch + downloaded weights)",
    )
    parser.add_argument(
        "--confusion", action="store_true",
        help="Print a confusion matrix after the summary",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="Save per-image results to a CSV file",
    )

    args = parser.parse_args()

    # Load dataset
    if args.csv:
        samples = load_from_csv(args.csv)
        print(f"Loaded {len(samples)} sample(s) from CSV: {args.csv}")
    else:
        samples = load_from_dir(args.dir)
        print(f"Loaded {len(samples)} sample(s) from directory: {args.dir}")

    if not samples:
        print("No valid samples found.")
        sys.exit(1)

    # Load classifier
    if args.real:
        from ml.classifier import RealClassifier
        try:
            clf = RealClassifier()
        except ImportError as exc:
            print(f"Cannot load RealClassifier: {exc}")
            sys.exit(1)
    else:
        from ml.classifier import MockClassifier
        clf = MockClassifier()
        print("Note: MockClassifier predictions are seeded from pixel stats, not real ML.")

    evaluate(samples, clf, show_confusion=args.confusion, out_csv=args.out)


if __name__ == "__main__":
    main()
