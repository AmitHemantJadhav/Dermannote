"""
Fine-tune a skin lesion classifier on HAM10000 with class balancing.

Starts from gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer
(already trained on HAM10000) and fine-tunes further with:

  1. WeightedRandomSampler  — every class equally likely per batch
  2. Class-weighted loss    — errors on rare classes penalized more heavily
  3. Augmented minority data — aug_ images already in the dataset

This directly fixes the bias toward Melanocytic Nevi seen in evaluation.

Training time estimates:
  Apple Silicon MPS : ~15–20 min for 5 epochs (8 190 images, batch=16)
  CUDA GPU          : ~5–10 min
  CPU only          : ~2–3 hours

After training the model is saved to models/skin_classifier/.
Add this line to backend/.env to use it:
  SKIN_MODEL_PATH=models/skin_classifier

Usage:
  python scripts/train_classifier.py                    # 5 epochs, default settings
  python scripts/train_classifier.py --epochs 10        # train longer for higher accuracy
  python scripts/train_classifier.py --lr 1e-5          # use smaller learning rate
  python scripts/train_classifier.py --out models/v2    # custom save path
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL    = "gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer"
DATASET_REPO  = "hawking32/ham10000_ttv"
DATA_DIR      = Path(__file__).parent.parent / "dataset" / "ham10000_full"
DEFAULT_OUT   = Path(__file__).parent.parent / "models" / "skin_classifier"

# HAM10000 short-code → gianlab model label index (alphabetical order)
HAM_CLASSES = {
    "akiec": (0, "Actinic Keratosis"),
    "bcc":   (1, "Basal Cell Carcinoma"),
    "bkl":   (2, "Benign Keratosis"),
    "df":    (3, "Dermatofibroma"),
    "nv":    (4, "Melanocytic Nevi"),
    "mel":   (5, "Melanoma"),
    "vasc":  (6, "Vascular Lesion"),
}


# ── Dataset download ──────────────────────────────────────────────────────────

def download_splits(splits: list[str]) -> None:
    """Download train + val images from HuggingFace if not already present."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Error: huggingface_hub not installed.")
        sys.exit(1)

    print(f"Listing files in {DATASET_REPO}...")
    all_files = list(list_repo_files(DATASET_REPO, repo_type="dataset"))

    by_split: dict[str, list[str]] = {s: [] for s in splits}
    for f in all_files:
        parts = f.split("/")
        if len(parts) == 3 and parts[0] in splits and parts[1] in HAM_CLASSES and f.endswith(".jpg"):
            by_split[parts[0]].append(f)

    total = sum(len(v) for v in by_split.values())
    already = sum(
        1 for files in by_split.values() for rp in files
        if (DATA_DIR / rp).exists()
    )
    if already == total:
        print(f"All {total} images already downloaded in {DATA_DIR}.")
        return

    need = total - already
    print(f"Downloading {need} missing image(s) → {DATA_DIR}")
    cache_dir = DATA_DIR.parent / "_hf_cache"
    done = 0
    for split, files in by_split.items():
        for repo_path in files:
            dest = DATA_DIR / repo_path
            if dest.exists():
                done += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                local = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=str(cache_dir),
                )
                shutil.copy2(local, dest)
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total} downloaded…")
            except Exception as exc:
                print(f"  SKIP {repo_path}: {exc}")

    print(f"Download complete: {done}/{total} images in {DATA_DIR}")


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

def build_dataset_class():
    """Return a Dataset class (imported lazily so torch import is deferred)."""
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    class SkinDataset(Dataset):
        def __init__(self, root_dir: Path, processor) -> None:
            self.processor = processor
            self.samples: list[tuple[Path, int]] = []

            for short_code, (label_idx, _) in HAM_CLASSES.items():
                cls_dir = root_dir / short_code
                if not cls_dir.exists():
                    continue
                for img_path in cls_dir.glob("*.jpg"):
                    self.samples.append((img_path, label_idx))

            if not self.samples:
                raise RuntimeError(f"No images found under {root_dir}")

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
            enc = self.processor(images=img, return_tensors="pt")
            return {
                "pixel_values": enc["pixel_values"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
            }

        def class_counts(self) -> list[int]:
            counts = Counter(label for _, label in self.samples)
            return [counts[i] for i in range(len(HAM_CLASSES))]

    return SkinDataset


# ── Weighted sampler ──────────────────────────────────────────────────────────

def make_weighted_sampler(dataset):
    """Return a WeightedRandomSampler that balances all classes per epoch."""
    from torch.utils.data import WeightedRandomSampler

    counts = dataset.class_counts()
    # weight per sample = 1 / count of its class
    weights = [1.0 / counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    epochs: int,
    lr: float,
    batch_size: int,
    output_dir: Path,
) -> None:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Model + processor ─────────────────────────────────────────────────────
    print(f"Loading base model: {BASE_MODEL}")
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForImageClassification.from_pretrained(BASE_MODEL)
    model.to(device)

    # ── Datasets ─────────────────────────────────────────────────────────────
    SkinDataset = build_dataset_class()
    train_ds = SkinDataset(DATA_DIR / "train", processor)
    val_ds   = SkinDataset(DATA_DIR / "val",   processor)

    print(f"\nTrain: {len(train_ds)} images")
    print(f"Val  : {len(val_ds)} images")

    train_counts = train_ds.class_counts()
    print("\nClass distribution (train):")
    for short, (idx, name) in HAM_CLASSES.items():
        bar = "█" * int(train_counts[idx] / max(train_counts) * 20)
        print(f"  {name:<25} {train_counts[idx]:>5}  {bar}")

    # ── Class weights for loss ────────────────────────────────────────────────
    # weight_i = total_samples / (num_classes * count_i)
    total = sum(train_counts)
    n_cls = len(train_counts)
    class_weights = torch.tensor(
        [total / (n_cls * c) if c > 0 else 0.0 for c in train_counts],
        dtype=torch.float32,
    ).to(device)
    print(f"\nClass weights: {[round(w.item(), 2) for w in class_weights]}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(device.type != "mps"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {epochs} epoch(s), lr={lr}, batch={batch_size}\n")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        train_loss = 0.0
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            if step % 50 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)}  "
                      f"loss={train_loss/step:.4f}", end="\r")

        avg_loss = train_loss / len(train_loader)
        elapsed = time.perf_counter() - t0

        # — Validate —
        model.eval()
        correct = total_val = 0
        per_class_correct = Counter()
        per_class_total   = Counter()

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels       = batch["labels"].to(device)
                logits = model(pixel_values=pixel_values).logits
                preds = logits.argmax(dim=-1)

                correct    += (preds == labels).sum().item()
                total_val  += labels.size(0)
                for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    per_class_total[l]   += 1
                    per_class_correct[l] += int(p == l)

        val_acc = correct / total_val
        print(f"\nEpoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
              f"val_acc={val_acc:.1%}  ({elapsed:.0f}s)")

        # Per-class accuracy
        for short, (idx, name) in HAM_CLASSES.items():
            t = per_class_total[idx]
            c = per_class_correct[idx]
            bar = "█" * int((c / t if t else 0) * 20)
            print(f"  {name:<25} {c:>4}/{t:<4} {bar}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_dir))
            processor.save_pretrained(str(output_dir))
            print(f"  ✓ New best ({val_acc:.1%}) — saved to {output_dir}")

    print(f"\nTraining done. Best val accuracy: {best_val_acc:.1%}")
    print(f"Model saved to: {output_dir}")
    print("\nTo use your trained model, add to backend/.env:")
    print(f"  SKIN_MODEL_PATH={output_dir}")
    print("\nThen re-evaluate:")
    print(f"  python scripts/evaluate.py --dir dataset/ham10000_test --real --confusion")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune skin lesion classifier with class balancing."
    )
    parser.add_argument("--epochs",     type=int,   default=5,       help="Training epochs (default: 5)")
    parser.add_argument("--lr",         type=float, default=2e-5,    help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int,   default=16,      help="Batch size (default: 16)")
    parser.add_argument("--out",        type=Path,  default=DEFAULT_OUT, help="Output directory for saved model")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading training data (assumes already downloaded)")
    args = parser.parse_args()

    if not args.skip_download:
        download_splits(["train", "val"])

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
