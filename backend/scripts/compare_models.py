"""
Compare 3 Vision Transformer architectures on HAM10000 skin lesion classification.

Models:
  1. Swin-Tiny  (Microsoft) — shifted window attention, ~28M params
  2. ViT-Base   (Google)    — original Vision Transformer, ~86M params
  3. DeiT-Small (Meta)      — data-efficient ViT with distillation, ~22M params

All models are fine-tuned under identical conditions (same data, hyperparams,
loss function, and weighted sampling) so the comparison is fair.

Usage:
  python scripts/compare_models.py --epochs 5           # full run, all 3 models
  python scripts/compare_models.py --models vit_base    # train only ViT-Base
  python scripts/compare_models.py --skip-training      # evaluate already-trained models
  python scripts/compare_models.py --skip-download       # skip dataset download step
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_REPO = "hawking32/ham10000_ttv"
DATA_DIR     = Path(__file__).parent.parent / "dataset" / "ham10000_full"
TEST_DIR     = Path(__file__).parent.parent / "dataset" / "ham10000_test"
MODELS_DIR   = Path(__file__).parent.parent / "models"
RESULTS_DIR  = Path(__file__).parent.parent / "results"

HAM_CLASSES = {
    "akiec": (0, "Actinic Keratosis"),
    "bcc":   (1, "Basal Cell Carcinoma"),
    "bkl":   (2, "Benign Keratosis"),
    "df":    (3, "Dermatofibroma"),
    "nv":    (4, "Melanocytic Nevi"),
    "mel":   (5, "Melanoma"),
    "vasc":  (6, "Vascular Lesion"),
}

ID2LABEL = {idx: name for _, (idx, name) in HAM_CLASSES.items()}
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}
NUM_CLASSES = len(HAM_CLASSES)


# ── Model configs ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_id: str           # HuggingFace model ID
    display_name: str       # pretty name for reports
    short_name: str         # directory / CLI key
    needs_head_replace: bool  # True if pretrained on ImageNet (1000 classes)

MODEL_CONFIGS = [
    ModelConfig(
        model_id="gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer",
        display_name="Swin-Tiny",
        short_name="swin_tiny",
        needs_head_replace=False,  # already has 7-class head
    ),
    ModelConfig(
        model_id="google/vit-base-patch16-224",
        display_name="ViT-Base-16",
        short_name="vit_base",
        needs_head_replace=True,   # ImageNet 1000-class head
    ),
    ModelConfig(
        model_id="facebook/deit-small-patch16-224",
        display_name="DeiT-Small-16",
        short_name="deit_small",
        needs_head_replace=True,   # ImageNet 1000-class head
    ),
]

CONFIG_BY_NAME = {c.short_name: c for c in MODEL_CONFIGS}


# ── Dataset download (reused from train_classifier.py) ───────────────────────

def download_splits(splits: list[str]) -> None:
    """Download train + val images from HuggingFace if not already present."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Error: huggingface_hub not installed.")
        sys.exit(1)

    import shutil

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
            return [counts[i] for i in range(NUM_CLASSES)]

    return SkinDataset


# ── Weighted sampler ──────────────────────────────────────────────────────────

def make_weighted_sampler(dataset):
    """Return a WeightedRandomSampler that balances all classes per epoch."""
    from torch.utils.data import WeightedRandomSampler

    counts = dataset.class_counts()
    weights = [1.0 / counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(config: ModelConfig, from_saved: bool = False):
    """Load model + processor. If from_saved, load from local directory."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if from_saved:
        model_path = str(MODELS_DIR / config.short_name)
        print(f"  Loading saved model from {model_path}")
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
        return model, processor

    print(f"  Loading pretrained: {config.model_id}")
    processor = AutoImageProcessor.from_pretrained(config.model_id)

    if config.needs_head_replace:
        # Replace ImageNet 1000-class head with 7-class head
        model = AutoModelForImageClassification.from_pretrained(
            config.model_id,
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(config.model_id)

    return model, processor


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_model(
    config: ModelConfig,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
) -> dict:
    """Fine-tune a single model. Returns dict with training metrics."""
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print(f"Training: {config.display_name}")
    print(f"{'='*60}")

    model, processor = load_model(config, from_saved=False)
    model.to(device)

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} total, {trainable_count:,} trainable")

    # Datasets (each model uses its own processor for correct normalization)
    SkinDataset = build_dataset_class()
    train_ds = SkinDataset(DATA_DIR / "train", processor)
    val_ds   = SkinDataset(DATA_DIR / "val",   processor)
    print(f"  Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # Class weights for loss
    train_counts = train_ds.class_counts()
    total = sum(train_counts)
    n_cls = len(train_counts)
    class_weights = torch.tensor(
        [total / (n_cls * c) if c > 0 else 0.0 for c in train_counts],
        dtype=torch.float32,
    ).to(device)

    # DataLoaders
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=(device.type != "mps"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

    # MPS warmup — run one dummy forward+backward to compile Metal shaders
    if device.type == "mps":
        print("  MPS warmup (compiling Metal shaders)...", flush=True)
        dummy = next(iter(train_loader))
        dummy_pv = dummy["pixel_values"].to(device)
        dummy_lb = dummy["labels"].to(device)
        with torch.no_grad():
            _ = model(pixel_values=dummy_pv)
        torch.mps.synchronize()
        del dummy, dummy_pv, dummy_lb
        print("  MPS warmup done.", flush=True)

    # Training loop
    best_val_acc = 0.0
    save_dir = MODELS_DIR / config.short_name
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Sync MPS to prevent command queue buildup
            if device.type == "mps" and step % 10 == 0:
                torch.mps.synchronize()

            train_loss += loss.item()
            if step % 50 == 0:
                print(f"    Epoch {epoch} step {step}/{len(train_loader)}  "
                      f"loss={train_loss/step:.4f}", flush=True)

        avg_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        correct = total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels       = batch["labels"].to(device)
                logits = model(pixel_values=pixel_values).logits
                preds  = logits.argmax(dim=-1)
                correct   += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct / total_val
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.1%}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_dir))
            processor.save_pretrained(str(save_dir))
            print(f"    ✓ New best ({val_acc:.1%}) — saved to {save_dir}")

    train_time = time.perf_counter() - train_start

    print(f"  Done. Best val acc: {best_val_acc:.1%} in {train_time:.0f}s")

    # Free GPU memory before next model
    del model, optimizer, scheduler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best_val_acc": best_val_acc,
        "train_time_s": round(train_time, 1),
        "param_count": param_count,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_one_model(config: ModelConfig, test_dir: Path, device) -> dict:
    """Evaluate a trained model on the test set. Returns metrics dict."""
    import torch
    from PIL import Image

    print(f"\n{'='*60}")
    print(f"Evaluating: {config.display_name}")
    print(f"{'='*60}")

    model_path = MODELS_DIR / config.short_name
    if not model_path.exists():
        print(f"  WARNING: No saved model at {model_path}, skipping.")
        return {}

    model, processor = load_model(config, from_saved=True)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    # Collect test images
    test_samples: list[tuple[Path, int]] = []
    for short_code, (label_idx, _) in HAM_CLASSES.items():
        cls_dir = test_dir / short_code
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.glob("*.jpg")):
            test_samples.append((img_path, label_idx))

    if not test_samples:
        print(f"  No test images found in {test_dir}")
        return {}

    print(f"  Test images: {len(test_samples)}")

    # Run inference
    correct_top1 = 0
    correct_top3 = 0
    per_class_correct = Counter()
    per_class_total   = Counter()
    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for img_path, label in test_samples:
            img = Image.open(img_path).convert("RGB")
            enc = processor(images=img, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device)

            t0 = time.perf_counter()
            logits = model(pixel_values=pixel_values).logits
            inference_times.append((time.perf_counter() - t0) * 1000)  # ms

            # Top-1
            pred = logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_labels.append(label)

            if pred == label:
                correct_top1 += 1
                per_class_correct[label] += 1
            per_class_total[label] += 1

            # Top-3
            top3 = logits.topk(min(3, NUM_CLASSES), dim=-1).indices[0].tolist()
            if label in top3:
                correct_top3 += 1

    top1_acc = correct_top1 / len(test_samples)
    top3_acc = correct_top3 / len(test_samples)
    avg_infer_ms = np.mean(inference_times)

    print(f"  Top-1 accuracy: {top1_acc:.1%}")
    print(f"  Top-3 accuracy: {top3_acc:.1%}")
    print(f"  Avg inference:  {avg_infer_ms:.1f} ms/image")

    # Per-class accuracy
    per_class_acc = {}
    for short_code, (idx, name) in HAM_CLASSES.items():
        t = per_class_total[idx]
        c = per_class_correct[idx]
        acc = c / t if t > 0 else 0.0
        per_class_acc[name] = acc
        print(f"    {name:<25} {c:>3}/{t:<3}  {acc:.1%}")

    # Confusion matrix (as nested list)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    # Free GPU memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "avg_infer_ms": round(avg_infer_ms, 1),
        "param_count": param_count,
        "per_class_acc": per_class_acc,
        "confusion": confusion.tolist(),
        "n_test": len(test_samples),
    }


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    results: dict[str, dict],
    train_metrics: dict[str, dict],
) -> None:
    """Generate text and CSV comparison reports."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Merge train + eval metrics
    merged = {}
    for config in MODEL_CONFIGS:
        name = config.short_name
        if name not in results or not results[name]:
            continue
        merged[name] = {
            "display_name": config.display_name,
            **results[name],
            "train_time_s": train_metrics.get(name, {}).get("train_time_s", "N/A"),
            "best_val_acc": train_metrics.get(name, {}).get("best_val_acc", "N/A"),
        }

    if not merged:
        print("No results to report.")
        return

    # Determine test set size
    n_test = next(iter(merged.values())).get("n_test", "?")

    # ── Text report ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 65)
    lines.append("  DermAnnotate — ViT Model Comparison Report")
    lines.append("=" * 65)
    lines.append(f"Test set: {n_test} images ({n_test // NUM_CLASSES} per class)")
    lines.append("")

    # Summary table
    hdr = f"{'Model':<20} {'Params':>8} {'Top-1':>7} {'Top-3':>7} {'Train(s)':>9} {'Infer(ms)':>10}"
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    for name, m in merged.items():
        params_str = f"{m['param_count']/1e6:.1f}M"
        top1_str = f"{m['top1_acc']:.1%}"
        top3_str = f"{m['top3_acc']:.1%}"
        train_str = str(m['train_time_s']) if m['train_time_s'] != "N/A" else "N/A"
        infer_str = f"{m['avg_infer_ms']:.1f}"
        lines.append(
            f"{m['display_name']:<20} {params_str:>8} {top1_str:>7} {top3_str:>7} {train_str:>9} {infer_str:>10}"
        )

    lines.append(sep)
    lines.append("")

    # Per-class comparison
    lines.append("Per-Class Top-1 Accuracy:")
    lines.append("")
    class_names = list(ID2LABEL.values())
    model_names = [m["display_name"] for m in merged.values()]

    # Header
    cls_hdr = f"{'Class':<25}" + "".join(f" {n:>15}" for n in model_names)
    lines.append(cls_hdr)
    lines.append("-" * len(cls_hdr))

    for cls_name in class_names:
        row = f"{cls_name:<25}"
        for m in merged.values():
            acc = m.get("per_class_acc", {}).get(cls_name, 0.0)
            row += f" {acc:>14.1%}"
        lines.append(row)

    lines.append("")

    # Recommendation
    best_acc_name = max(merged, key=lambda k: merged[k]["top1_acc"])
    best_ratio_name = max(merged, key=lambda k: merged[k]["top1_acc"] / (merged[k]["param_count"] / 1e6))

    lines.append("Recommendation:")
    b = merged[best_acc_name]
    lines.append(f"  Best overall accuracy: {b['display_name']} ({b['top1_acc']:.1%} top-1)")
    r = merged[best_ratio_name]
    ratio = r["top1_acc"] / (r["param_count"] / 1e6)
    lines.append(f"  Best accuracy/size:    {r['display_name']} ({r['top1_acc']:.1%} / {r['param_count']/1e6:.1f}M = {ratio:.4f} acc/M)")
    lines.append("")
    lines.append("=" * 65)

    report_text = "\n".join(lines)

    # Write text report
    txt_path = RESULTS_DIR / "model_comparison.txt"
    txt_path.write_text(report_text)
    print(f"\nText report saved to {txt_path}")
    print(report_text)

    # ── CSV report ───────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "model_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["model", "params", "top1_acc", "top3_acc", "train_time_s", "avg_infer_ms"]
        header += [f"class_{cls}" for cls in class_names]
        writer.writerow(header)
        # Rows
        for name, m in merged.items():
            row = [
                m["display_name"],
                m["param_count"],
                round(m["top1_acc"], 4),
                round(m["top3_acc"], 4),
                m["train_time_s"],
                m["avg_infer_ms"],
            ]
            for cls_name in class_names:
                row.append(round(m.get("per_class_acc", {}).get(cls_name, 0.0), 4))
            writer.writerow(row)

    print(f"CSV report saved to {csv_path}")


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Vision Transformer architectures on HAM10000."
    )
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading training data")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only evaluate and generate report")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(CONFIG_BY_NAME.keys()),
                        help="Which models to train/evaluate (default: all)")
    args = parser.parse_args()

    # Select models
    if args.models:
        configs = [CONFIG_BY_NAME[m] for m in args.models]
    else:
        configs = MODEL_CONFIGS

    model_names = [c.display_name for c in configs]
    print(f"Models to compare: {', '.join(model_names)}")

    # Device selection
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Download data
    if not args.skip_download and not args.skip_training:
        download_splits(["train", "val"])

    # Train models sequentially
    train_metrics: dict[str, dict] = {}
    if not args.skip_training:
        for config in configs:
            metrics = train_one_model(
                config,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
            )
            train_metrics[config.short_name] = metrics

    # Evaluate all models on test set
    if not TEST_DIR.exists():
        print(f"\nWARNING: Test directory {TEST_DIR} not found.")
        print("Run `python scripts/download_ham10000_test.py` first.")
        if train_metrics:
            print("\nTraining completed but cannot evaluate without test data.")
        return

    eval_results: dict[str, dict] = {}
    for config in configs:
        result = evaluate_one_model(config, TEST_DIR, device)
        if result:
            eval_results[config.short_name] = result

    # Generate comparison report
    if eval_results:
        generate_report(eval_results, train_metrics)
    else:
        print("\nNo evaluation results to report.")


if __name__ == "__main__":
    main()
