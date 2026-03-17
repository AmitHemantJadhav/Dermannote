"""
Download a sample of the HAM10000 test split from HuggingFace.

Source: hawking32/ham10000_ttv (public, CC BY-NC 4.0)
Downloads N images per class into dataset/ham10000_test/<class>/<image>.jpg

Usage:
  python scripts/download_ham10000_test.py              # 25 per class (175 total)
  python scripts/download_ham10000_test.py --per-class 50
  python scripts/download_ham10000_test.py --all         # full test split (1170 images)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_REPO = "hawking32/ham10000_ttv"
DEFAULT_PER_CLASS = 25
OUT_DIR = Path(__file__).parent.parent / "dataset" / "ham10000_test"

# All 7 HAM10000 class short-codes used as directory names in the dataset
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def download(per_class: int | None) -> None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Listing files in {DATASET_REPO}...")
    all_files = list(list_repo_files(DATASET_REPO, repo_type="dataset"))

    # Group test files by class
    by_class: dict[str, list[str]] = {cls: [] for cls in HAM_CLASSES}
    for f in all_files:
        parts = f.split("/")
        if len(parts) == 3 and parts[0] == "test" and parts[1] in HAM_CLASSES and f.endswith(".jpg"):
            by_class[parts[1]].append(f)

    total_to_download = sum(
        (len(files) if per_class is None else min(len(files), per_class))
        for files in by_class.values()
    )

    print(f"\nDownloading {total_to_download} images → {OUT_DIR}\n")

    downloaded = 0
    for cls, files in sorted(by_class.items()):
        subset = files if per_class is None else files[:per_class]
        cls_dir = OUT_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for repo_path in subset:
            filename = Path(repo_path).name
            dest = cls_dir / filename
            if dest.exists():
                downloaded += 1
                continue
            try:
                local = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=str(OUT_DIR.parent / "_hf_cache"),
                )
                # Copy to our organised directory
                import shutil
                shutil.copy2(local, dest)
                downloaded += 1
                print(f"  [{downloaded}/{total_to_download}] {cls}/{filename}")
            except Exception as exc:
                print(f"  SKIP {repo_path}: {exc}")

    print(f"\nDone. {downloaded} images saved to: {OUT_DIR}")
    print("\nRun evaluation with:")
    print(f"  python scripts/evaluate.py --dir {OUT_DIR} --real --confusion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HAM10000 test images from HuggingFace.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--per-class", type=int, default=DEFAULT_PER_CLASS, metavar="N",
        help=f"Images per class to download (default: {DEFAULT_PER_CLASS})",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Download the full test split (1170 images)",
    )
    args = parser.parse_args()

    per_class = None if args.all else args.per_class
    download(per_class)


if __name__ == "__main__":
    main()
