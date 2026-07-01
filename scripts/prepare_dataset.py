"""Prepare a 30-breed subset from Stanford Dogs with train/val/test splits."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import (
    CLASSES_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SELECTED_BREEDS_PATH,
    load_selected_breeds,
)


def find_images_root(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "Images",
        raw_dir / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate Images/ directory in raw dataset")


def find_lists_dir(raw_dir: Path) -> Path:
    for candidate in [raw_dir, raw_dir / "lists"]:
        train_list = candidate / "train_list.mat"
        if train_list.exists():
            return candidate
    raise FileNotFoundError("Could not locate train_list.mat")


def resolve_breed_folder(images_root: Path, suffix: str) -> Path:
    matches = sorted(images_root.glob(f"*-{suffix}"))
    if not matches:
        matches = sorted(images_root.glob(f"*{suffix}*"))
    if not matches:
        raise FileNotFoundError(f"No folder found for breed suffix: {suffix}")
    return matches[0]


def load_split_lists(lists_dir: Path) -> tuple[set[str], set[str]]:
    train_mat = scipy.io.loadmat(lists_dir / "train_list.mat")
    test_mat = scipy.io.loadmat(lists_dir / "test_list.mat")

    train_files = {row[0][0] for row in train_mat["file_list"]}
    test_files = {row[0][0] for row in test_mat["file_list"]}
    return train_files, test_files


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)


def prepare_dataset(val_ratio: float = 0.12, seed: int = 42) -> None:
    breeds = load_selected_breeds()
    images_root = find_images_root(RAW_DATA_DIR)
    lists_dir = find_lists_dir(RAW_DATA_DIR)
    train_files, test_files = load_split_lists(lists_dir)

    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    random.seed(seed)
    class_names: list[str] = []
    stats: dict[str, dict[str, int]] = {}

    for breed in breeds:
        breed_id = breed["id"]
        suffix = breed["stanford_suffix"]
        folder = resolve_breed_folder(images_root, suffix)
        class_names.append(breed_id)

        train_candidates: list[Path] = []
        test_candidates: list[Path] = []

        for image_path in sorted(folder.glob("*.jpg")):
            relative_name = f"{folder.name}/{image_path.name}"
            if relative_name in test_files:
                test_candidates.append(image_path)
            elif relative_name in train_files:
                train_candidates.append(image_path)
            else:
                train_candidates.append(image_path)

        random.shuffle(train_candidates)
        val_count = max(1, int(len(train_candidates) * val_ratio))
        val_candidates = train_candidates[:val_count]
        remaining_train = train_candidates[val_count:]

        stats[breed_id] = {
            "train": len(remaining_train),
            "val": len(val_candidates),
            "test": len(test_candidates),
        }

        for split_name, items in [
            ("train", remaining_train),
            ("val", val_candidates),
            ("test", test_candidates),
        ]:
            for image_path in items:
                destination = PROCESSED_DATA_DIR / split_name / breed_id / image_path.name
                copy_image(image_path, destination)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with CLASSES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)

    summary_path = PROCESSED_DATA_DIR / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"breeds": len(class_names), "splits": stats}, handle, indent=2)

    print(f"Prepared dataset at {PROCESSED_DATA_DIR}")
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 30-breed Stanford Dogs subset")
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
