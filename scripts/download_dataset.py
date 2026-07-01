"""Download Stanford Dogs dataset archives."""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import RAW_DATA_DIR, STANFORD_FILES


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination.name}")
        return

    print(f"Downloading {destination.name} ...")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved {destination}")


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    marker = target_dir / f".extracted_{archive_path.name}"
    if marker.exists():
        print(f"Already extracted: {archive_path.name}")
        return

    print(f"Extracting {archive_path.name} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=target_dir, filter="data")
    marker.write_text("ok", encoding="utf-8")
    print(f"Extracted to {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Stanford Dogs dataset")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in STANFORD_FILES.items():
        archive_path = RAW_DATA_DIR / filename
        download_file(url, archive_path)
        if not args.skip_extract:
            extract_archive(archive_path, RAW_DATA_DIR)


if __name__ == "__main__":
    main()
