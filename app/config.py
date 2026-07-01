from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"
RAW_DATA_DIR = DATA_DIR / "raw" / "stanford_dogs"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SELECTED_BREEDS_PATH = DATA_DIR / "selected_breeds.json"
BREED_PROFILES_PATH = DATA_DIR / "breed_profiles.json"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "dog_breed_model.pt"
CLASSES_PATH = MODELS_DIR / "classes.json"

CONFIDENCE_THRESHOLD = 0.55
TOP_K = 3
IMAGE_SIZE = 224

STANFORD_BASE_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs"
STANFORD_FILES = {
    "images.tar": f"{STANFORD_BASE_URL}/images.tar",
    "annotation.tar": f"{STANFORD_BASE_URL}/annotation.tar",
    "lists.tar": f"{STANFORD_BASE_URL}/lists.tar",
}


def load_selected_breeds() -> list[dict]:
    with SELECTED_BREEDS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["breeds"]
