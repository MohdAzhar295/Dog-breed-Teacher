from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.config import (
    CLASSES_PATH,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_CHECKPOINT_PATH,
    TOP_K,
)


def build_model(num_classes: int) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class BreedClassifier:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = _build_transform()
        self.model: nn.Module | None = None
        self.class_names: list[str] = []
        self._load()

    def _load(self) -> None:
        if MODEL_CHECKPOINT_PATH.exists():
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=self.device)
            self.class_names = checkpoint["class_names"]
            model = build_model(len(self.class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.model = model
            return

        if CLASSES_PATH.exists():
            with CLASSES_PATH.open(encoding="utf-8") as handle:
                self.class_names = json.load(handle)
        else:
            self.class_names = []

    @property
    def is_ready(self) -> bool:
        return self.model is not None and bool(self.class_names)

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError(
                "Model checkpoint not found. Run scripts/download_dataset.py, "
                "scripts/prepare_dataset.py, and scripts/train.py first."
            )

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_values, top_indices = torch.topk(probabilities, k=min(TOP_K, len(self.class_names)))
        top_k = [
            {
                "breed_id": self.class_names[index.item()],
                "confidence": round(top_values[position].item(), 4),
            }
            for position, index in enumerate(top_indices)
        ]

        best = top_k[0]
        is_confident = best["confidence"] >= CONFIDENCE_THRESHOLD

        return {
            "predicted_breed": best["breed_id"] if is_confident else "unknown",
            "confidence": best["confidence"],
            "is_confident": is_confident,
            "top_k": top_k,
            "supported_breeds": self.class_names,
        }
