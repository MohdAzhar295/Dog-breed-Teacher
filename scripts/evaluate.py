"""Evaluate trained model on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import MODEL_CHECKPOINT_PATH, MODELS_DIR, PROCESSED_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dog breed classifier")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not MODEL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError("Model checkpoint not found. Train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_ds = datasets.ImageFolder(PROCESSED_DATA_DIR / "test", transform=eval_tfms)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    from app.model import build_model

    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.int32)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for truth, pred in zip(labels.cpu(), preds.cpu()):
                confusion[truth, pred] += 1

    accuracy = correct / max(total, 1)
    report = {
        "test_accuracy": accuracy,
        "num_samples": total,
        "class_names": class_names,
        "confusion_matrix": confusion.tolist(),
    }

    report_path = MODELS_DIR / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
