"""Train a transfer-learning classifier for 30 dog breeds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import CLASSES_PATH, MODEL_CHECKPOINT_PATH, MODELS_DIR, PROCESSED_DATA_DIR
from app.model import build_model


def build_transforms(image_size: int = 224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_model(epochs: int, batch_size: int, learning_rate: float, num_workers: int) -> None:
    if not PROCESSED_DATA_DIR.exists():
        raise FileNotFoundError("Processed dataset not found. Run scripts/prepare_dataset.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfms, eval_tfms = build_transforms()
    train_ds = datasets.ImageFolder(PROCESSED_DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(PROCESSED_DATA_DIR / "val", transform=eval_tfms)

    class_names = train_ds.classes
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        elapsed = time.time() - start
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": elapsed,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
            f"({elapsed:.1f}s)"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "architecture": "mobilenet_v3_small",
                    "val_acc": val_acc,
                },
                MODEL_CHECKPOINT_PATH,
            )

    with CLASSES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)

    history_path = MODELS_DIR / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Saved checkpoint to {MODEL_CHECKPOINT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dog breed classifier")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
