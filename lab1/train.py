import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import (
    BATCH_SIZE,
    CLASS_WEIGHTING,
    COLOR_JITTER,
    DATA_ROOT,
    EPOCHS_FT,
    EPOCHS_SCRATCH,
    EXCLUDE_CLASSES,
    FREEZE_BACKBONE_EPOCHS,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LR_FT,
    LR_SCRATCH,
    MIN_COUNT_PER_CLASS,
    NUM_WORKERS,
    OUT_DIR,
    SAMPLER,
    SEED,
    TEST_SIZE,
    TRAIN_ARCH,
    USE_AMP,
    VAL_SIZE,
    WEIGHT_DECAY,
)
from dataset import DVMColorDataset, build_samples, filter_samples, make_splits
from model_factory import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_class_weights(samples, class_to_idx: dict[str, int]) -> torch.Tensor:
    counts = Counter(label for _, label in samples)
    weights = torch.ones(len(class_to_idx), dtype=torch.float32)
    total = sum(counts.values())

    for label, idx in class_to_idx.items():
        weights[idx] = total / max(1, counts[label])

    return weights / weights.mean()


def make_effective_class_weights(
    samples,
    class_to_idx: dict[str, int],
    beta: float = 0.999,
) -> torch.Tensor:
    counts = Counter(label for _, label in samples)
    weights = torch.ones(len(class_to_idx), dtype=torch.float32)

    for label, idx in class_to_idx.items():
        count = max(1, counts[label])
        weights[idx] = (1.0 - beta) / (1.0 - beta**count)

    return weights / weights.mean()


def make_sample_weights(samples, mode: str) -> list[float]:
    counts = Counter(label for _, label in samples)
    weights = []

    for _, label in samples:
        count = max(1, counts[label])
        if mode == "weighted_sqrt":
            weights.append(1.0 / (count**0.5))
        elif mode == "weighted":
            weights.append(1.0 / count)
        else:
            raise ValueError(f"Unknown sampler mode: {mode}")

    return weights


def freeze_backbone(model: nn.Module, arch: str) -> None:
    if arch == "resnet50_scratch":
        return

    for name, param in model.named_parameters():
        train_head = (
            name.startswith("fc.")
            or name.startswith("classifier.")
            or name.startswith("classifier")
        )
        param.requires_grad = train_head


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


@torch.no_grad()
def evaluate_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds_all = []
    targets_all = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(targets.tolist())

    return {
        "accuracy": accuracy_score(targets_all, preds_all),
        "f1_macro": f1_score(targets_all, preds_all, average="macro", zero_division=0),
    }


def write_history(history: list[dict[str, float]], out_dir: Path) -> None:
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not history:
        return

    with (out_dir / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def build_transforms(img_size: int):
    train_transforms = [
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if COLOR_JITTER:
        train_transforms.append(transforms.ColorJitter(brightness=0.10, contrast=0.10))

    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    eval_transforms = [
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(eval_transforms)


def train_model(
    arch: str,
    epochs: int | None = None,
    lr: float | None = None,
    out_dir: str | Path = OUT_DIR,
) -> Path:
    epochs = epochs if epochs is not None else (EPOCHS_SCRATCH if arch == "resnet50_scratch" else EPOCHS_FT)
    lr = lr if lr is not None else (LR_SCRATCH if arch == "resnet50_scratch" else LR_FT)

    set_seed(SEED)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    run_dir = Path(out_dir) / arch
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = filter_samples(build_samples(DATA_ROOT), EXCLUDE_CLASSES)
    split = make_splits(
        samples,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=SEED,
        min_count_per_class=MIN_COUNT_PER_CLASS,
    )
    num_classes = len(split.class_to_idx)
    print("Classes:", num_classes)
    print("Train/Val/Test:", len(split.train), len(split.val), len(split.test))

    train_tfms, eval_tfms = build_transforms(IMG_SIZE)
    train_ds = DVMColorDataset(split.train, split.class_to_idx, transform=train_tfms)
    val_ds = DVMColorDataset(split.val, split.class_to_idx, transform=eval_tfms)

    pin_memory = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(SEED)

    sampler = None
    shuffle = True
    if SAMPLER != "shuffle":
        sampler = WeightedRandomSampler(
            weights=make_sample_weights(split.train, SAMPLER),
            num_samples=len(split.train),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    pretrained = arch != "resnet50_scratch"
    model = build_model(arch, num_classes=num_classes, pretrained=pretrained).to(device)

    class_weights = None
    if CLASS_WEIGHTING == "inverse":
        class_weights = make_class_weights(split.train, split.class_to_idx).to(device)
    elif CLASS_WEIGHTING == "sqrt":
        class_weights = torch.sqrt(make_class_weights(split.train, split.class_to_idx)).to(device)
        class_weights = class_weights / class_weights.mean()
    elif CLASS_WEIGHTING == "effective":
        class_weights = make_effective_class_weights(split.train, split.class_to_idx).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    if pretrained and FREEZE_BACKBONE_EPOCHS > 0:
        freeze_backbone(model, arch)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    meta = {
        "arch": arch,
        "num_classes": num_classes,
        "class_to_idx": split.class_to_idx,
        "idx_to_class": split.idx_to_class,
        "data_root": str(DATA_ROOT),
        "img_size": IMG_SIZE,
        "seed": SEED,
        "min_count_per_class": MIN_COUNT_PER_CLASS,
        "exclude_classes": EXCLUDE_CLASSES,
        "class_weighting": CLASS_WEIGHTING,
        "sampler": SAMPLER,
        "label_smoothing": LABEL_SMOOTHING,
        "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
        "color_jitter": COLOR_JITTER,
        "pretrained": pretrained,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_val_f1 = -1.0
    history: list[dict[str, float]] = []
    best_path = run_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        if pretrained and FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS + 1:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
            remaining_epochs = max(1, epochs - FREEZE_BACKBONE_EPOCHS)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

        model.train()
        started = time.time()
        total_loss = 0.0
        seen = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * targets.size(0)
            seen += targets.size(0)

        scheduler.step()
        train_loss = total_loss / max(1, seen)
        val_metrics = evaluate_on_loader(model, val_loader, device)
        elapsed = time.time() - started

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": elapsed,
        }
        history.append(row)
        write_history(history, run_dir)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save({"model": model.state_dict(), "meta": meta}, best_path)
            print(f"  saved: {best_path}")

    print("Done. Best val_f1_macro:", best_val_f1)
    return best_path


def main() -> None:
    train_model(TRAIN_ARCH)


if __name__ == "__main__":
    main()
