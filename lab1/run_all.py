import json
from pathlib import Path

from config import (
    ARCHS,
    BATCH_SIZE,
    DATA_ROOT,
    EPOCHS_FT,
    EPOCHS_SCRATCH,
    EVAL_TTA_HFLIP,
    LR_FT,
    LR_SCRATCH,
    NUM_WORKERS,
    OUT_DIR,
)
from eval import evaluate_checkpoint
from train import train_model


def main() -> None:
    out_dir = Path(OUT_DIR)
    comparison = []

    for arch in ARCHS:
        epochs = EPOCHS_SCRATCH if arch == "resnet50_scratch" else EPOCHS_FT
        lr = LR_SCRATCH if arch == "resnet50_scratch" else LR_FT

        print("\n" + "=" * 80)
        print(f"Training {arch}: epochs={epochs}, lr={lr}")
        print("=" * 80)

        ckpt = train_model(arch=arch, epochs=epochs, lr=lr, out_dir=out_dir)
        metrics = evaluate_checkpoint(
            ckpt,
            data_root=DATA_ROOT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            tta_hflip=EVAL_TTA_HFLIP,
        )
        comparison.append(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nComparison:")
    for row in sorted(comparison, key=lambda item: item["f1_macro"], reverse=True):
        print(f"{row['arch']}: F1_macro={row['f1_macro']:.4f}, accuracy={row['accuracy']:.4f}")


if __name__ == "__main__":
    main()
