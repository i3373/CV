from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (
    DATA_ROOT,
    EXAMPLE_BATCH_SIZE,
    EXAMPLE_NUM_WORKERS,
    EXCLUDE_CLASSES,
    IMG_SIZE,
    MIN_COUNT_PER_CLASS,
    SAMPLE_BATCH_PATH,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
)
from dataset import DVMColorDataset, build_samples, filter_samples, make_splits


def save_contact_sheet(batch_samples, split, out_path: str | Path, thumb_size: int = 180) -> None:
    columns = 4
    rows = (len(batch_samples) + columns - 1) // columns
    card_w = thumb_size + 24
    card_h = thumb_size + 44

    sheet = Image.new("RGB", (columns * card_w, rows * card_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i, (image_path, color_name) in enumerate(batch_samples):
        col = i % columns
        row = i // columns
        x0 = col * card_w
        y0 = row * card_h

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_size, thumb_size))
            image_x = x0 + (card_w - image.width) // 2
            image_y = y0 + 10
            sheet.paste(image, (image_x, image_y))

        class_idx = split.class_to_idx[color_name]
        draw.text((x0 + 8, y0 + thumb_size + 16), f"{i}: {color_name} ({class_idx})", fill=(0, 0, 0), font=font)
        draw.rectangle((x0, y0, x0 + card_w - 1, y0 + card_h - 1), outline=(220, 220, 220))

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    samples = filter_samples(build_samples(DATA_ROOT), EXCLUDE_CLASSES)
    split = make_splits(
        samples,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=SEED,
        min_count_per_class=MIN_COUNT_PER_CLASS,
    )

    print("Classes:", len(split.class_to_idx))
    print("Train/Val/Test:", len(split.train), len(split.val), len(split.test))
    print("Class mapping:", split.class_to_idx)

    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    batch_samples = split.train[:EXAMPLE_BATCH_SIZE]
    dataset = DVMColorDataset(batch_samples, split.class_to_idx, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=EXAMPLE_BATCH_SIZE,
        shuffle=False,
        num_workers=EXAMPLE_NUM_WORKERS,
    )

    images, labels = next(iter(loader))
    idx_to_class = {idx: label for label, idx in split.class_to_idx.items()}

    print("Batch images:", tuple(images.shape), images.dtype)
    print("Batch labels:", tuple(labels.shape), labels.dtype, "min/max:", labels.min().item(), labels.max().item())
    print("Batch colors:", [idx_to_class[int(label)] for label in labels])

    save_contact_sheet(batch_samples, split, SAMPLE_BATCH_PATH)
    print("Saved batch preview:", SAMPLE_BATCH_PATH)
    print("OK")


if __name__ == "__main__":
    main()
