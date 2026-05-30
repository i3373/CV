from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

try:
    from sklearn.model_selection import train_test_split
except ImportError as exc:
    raise ImportError("Install scikit-learn: pip install scikit-learn") from exc


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _resolve_data_root(root_dir: str | Path) -> Path:
    root = Path(root_dir)
    if root.exists() or root.is_absolute():
        return root

    candidate = Path(__file__).resolve().parent / root
    if candidate.exists():
        return candidate

    candidate = Path(__file__).resolve().parent.parent / root
    if candidate.exists():
        return candidate

    return root


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def parse_color_from_filename(filename: str) -> str:
    parts = Path(filename).name.split("$$")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse color from '{filename}': expected at least 4 '$$' tokens.")

    color = parts[3].strip()
    if not color:
        raise ValueError(f"Empty color label in filename '{filename}'.")
    return color


def build_samples(root_dir: str | Path) -> List[Tuple[str, str]]:
    root = _resolve_data_root(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    samples: List[Tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_image(path):
            continue

        try:
            color = parse_color_from_filename(path.name)
        except ValueError:
            continue
        samples.append((str(path), color))

    if not samples:
        raise RuntimeError(
            f"No DVM images found in {root}. Check that the DVM frontal views are unpacked."
        )
    return samples


def filter_samples(
    samples: Sequence[Tuple[str, str]],
    exclude_classes: Sequence[str] | None = None,
) -> List[Tuple[str, str]]:
    excluded = {label.strip() for label in (exclude_classes or []) if label.strip()}
    if not excluded:
        return list(samples)

    filtered = [(path, label) for path, label in samples if label not in excluded]
    removed = len(samples) - len(filtered)
    if removed:
        print(f"[filter_samples] Removed {removed} samples from excluded classes: {sorted(excluded)}")
    return filtered


@dataclass(frozen=True)
class SplitResult:
    train: List[Tuple[str, str]]
    val: List[Tuple[str, str]]
    test: List[Tuple[str, str]]
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]


def make_splits(
    samples: Sequence[Tuple[str, str]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    min_count_per_class: int = 50,
) -> SplitResult:
    from collections import Counter

    counts = Counter(label for _, label in samples)
    filtered = [(path, label) for path, label in samples if counts[label] >= min_count_per_class]
    if not filtered:
        raise RuntimeError(
            "All classes were filtered out. Lower MIN_COUNT_PER_CLASS in config.py."
        )

    removed = len(samples) - len(filtered)
    if removed:
        print(f"[make_splits] Removed {removed} samples from rare classes (<{min_count_per_class}).")

    x_all = [path for path, _ in filtered]
    y_all = [label for _, label in filtered]

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x_all,
        y_all,
        test_size=test_size,
        random_state=seed,
        stratify=y_all,
    )

    val_ratio_in_trainval = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_ratio_in_trainval,
        random_state=seed,
        stratify=y_trainval,
    )

    classes = sorted(set(y_train))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    return SplitResult(
        train=list(zip(x_train, y_train)),
        val=list(zip(x_val, y_val)),
        test=list(zip(x_test, y_test)),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
    )


class DVMColorDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, str]],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
    ):
        self.samples = list(samples)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform

        missing = {label for _, label in self.samples if label not in self.class_to_idx}
        if missing:
            raise ValueError(f"Labels missing from class_to_idx: {sorted(missing)[:20]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        target = self.class_to_idx[label]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.long)
