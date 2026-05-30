import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lab4lib.utils import ensure_dir, project_root, seed_everything, save_json, write_csv_rows


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROOT = project_root()
RAW_DIR = ROOT / "img_align_celeba" / "img_align_celeba"
OUTPUT_DIR = Path("data/processed")
IMAGE_SIZE = 64
LIMIT = 0
SEED = 42
MARGIN = 0.45
VAL_RATIO = 0.1
TEST_RATIO = 0.1
CONDITION_COL = "auto"
USE_EVAL_PARTITION = True


def find_image_dir(root):
    counts = Counter()
    ignored_parts = {"data", "runs", "figures", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        counts[path.parent] += 1
    if not counts:
        raise FileNotFoundError(f"No images found below {root}")
    return counts.most_common(1)[0][0]


def find_attr_file(root):
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".txt"}:
            continue
        name = path.name.lower()
        if "attr" in name and "celeba" in name:
            candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def parse_attr_file(path):
    if path is None:
        return {}, []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = {}
            columns = [c for c in (reader.fieldnames or []) if c not in {"image_id", "image"}]
            for row in reader:
                image_id = row.get("image_id") or row.get("image")
                if not image_id:
                    continue
                rows[image_id] = {c: int(float(row[c])) for c in columns if row.get(c, "") != ""}
        return rows, columns

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 3:
        return {}, []
    columns = lines[1].split()
    attrs = {}
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < len(columns) + 1:
            continue
        attrs[parts[0]] = {col: int(val) for col, val in zip(columns, parts[1:])}
    return attrs, columns


def find_eval_partition_file(root):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".txt"}:
            continue
        if "eval_partition" in path.name.lower():
            return path
    return None


def parse_eval_partition(path):
    if path is None:
        return {}
    split_map = {"0": "train", "1": "val", "2": "test", 0: "train", 1: "val", 2: "test"}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            result = {}
            for row in reader:
                image_id = row.get("image_id")
                partition = row.get("partition")
                if image_id is not None and partition is not None:
                    result[image_id] = split_map.get(partition, str(partition))
            return result

    result = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() != "image_id":
            result[parts[0]] = split_map.get(parts[1], parts[1])
    return result


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def center_square_crop(image):
    h, w = image.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def crop_with_face_detector(image, detector, margin):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 35))
    if len(faces) == 0:
        crop, box = center_square_crop(image)
        return crop, False, box

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx, cy = x + w / 2.0, y + h / 2.0
    side = int(max(w, h) * (1.0 + margin))
    h_img, w_img = image.shape[:2]
    x1 = max(0, int(cx - side / 2.0))
    y1 = max(0, int(cy - side / 2.0))
    x2 = min(w_img, int(cx + side / 2.0))
    y2 = min(h_img, int(cy + side / 2.0))

    # Make the detector crop square after clipping to image borders.
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w != crop_h:
        side2 = min(crop_w, crop_h)
        x1 = x1 + max(0, crop_w - side2) // 2
        y1 = y1 + max(0, crop_h - side2) // 2
        x2 = x1 + side2
        y2 = y1 + side2
    return image[y1:y2, x1:x2], True, (x1, y1, x2, y2)


def main():
    root = ROOT.resolve()
    seed_everything(SEED)
    raw_dir = RAW_DIR.resolve() if RAW_DIR is not None else find_image_dir(root)
    out_dir = ensure_dir(root / OUTPUT_DIR)
    faces_dir = ensure_dir(out_dir / f"faces{IMAGE_SIZE}")

    attr_file = find_attr_file(root)
    eval_partition_file = find_eval_partition_file(root) if USE_EVAL_PARTITION else None
    attrs, attr_columns = parse_attr_file(attr_file)
    eval_partition = parse_eval_partition(eval_partition_file)
    has_male = "Male" in attr_columns

    detector_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(detector_path))
    if detector.empty():
        raise RuntimeError(f"Cannot load OpenCV Haar cascade from {detector_path}")

    image_paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if LIMIT and LIMIT > 0:
        image_paths = image_paths[:LIMIT]

    rows = []
    skipped = 0
    for src in tqdm(image_paths, desc="preprocess"):
        image = read_image(src)
        if image is None:
            skipped += 1
            continue
        crop, detected, box = crop_with_face_detector(image, detector, margin=MARGIN)
        if crop.size == 0:
            skipped += 1
            continue
        resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        dst = faces_dir / src.name
        if not write_image(dst, resized):
            skipped += 1
            continue

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean() / 255.0)
        attr_values = {col: "" for col in attr_columns}
        if src.name in attrs:
            attr_values = {col: 1 if int(attrs[src.name][col]) == 1 else 0 for col in attr_columns}

        row = {
                "image_id": src.name,
                "source_path": str(src.relative_to(root)),
                "processed_path": str(dst.relative_to(root)),
                "detected_face": int(detected),
                "crop_x1": box[0],
                "crop_y1": box[1],
                "crop_x2": box[2],
                "crop_y2": box[3],
                "brightness": f"{brightness:.6f}",
                "Bright": 0,
                "condition_col": "",
                "split": "",
            }
        row.update(attr_values)
        rows.append(row)

    if not rows:
        raise RuntimeError("No images were preprocessed.")

    median_brightness = float(np.median([float(r["brightness"]) for r in rows]))
    for row in rows:
        row["Bright"] = 1 if float(row["brightness"]) >= median_brightness else 0

    requested = CONDITION_COL.strip()
    if requested.lower() == "auto":
        condition_col = "Male" if has_male and any(r["Male"] != "" for r in rows) else "Bright"
    else:
        condition_col = requested
    if condition_col not in rows[0] or not any(str(r[condition_col]) != "" for r in rows):
        condition_col = "Bright"

    if eval_partition:
        for row in rows:
            row["split"] = eval_partition.get(str(row["image_id"]), "train")
    else:
        rng = np.random.default_rng(SEED)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        n = len(rows)
        n_test = int(round(n * TEST_RATIO))
        n_val = int(round(n * VAL_RATIO))
        test_ids = set(indices[:n_test])
        val_ids = set(indices[n_test : n_test + n_val])
        for i, row in enumerate(rows):
            if i in test_ids:
                row["split"] = "test"
            elif i in val_ids:
                row["split"] = "val"
            else:
                row["split"] = "train"
    for row in rows:
        row["condition_col"] = condition_col

    fieldnames = [
        "image_id",
        "source_path",
        "processed_path",
        "detected_face",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "brightness",
        "Bright",
        *attr_columns,
        "condition_col",
        "split",
    ]
    metadata_csv = out_dir / "metadata.csv"
    write_csv_rows(metadata_csv, rows, fieldnames)

    split_counts = Counter(str(r["split"]) for r in rows)
    label_counts = Counter(str(r[condition_col]) for r in rows)
    summary = {
        "raw_dir": str(raw_dir.relative_to(root)),
        "attr_file": str(attr_file.relative_to(root)) if attr_file else None,
        "eval_partition_file": str(eval_partition_file.relative_to(root)) if eval_partition_file else None,
        "images_total": len(image_paths),
        "images_processed": len(rows),
        "images_skipped": skipped,
        "image_size": IMAGE_SIZE,
        "face_detector": "OpenCV Haar cascade",
        "detected_faces": int(sum(int(r["detected_face"]) for r in rows)),
        "fallback_center_crops": int(sum(1 - int(r["detected_face"]) for r in rows)),
        "condition_col": condition_col,
        "condition_counts": dict(label_counts),
        "attr_columns": attr_columns,
        "median_brightness": median_brightness,
        "splits": dict(split_counts),
        "metadata_csv": str(metadata_csv.relative_to(root)),
    }
    save_json(out_dir / "summary.json", summary)

    print("Prepared dataset:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
