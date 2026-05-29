import argparse
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPConfig, CLIPModel, CLIPProcessor


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = BASE_DIR / "outputs" / "generated"
DEFAULT_REFERENCE_DIR = (BASE_DIR / "../me").resolve()
CLIP_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-base-patch32"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    return parser.parse_args()


def collect_images(image_dir):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(image_dir.rglob(ext))
    return sorted(paths)


def sharpness(path):
    image = np.array(Image.open(path).convert("L"))
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def image_features(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(outputs.pooler_output)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features[0].cpu().numpy()


def reference_similarity(vector, reference_vectors, top_k=3):
    similarities = sorted((float(np.dot(vector, ref)) for ref in reference_vectors), reverse=True)
    return float(np.mean(similarities[:top_k]))


def mean_pairwise_distance(vectors):
    if len(vectors) < 2:
        return 0.0
    distances = []
    for left, right in combinations(vectors, 2):
        distances.append(1.0 - float(np.dot(left, right)))
    return float(np.mean(distances))


def load_clip(device):
    main_ref = (CLIP_CACHE_DIR / "refs" / "main").read_text(encoding="utf-8").strip()
    safetensors_ref = (CLIP_CACHE_DIR / "refs" / "refs" / "pr" / "66").read_text(encoding="utf-8").strip()
    main_snapshot = CLIP_CACHE_DIR / "snapshots" / main_ref
    safetensors_path = CLIP_CACHE_DIR / "snapshots" / safetensors_ref / "model.safetensors"

    processor = CLIPProcessor.from_pretrained(str(main_snapshot), local_files_only=True, use_fast=False)
    config = CLIPConfig.from_pretrained(str(main_snapshot), local_files_only=True)
    model = CLIPModel(config)
    model.load_state_dict(load_file(str(safetensors_path)), strict=False)
    model.to(device)
    model.eval()
    return processor, model


def main():
    args = parse_args()
    images = collect_images(args.image_dir)
    references = collect_images(args.reference_dir)

    if not images:
        raise RuntimeError(f"В папке нет изображений: {args.image_dir}")
    if not references:
        raise RuntimeError(f"В папке нет исходных фото: {args.reference_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_clip(device)

    reference_vectors = []
    for path in references:
        image = Image.open(path).convert("RGB")
        reference_vectors.append(image_features(model, processor, image, device))

    rows = []
    vectors_by_group = {}

    for path in images:
        group = path.relative_to(args.image_dir).parts[0] if path.parent != args.image_dir else args.image_dir.name
        image = Image.open(path).convert("RGB")
        vector = image_features(model, processor, image, device)
        vectors_by_group.setdefault(group, []).append(vector)
        rows.append(
            {
                "group": group,
                "reference_similarity": reference_similarity(vector, reference_vectors),
                "sharpness": sharpness(path),
            }
        )

    diversity_by_group = {group: mean_pairwise_distance(vectors) for group, vectors in vectors_by_group.items()}
    reference_scores = [row["reference_similarity"] for row in rows]
    sharpness_scores = [row["sharpness"] for row in rows]
    diversity_scores = list(diversity_by_group.values())
    
    print(f"ReferenceSimilarity: {np.mean(reference_scores):.4f}")
    print(f"Sharpness: {np.mean(sharpness_scores):.2f}")
    print(f"Diversity: {np.mean(diversity_scores):.4f}")



if __name__ == "__main__":
    main()
