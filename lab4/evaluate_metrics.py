from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from lab4lib.data import FaceMetadataDataset, resolve_condition_column
from lab4lib.model import ConditionalVAE
from lab4lib.utils import ensure_dir, project_root, save_json, save_tensor_grid, seed_everything

METADATA = Path("data/processed/metadata.csv")
CHECKPOINT = Path("runs/cvae_conditioned/best.pt")
OUTPUT_DIR = Path("runs/cvae_conditioned/metrics")
CONDITION_COL = "auto"
NUM_IMAGES = 64
BATCH_SIZE = 16
FID_DIM = 256
BACKBONE = "inception"
SEED = 42
DEVICE = "auto"


class MetricExtractor:
    def __init__(self, device, backbone="inception"):
        self.device = device
        self.kind = "fallback"
        self.note = ""
        self.model = None
        self._features = None
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        if backbone == "inception":
            try:
                weights = models.Inception_V3_Weights.DEFAULT
                model = models.inception_v3(weights=weights)
                model.eval().to(device)
                model.avgpool.register_forward_hook(self._hook)
                self.model = model
                self.kind = "inception_v3_imagenet"
                self.note = "FID uses Inception-v3 pool features; IS uses ImageNet class probabilities."
            except Exception as exc:  # noqa: BLE001
                self.kind = "fallback_random_projection"
                self.note = (
                    "Inception-v3 weights were unavailable, so the script used a deterministic "
                    f"random-projection fallback. This is not the canonical FID/IS. Error: {exc}"
                )
        else:
            self.kind = "fallback_random_projection"
            self.note = "Manual fallback selected; this is not the canonical FID/IS."

    def _hook(self, _module, _inputs, output):
        self._features = output.detach()

    @torch.no_grad()
    def extract(self, images, batch_size):
        features = []
        logits_all = []
        for batch in tqdm(list(images.split(batch_size)), desc=f"features/{self.kind}", leave=False):
            batch = batch.to(self.device)
            if self.model is not None:
                x = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
                x = (x - self.mean) / self.std
                self._features = None
                logits = self.model(x)
                feats = self._features
                if feats is None:
                    raise RuntimeError("Inception feature hook did not fire.")
                features.append(torch.flatten(feats, start_dim=1).cpu().numpy())
                logits_all.append(logits.cpu().numpy())
            else:
                x = F.interpolate(batch, size=(32, 32), mode="bilinear", align_corners=False)
                flat = torch.flatten(x, start_dim=1).cpu().numpy()
                rng = np.random.default_rng(123)
                w_feat = rng.normal(0.0, 1.0 / np.sqrt(flat.shape[1]), size=(flat.shape[1], 256))
                w_log = rng.normal(0.0, 1.0 / np.sqrt(flat.shape[1]), size=(flat.shape[1], 10))
                features.append(flat @ w_feat)
                logits_all.append(flat @ w_log)
        return np.concatenate(features, axis=0), np.concatenate(logits_all, axis=0)


def project_features(features, target_dim, seed=1234):
    if target_dim <= 0 or target_dim >= features.shape[1]:
        return features
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(target_dim), size=(features.shape[1], target_dim))
    return features @ projection


def calculate_fid(real_features, fake_features):
    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)
    diff = mu_r - mu_f
    eps = 1e-6
    covmean, _ = scipy.linalg.sqrtm((sigma_r + eps * np.eye(sigma_r.shape[0])) @ (sigma_f + eps * np.eye(sigma_f.shape[0])), disp=False)
    if not np.isfinite(covmean).all():
        covmean = scipy.linalg.sqrtm((sigma_r + 1e-4 * np.eye(sigma_r.shape[0])) @ (sigma_f + 1e-4 * np.eye(sigma_f.shape[0])))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean)
    return float(np.real(fid))


def calculate_inception_score(logits, splits=5):
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    split_scores = []
    for part in np.array_split(probs, min(splits, len(probs))):
        if len(part) == 0:
            continue
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(np.maximum(part, 1e-12)) - np.log(np.maximum(py, 1e-12)))
        split_scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(split_scores)), float(np.std(split_scores))


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = ConditionalVAE(
        latent_dim=int(checkpoint.get("latent_dim", config.get("latent_dim", 64))),
        cond_dim=int(checkpoint.get("cond_dim", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate_images(model, n, batch_size, device):
    batches = []
    remaining = n
    label_cursor = 0
    while remaining > 0:
        b = min(batch_size, remaining)
        if model.cond_dim > 0:
            labels = torch.tensor([label_cursor % 2 for label_cursor in range(label_cursor, label_cursor + b)], dtype=torch.float32)
            label_cursor += b
            c = labels.view(-1, 1).to(device)
        else:
            c = None
        fake = model.sample(b, c, device)
        batches.append(((fake.cpu() + 1.0) / 2.0).clamp(0.0, 1.0))
        remaining -= b
    return torch.cat(batches, dim=0)


def main():
    root = project_root()
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available() else "cpu")

    metadata = (root / METADATA).resolve() if not METADATA.is_absolute() else METADATA
    checkpoint_path = (root / CHECKPOINT).resolve() if not CHECKPOINT.is_absolute() else CHECKPOINT
    output_dir = ensure_dir(root / OUTPUT_DIR if not OUTPUT_DIR.is_absolute() else OUTPUT_DIR)
    condition_col = resolve_condition_column(metadata, CONDITION_COL)

    model, checkpoint = load_model(checkpoint_path, device)
    if model.cond_dim == 0:
        condition_col = None

    real_ds = FaceMetadataDataset(metadata, "test", condition_col=condition_col, limit=NUM_IMAGES, augment=False, root=root)
    n_eval = min(NUM_IMAGES, len(real_ds))
    real_loader = DataLoader(real_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    real_images = []
    for x, _ in real_loader:
        real_images.append(((x + 1.0) / 2.0).clamp(0.0, 1.0))
    real = torch.cat(real_images, dim=0)[:n_eval]
    fake = generate_images(model, n_eval, BATCH_SIZE, device)

    save_tensor_grid(real[: min(32, n_eval)] * 2.0 - 1.0, output_dir / "real_grid.png", nrow=8)
    save_tensor_grid(fake[: min(32, n_eval)] * 2.0 - 1.0, output_dir / "generated_grid.png", nrow=8)

    extractor = MetricExtractor(device=device, backbone=BACKBONE)
    real_features, _ = extractor.extract(real, BATCH_SIZE)
    fake_features, fake_logits = extractor.extract(fake, BATCH_SIZE)
    real_features_proj = project_features(real_features, FID_DIM)
    fake_features_proj = project_features(fake_features, FID_DIM)

    fid = calculate_fid(real_features_proj, fake_features_proj)
    is_mean, is_std = calculate_inception_score(fake_logits)
    metrics = {
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "condition_col": condition_col or "none",
        "num_images": int(n_eval),
        "fid": fid,
        "fid_dim": int(real_features_proj.shape[1]),
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "backbone": extractor.kind,
        "note": extractor.note,
        "device": str(device),
    }
    save_json(output_dir / "metrics.json", metrics)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
