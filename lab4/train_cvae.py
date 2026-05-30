import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lab4lib.data import FaceMetadataDataset, resolve_condition_column
from lab4lib.model import ConditionalVAE
from lab4lib.utils import (
    count_parameters,
    ensure_dir,
    plot_history,
    project_root,
    save_json,
    save_tensor_grid,
    seed_everything,
    write_csv_rows,
)

METADATA = Path("data/processed/metadata.csv")
OUTPUT_DIR = Path("runs/cvae")
CONDITION_COL = "auto"
EPOCHS = 8
BATCH_SIZE = 64
LATENT_DIM = 128
LR = 1e-3
BETA_MAX = 0.0005
NUM_WORKERS = 0
MAX_TRAIN_SAMPLES = 0
MAX_VAL_SAMPLES = 0
SEED = 42
DEVICE = "auto"
SAMPLE_EVERY = 1


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = F.l1_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    loss = recon_loss + beta * kl
    return loss, recon_loss, kl


def run_epoch(model, loader, optimizer, device, beta):
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for x, c in pbar:
        x = x.to(device)
        c = c.to(device) if c.numel() else None
        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(x, c)
            loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_n = x.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_n
        totals["recon"] += float(recon_loss.detach().cpu()) * batch_n
        totals["kl"] += float(kl.detach().cpu()) * batch_n
        totals["n"] += batch_n
        pbar.set_postfix(loss=totals["loss"] / max(1, totals["n"]))
    n = max(1, int(totals["n"]))
    return {k: totals[k] / n for k in ("loss", "recon", "kl")}


@torch.no_grad()
def save_samples(model, val_loader, device, output_dir, epoch, condition_col):
    model.eval()
    x, c = next(iter(val_loader))
    x = x[:16].to(device)
    c_device = c[:16].to(device) if c.numel() else None
    recon, _, _ = model(x, c_device)
    save_tensor_grid(torch.cat([x.cpu(), recon.cpu()], dim=0), output_dir / "samples" / f"recon_epoch_{epoch:03d}.png", nrow=8)

    n = 32
    if model.cond_dim > 0:
        labels = torch.cat([torch.zeros(n // 2, 1), torch.ones(n - n // 2, 1)], dim=0).to(device)
    else:
        labels = None
    gen = model.sample(n, labels, device)
    name = f"generated_epoch_{epoch:03d}.png"
    if condition_col:
        name = f"generated_{condition_col}_epoch_{epoch:03d}.png"
    save_tensor_grid(gen, output_dir / "samples" / name, nrow=8)


def main():
    root = project_root()
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available() else "cpu")

    metadata = (root / METADATA).resolve() if not METADATA.is_absolute() else METADATA
    output_dir = ensure_dir(root / OUTPUT_DIR if not OUTPUT_DIR.is_absolute() else OUTPUT_DIR)
    condition_col = resolve_condition_column(metadata, CONDITION_COL)

    train_ds = FaceMetadataDataset(metadata, "train", condition_col=condition_col, limit=MAX_TRAIN_SAMPLES, augment=True, root=root)
    val_ds = FaceMetadataDataset(metadata, "val", condition_col=condition_col, limit=MAX_VAL_SAMPLES, augment=False, root=root)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    model = ConditionalVAE(latent_dim=LATENT_DIM, cond_dim=train_ds.cond_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=1e-5)

    config = {
        "metadata": str(metadata.relative_to(root)),
        "output_dir": str(output_dir.relative_to(root)),
        "condition_col": condition_col or "none",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "lr": LR,
        "beta_max": BETA_MAX,
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "parameters": count_parameters(model),
    }
    save_json(output_dir / "config.json", config)
    print("Training config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    history = []
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        beta = BETA_MAX * min(1.0, epoch / max(1.0, EPOCHS * 0.4))
        print(f"\nEpoch {epoch}/{EPOCHS} beta={beta:.6f}")
        train_metrics = run_epoch(model, train_loader, optimizer, device, beta)
        val_metrics = run_epoch(model, val_loader, None, device, beta)
        row = {
            "epoch": epoch,
            "beta": f"{beta:.8f}",
            "train_loss": f"{train_metrics['loss']:.8f}",
            "train_recon": f"{train_metrics['recon']:.8f}",
            "train_kl": f"{train_metrics['kl']:.8f}",
            "val_loss": f"{val_metrics['loss']:.8f}",
            "val_recon": f"{val_metrics['recon']:.8f}",
            "val_kl": f"{val_metrics['kl']:.8f}",
        }
        history.append(row)
        print(
            f"loss train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} "
            f"recon val={val_metrics['recon']:.4f} kl val={val_metrics['kl']:.2f}"
        )

        if epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS:
            save_samples(model, val_loader, device, output_dir, epoch, condition_col)

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "condition_col": condition_col,
            "cond_dim": train_ds.cond_dim,
            "latent_dim": LATENT_DIM,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, output_dir / "best.pt")

        write_csv_rows(
            output_dir / "history.csv",
            history,
            ["epoch", "beta", "train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl"],
        )
        plot_history(output_dir / "history.csv", output_dir / "loss_curves.png")

    elapsed = time.time() - start
    config["elapsed_sec"] = round(elapsed, 2)
    config["best_val_loss"] = best_val
    save_json(output_dir / "config.json", config)
    print(f"Done in {elapsed / 60.0:.1f} min. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
