import sys

import evaluate_metrics
import make_report
import prepare_data
import train_cvae
from lab4lib.utils import project_root


EPOCHS = 30
BATCH_SIZE = 64
LATENT_DIM = 128
BETA_MAX = 0.0005
METRIC_IMAGES = 64
FID_DIM = 256
PREPARE_LIMIT = 0
USE_EVAL_PARTITION = True
MAX_TRAIN_SAMPLES = 0
MAX_VAL_SAMPLES = 0
SKIP_METRICS = False


def run_step(name, func):
    print(f"\n=== {name} ===")
    func()


def main():
    root = project_root()
    print(f"Project root: {root}")
    print(f"Python: {sys.executable}")

    prepare_data.LIMIT = PREPARE_LIMIT
    prepare_data.USE_EVAL_PARTITION = USE_EVAL_PARTITION
    run_step("prepare_data.py", prepare_data.main)

    train_cvae.EPOCHS = EPOCHS
    train_cvae.BATCH_SIZE = BATCH_SIZE
    train_cvae.LATENT_DIM = LATENT_DIM
    train_cvae.BETA_MAX = BETA_MAX
    train_cvae.MAX_TRAIN_SAMPLES = MAX_TRAIN_SAMPLES
    train_cvae.MAX_VAL_SAMPLES = MAX_VAL_SAMPLES

    train_cvae.CONDITION_COL = "none"
    train_cvae.OUTPUT_DIR = train_cvae.Path("runs/vae_unconditional")
    run_step("train unconditional VAE", train_cvae.main)

    train_cvae.CONDITION_COL = "auto"
    train_cvae.OUTPUT_DIR = train_cvae.Path("runs/cvae_conditioned")
    run_step("train conditional CVAE", train_cvae.main)

    if not SKIP_METRICS:
        evaluate_metrics.NUM_IMAGES = METRIC_IMAGES
        evaluate_metrics.FID_DIM = FID_DIM

        evaluate_metrics.CHECKPOINT = evaluate_metrics.Path("runs/vae_unconditional/best.pt")
        evaluate_metrics.OUTPUT_DIR = evaluate_metrics.Path("runs/vae_unconditional/metrics")
        evaluate_metrics.CONDITION_COL = "none"
        run_step("evaluate unconditional VAE", evaluate_metrics.main)

        evaluate_metrics.CHECKPOINT = evaluate_metrics.Path("runs/cvae_conditioned/best.pt")
        evaluate_metrics.OUTPUT_DIR = evaluate_metrics.Path("runs/cvae_conditioned/metrics")
        evaluate_metrics.CONDITION_COL = "auto"
        run_step("evaluate conditional CVAE", evaluate_metrics.main)

    run_step("make_report.py", make_report.main)
    print("\nDone. Open REPORT.md for the lab write-up.")


if __name__ == "__main__":
    main()
