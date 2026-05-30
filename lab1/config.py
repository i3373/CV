from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_ROOT = BASE_DIR / "confirmed_fronts_clean"
OUT_DIR = BASE_DIR / "outputs_min50_best20"
SAMPLE_BATCH_PATH = BASE_DIR / "sample_batch.jpg"

VAL_SIZE = 0.15
TEST_SIZE = 0.15
MIN_COUNT_PER_CLASS = 50
EXCLUDE_CLASSES: list[str] = []

ARCHS = ["resnet50_scratch", "efficientnet_b0_ft", "densenet121_ft"]
TRAIN_ARCH = "resnet50_scratch"
EPOCHS_SCRATCH = 20
EPOCHS_FT = 20
BATCH_SIZE = 32
EXAMPLE_BATCH_SIZE = 8
NUM_WORKERS = 2
EXAMPLE_NUM_WORKERS = 0
IMG_SIZE = 224
SEED = 42
WEIGHT_DECAY = 1e-4
LR_SCRATCH = 3e-4
LR_FT = 1e-4
CLASS_WEIGHTING = "effective" 
SAMPLER = "weighted_sqrt" 
LABEL_SMOOTHING = 0.03
FREEZE_BACKBONE_EPOCHS = 0
COLOR_JITTER = False
USE_AMP = True

EVAL_TTA_HFLIP = True
INFER_CKPT = OUT_DIR / "efficientnet_b0_ft" / "best.pt"
INFER_IMAGE = ""
