import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DIFFUSERS_DIR = (BASE_DIR / "../diffusers").resolve()
DIFFUSERS_SRC_DIR = DIFFUSERS_DIR / "src"
TRAIN_SCRIPT = DIFFUSERS_DIR / "examples" / "dreambooth" / "train_dreambooth_lora_sdxl.py"

INSTANCE_DIR = (BASE_DIR / "../me").resolve()
CLASS_DIR = BASE_DIR / "data" / "class"
OUTPUT_DIR = BASE_DIR / "outputs" / "sdxl_me_lora"

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
TOKEN = "i3373_person"
SUBJECT_CLASS = "man"

INSTANCE_PROMPT = f"photo of {TOKEN} {SUBJECT_CLASS}"
CLASS_PROMPT = f"photo of a {SUBJECT_CLASS}"

RESOLUTION = 1024
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 800
CHECKPOINTING_STEPS = 200
NUM_CLASS_IMAGES = 150
SAMPLE_BATCH_SIZE = 1
RANK = 8
SEED = 42


def build_command():
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        "fp16",
        "--dynamo_backend",
        "no",
        str(TRAIN_SCRIPT),
        "--pretrained_model_name_or_path",
        MODEL_NAME,
        "--instance_data_dir",
        str(INSTANCE_DIR),
        "--class_data_dir",
        str(CLASS_DIR),
        "--output_dir",
        str(OUTPUT_DIR),
        "--instance_prompt",
        INSTANCE_PROMPT,
        "--class_prompt",
        CLASS_PROMPT,
        "--with_prior_preservation",
        "--prior_loss_weight",
        "1.0",
        "--num_class_images",
        str(NUM_CLASS_IMAGES),
        "--sample_batch_size",
        str(SAMPLE_BATCH_SIZE),
        "--resolution",
        str(RESOLUTION),
        "--train_batch_size",
        str(TRAIN_BATCH_SIZE),
        "--gradient_accumulation_steps",
        str(GRAD_ACCUM_STEPS),
        "--gradient_checkpointing",
        "--learning_rate",
        str(LEARNING_RATE),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--max_train_steps",
        str(MAX_TRAIN_STEPS),
        "--checkpointing_steps",
        str(CHECKPOINTING_STEPS),
        "--checkpoints_total_limit",
        "3",
        "--mixed_precision",
        "fp16",
        "--rank",
        str(RANK),
        "--seed",
        str(SEED),
        "--pre_compute_text_embeddings",
        "--allow_tf32",
    ]


def main():
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        str(DIFFUSERS_SRC_DIR) if not existing_pythonpath else str(DIFFUSERS_SRC_DIR) + os.pathsep + existing_pythonpath
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = build_command()
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print("Стартую обучение LoRA.")
    print(printable)
    print()

    subprocess.run(cmd, check=True)
    print(f"\nГотово. LoRA лежит здесь: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
