from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


BASE_DIR = Path(__file__).resolve().parent
LORA_DIR = BASE_DIR / "outputs" / "sdxl_me_lora"
SAVE_DIR = BASE_DIR / "outputs" / "generated_yacht"

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
TOKEN = "i3373_person"
PROMPT = f"photo of {TOKEN} man on a yacht, realistic, high quality, detailed face"
NEGATIVE_PROMPT = "low quality, blurry, deformed face, bad anatomy, text, watermark"

IMAGES = 4
STEPS = 40
GUIDANCE = 7.0
WIDTH = 1024
HEIGHT = 1024
SEED = 1000


def main():
    if not LORA_DIR.exists():
        raise FileNotFoundError(f"Не нашел LoRA: {LORA_DIR}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "use_safetensors": True,
    }

    if device == "cuda":
        pipe_kwargs["variant"] = "fp16"

    print("Загружаю SDXL.")
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_NAME, **pipe_kwargs).to(device)

    print("Подключаю LoRA.")
    pipe.load_lora_weights(str(LORA_DIR))

    print(f"Промпт: {PROMPT}")

    for index in range(IMAGES):
        seed = SEED + index
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            width=WIDTH,
            height=HEIGHT,
            generator=generator,
        ).images[0]

        out_path = SAVE_DIR / f"yacht_{index:02d}_seed{seed}.png"
        image.save(out_path)
        print(f"Сохранено: {out_path}")

    print(f"Готово: {SAVE_DIR}")


if __name__ == "__main__":
    main()
