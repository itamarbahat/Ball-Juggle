"""
Generates ground-truth ball masks using SAM3.
Input: evaluation/eval_input/   Output: evaluation/ground_truth/
"""

import os
import glob
import time
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
INPUT_DIR = os.path.join(EVAL_DIR, "eval_input")
GT_DIR = os.path.join(EVAL_DIR, "ground_truth")

TEXT_PROMPT = "ball"
CONFIDENCE_THRESHOLD = 0.3
MASK_THRESHOLD = 0.5


def load_model():
    from huggingface_hub import login
    try:
        login()
    except Exception:
        token = input("Enter your Hugging Face token: ").strip()
        login(token=token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM3] Loading model on {device}...")

    t0 = time.time()
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    t_model = time.time() - t0

    t0 = time.time()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    t_proc = time.time() - t0

    print(f"[SAM3] Model loaded in {t_model:.1f}s  |  Processor loaded in {t_proc:.1f}s")
    return model, processor, device


def generate_mask(image, model, processor, device):
    timings = {}

    t0 = time.time()
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    timings["preprocess"] = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    timings["inference"] = time.time() - t0

    t0 = time.time()
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=CONFIDENCE_THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    timings["postprocess"] = time.time() - t0

    masks = results.get("masks", [])
    scores = results.get("scores", [])

    if len(masks) == 0:
        return None, timings

    best_idx = int(torch.argmax(torch.tensor(scores)))
    mask = masks[best_idx].cpu().numpy().astype(np.uint8) * 255
    return mask, timings


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def progress_bar(current, total, width=30):
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct * 100:.0f}%)"


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)

    image_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.png")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
    )

    if not image_files:
        print(f"[ERROR] No images found in {INPUT_DIR}/")
        print(f"        Drop your frames there and run again.")
        return

    total = len(image_files)
    print(f"[SAM3] Found {total} images in {INPUT_DIR}/")
    print(f"[SAM3] Output: {GT_DIR}/")
    print()

    model, processor, device = load_model()
    print()

    found = 0
    missed = 0
    total_time = 0
    per_image_times = []
    cumulative_timings = {"preprocess": 0, "inference": 0, "postprocess": 0}

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        image = Image.open(img_path).convert("RGB")

        img_start = time.time()
        mask, timings = generate_mask(image, model, processor, device)
        img_elapsed = time.time() - img_start

        total_time += img_elapsed
        per_image_times.append(img_elapsed)
        for k in cumulative_timings:
            cumulative_timings[k] += timings.get(k, 0)

        gt_path = os.path.join(GT_DIR, f"{name_no_ext}.png")
        if mask is not None:
            Image.fromarray(mask).save(gt_path)
            found += 1
            status = "ball detected"
        else:
            blank = np.zeros((image.height, image.width), dtype=np.uint8)
            Image.fromarray(blank).save(gt_path)
            missed += 1
            status = "no ball"

        avg_time = total_time / (i + 1)
        remaining = avg_time * (total - i - 1)

        print(f"  {progress_bar(i + 1, total)}  "
              f"{filename} -> {status}  "
              f"({img_elapsed:.2f}s)  "
              f"ETA: {format_time(remaining)}")

    print()
    print("=" * 65)
    print(f"  DONE  |  {found} detected, {missed} missed  |  {total} images")
    print(f"  Total time: {format_time(total_time)}")
    if per_image_times:
        print(f"  Per image:  avg {np.mean(per_image_times):.2f}s  |  "
              f"min {np.min(per_image_times):.2f}s  |  max {np.max(per_image_times):.2f}s")
    print()
    print("  Time breakdown (total across all images):")
    for step, t in cumulative_timings.items():
        pct = (t / total_time * 100) if total_time > 0 else 0
        print(f"    {step:<14} {format_time(t):>8}  ({pct:.1f}%)")
    print("=" * 65)
    print(f"  Masks saved to {GT_DIR}/")


if __name__ == "__main__":
    main()
