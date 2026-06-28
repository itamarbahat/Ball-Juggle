"""
Runs the single-frame detection pipeline (blur, HSV, morphology, Hough)
on evaluation images and saves binary masks and Hough overlays.
"""

import cv2
import numpy as np
import os
import glob
import time
from config import CONFIG
from Utils.config_utils import load_hsv_config

EVAL_DIR = "evaluation"
INPUT_DIR = os.path.join(EVAL_DIR, "eval_input")
MASKS_DIR = os.path.join(EVAL_DIR, "pipeline_masks")
HOUGH_DIR = os.path.join(EVAL_DIR, "pipeline_hough")

CAMERA_PROFILE = "side"


def progress_bar(current, total, width=30):
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct * 100:.0f}%)"


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(HOUGH_DIR, exist_ok=True)

    image_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.png")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
    )

    if not image_files:
        print(f"[ERROR] No images found in {INPUT_DIR}/")
        print(f"        Drop your frames there and run again.")
        return

    lower_hsv, upper_hsv = load_hsv_config(CAMERA_PROFILE)
    blur_k = tuple(CONFIG["processing"]["blur_kernel"])
    kernel_open = np.ones(CONFIG["processing"]["morph_open_kernel"], np.uint8)
    kernel_close = np.ones(CONFIG["processing"]["morph_close_kernel"], np.uint8)

    min_hough_r = int(np.sqrt(CONFIG["detection"]["min_area"] / np.pi) * 0.25)
    max_hough_r = int(np.sqrt(CONFIG["detection"]["max_area"] / np.pi) * 1.75)
    hough_param1 = CONFIG["detection"].get("hough_param1", 50)
    hough_param2 = CONFIG["detection"].get("hough_param2", 20)

    total = len(image_files)
    print("=" * 60)
    print("  Classic Pipeline Runner")
    print("=" * 60)
    print(f"  Input        : {INPUT_DIR}/  ({total} images)")
    print(f"  Masks output : {MASKS_DIR}/")
    print(f"  Hough output : {HOUGH_DIR}/")
    print(f"  HSV range    : {lower_hsv} -> {upper_hsv}")
    print(f"  Hough radius : [{min_hough_r}, {max_hough_r}]")
    print("=" * 60)
    print()

    detected = 0
    total_time = 0

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARNING] Could not read: {filename}")
            continue

        t0 = time.time()

        blurred = cv2.GaussianBlur(frame, blur_k, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        mask_for_hough = cv2.GaussianBlur(color_mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            mask_for_hough,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=hough_param1,
            param2=hough_param2,
            minRadius=min_hough_r,
            maxRadius=max_hough_r
        )

        contour_circle = None
        if circles is None:
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > CONFIG["detection"].get("relaxed_min_area", 100):
                    ((cx, cy), radius) = cv2.minEnclosingCircle(largest)
                    contour_circle = (int(cx), int(cy), int(radius))

        elapsed = time.time() - t0
        total_time += elapsed

        cv2.imwrite(os.path.join(MASKS_DIR, f"{name_no_ext}.png"), color_mask)

        hough_vis = frame.copy()
        if circles is not None:
            circles_rounded = np.round(circles[0, :]).astype("int")
            cx, cy, r = circles_rounded[0]
            cv2.circle(hough_vis, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(hough_vis, (cx, cy), 3, (0, 0, 255), -1)
            detected += 1
            status = "Hough"
        elif contour_circle is not None:
            cx, cy, r = contour_circle
            cv2.circle(hough_vis, (cx, cy), r, (255, 0, 255), 2)
            cv2.circle(hough_vis, (cx, cy), 3, (0, 0, 255), -1)
            detected += 1
            status = "Contour"
        else:
            status = "no detection"

        cv2.imwrite(os.path.join(HOUGH_DIR, f"{name_no_ext}.png"), hough_vis)

        print(f"  {progress_bar(i + 1, total)}  {filename} -> {status}  ({elapsed * 1000:.0f}ms)")

    print()
    print("=" * 60)
    print(f"  DONE  |  {detected}/{total} images with detections")
    print(f"  Total time: {total_time:.2f}s  |  Avg: {total_time / total * 1000:.0f}ms per image")
    print(f"  Masks -> {MASKS_DIR}/")
    print(f"  Hough -> {HOUGH_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
