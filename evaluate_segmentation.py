"""
Compares pipeline masks against SAM3 ground-truth using raw HSV mask
and Hough circle mask. Computes DICE, IoU, Precision, Recall, F1, Accuracy.
"""

import cv2
import numpy as np
import os
import csv
from config import CONFIG

EVAL_DIR = "evaluation"
INPUT_DIR = os.path.join(EVAL_DIR, "eval_input")
GT_DIR = os.path.join(EVAL_DIR, "ground_truth")
MASKS_DIR = os.path.join(EVAL_DIR, "pipeline_masks")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def compute_metrics(pred, gt):
    p = (pred > 127).astype(np.uint8).ravel()
    g = (gt > 127).astype(np.uint8).ravel()

    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    tn = int(np.sum((p == 0) & (g == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    union = tp + fp + fn
    iou  = tp / union if union else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "dice": dice, "iou": iou,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def hough_to_mask(binary_mask, shape):
    min_r = int(np.sqrt(CONFIG["detection"]["min_area"] / np.pi) * 0.25)
    max_r = int(np.sqrt(CONFIG["detection"]["max_area"] / np.pi) * 1.75)

    blurred = cv2.GaussianBlur(binary_mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=CONFIG["detection"].get("hough_param1", 50),
        param2=CONFIG["detection"].get("hough_param2", 20),
        minRadius=min_r,
        maxRadius=max_r
    )

    filled = np.zeros(shape[:2], dtype=np.uint8)
    if circles is not None:
        cx, cy, r = np.round(circles[0, 0]).astype(int)
        cv2.circle(filled, (cx, cy), r, 255, -1)
        return filled, (cx, cy, r)
    return filled, None


GT_COLOR = (255, 255, 255)       # white
RAW_MASK_COLOR = (0, 255, 255)   # yellow
HOUGH_COLOR = (0, 255, 0)        # green


def save_overlay(frame, gt, raw_mask, circle, raw_metrics, hough_metrics, name, out_dir):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    gt_contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, GT_COLOR, 2)

    mask_contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, mask_contours, -1, RAW_MASK_COLOR, 2)

    if circle:
        cx, cy, r = circle
        cv2.circle(overlay, (cx, cy), r, HOUGH_COLOR, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    legend_items = [
        ("GT (SAM3)", GT_COLOR),
        ("HSV Mask", RAW_MASK_COLOR),
        ("Hough Circle", HOUGH_COLOR),
    ]
    lx, ly = 8, 18
    for label, color in legend_items:
        cv2.rectangle(overlay, (lx, ly - 10), (lx + 14, ly + 2), color, -1)
        cv2.putText(overlay, label, (lx + 20, ly), font, 0.42, (255, 255, 255), 1)
        ly += 18

    lines = [
        f"HSV Mask:  DICE={raw_metrics['dice']:.3f}  IoU={raw_metrics['iou']:.3f}  "
        f"Prec={raw_metrics['precision']:.3f}  Rec={raw_metrics['recall']:.3f}  F1={raw_metrics['f1']:.3f}",
    ]
    if circle:
        lines.append(
            f"Hough:     DICE={hough_metrics['dice']:.3f}  IoU={hough_metrics['iou']:.3f}  "
            f"Prec={hough_metrics['precision']:.3f}  Rec={hough_metrics['recall']:.3f}  F1={hough_metrics['f1']:.3f}"
        )
    else:
        lines.append("Hough:     no circle detected")

    text_y = h - 10 - (len(lines) - 1) * 18
    for line in lines:
        sz = cv2.getTextSize(line, font, 0.38, 1)[0]
        cv2.rectangle(overlay, (6, text_y - 12), (12 + sz[0], text_y + 4), (0, 0, 0), -1)
        cv2.putText(overlay, line, (8, text_y), font, 0.38, (255, 255, 255), 1)
        text_y += 18

    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), overlay)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for folder, label in [(GT_DIR, "ground_truth"), (MASKS_DIR, "pipeline_masks")]:
        if not os.path.exists(folder):
            print(f"[ERROR] {label} folder not found: {folder}")
            print(f"        Run generate_ground_truth.py / run_pipeline.py first.")
            return

    gt_files = {os.path.splitext(f)[0]: os.path.join(GT_DIR, f)
                for f in os.listdir(GT_DIR) if f.lower().endswith(".png")}
    mask_files = {os.path.splitext(f)[0]: os.path.join(MASKS_DIR, f)
                  for f in os.listdir(MASKS_DIR) if f.lower().endswith(".png")}
    input_files = {}
    if os.path.exists(INPUT_DIR):
        input_files = {os.path.splitext(f)[0]: os.path.join(INPUT_DIR, f)
                       for f in os.listdir(INPUT_DIR)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))}

    common = sorted(set(gt_files.keys()) & set(mask_files.keys()))
    if not common:
        print("[ERROR] No matching filenames between ground_truth/ and pipeline_masks/.")
        print(f"  GT has: {sorted(gt_files.keys())[:5]} ...")
        print(f"  Pipeline has: {sorted(mask_files.keys())[:5]} ...")
        return

    print("=" * 70)
    print("  Segmentation Evaluation")
    print("=" * 70)
    print(f"  GT folder        : {GT_DIR}/  ({len(gt_files)} masks)")
    print(f"  Pipeline folder  : {MASKS_DIR}/  ({len(mask_files)} masks)")
    print(f"  Matched images   : {len(common)}")
    print(f"  Results output   : {RESULTS_DIR}/")
    print("=" * 70)
    print()

    results = []

    for i, name in enumerate(common):
        gt_mask = cv2.imread(gt_files[name], cv2.IMREAD_GRAYSCALE)
        raw_mask = cv2.imread(mask_files[name], cv2.IMREAD_GRAYSCALE)

        if gt_mask is None or raw_mask is None:
            print(f"  [WARNING] Could not read masks for {name}")
            continue

        if gt_mask.shape != raw_mask.shape:
            gt_mask = cv2.resize(gt_mask, (raw_mask.shape[1], raw_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        raw_metrics = compute_metrics(raw_mask, gt_mask)

        hough_mask, circle = hough_to_mask(raw_mask, raw_mask.shape)
        hough_metrics = compute_metrics(hough_mask, gt_mask)

        row = {"image": name}
        for k, v in raw_metrics.items():
            row[f"raw_{k}"] = v
        for k, v in hough_metrics.items():
            row[f"hough_{k}"] = v
        row["hough_detected"] = 1 if circle else 0
        results.append(row)

        frame = cv2.imread(input_files[name]) if name in input_files else \
                cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
        if frame is not None and frame.shape[:2] != raw_mask.shape[:2]:
            frame = cv2.resize(frame, (raw_mask.shape[1], raw_mask.shape[0]))
        save_overlay(frame, gt_mask, raw_mask, circle, raw_metrics, hough_metrics, name, RESULTS_DIR)

        print(f"  [{i+1}/{len(common)}] {name:<20}  "
              f"Raw IoU={raw_metrics['iou']:.3f}  DICE={raw_metrics['dice']:.3f}  |  "
              f"Hough IoU={hough_metrics['iou']:.3f}  DICE={hough_metrics['dice']:.3f}"
              f"{'  (no circle)' if not circle else ''}")

    if not results:
        print("[WARNING] No images were evaluated.")
        return

    csv_path = os.path.join(RESULTS_DIR, "metrics_per_frame.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    metric_keys = ["dice", "iou", "precision", "recall", "f1", "accuracy"]
    methods = {"Raw Mask": "raw", "Hough Circle": "hough"}

    summary = {}
    for label, prefix in methods.items():
        summary[label] = {}
        for mk in metric_keys:
            vals = [r[f"{prefix}_{mk}"] for r in results]
            summary[label][mk] = {
                "mean": np.mean(vals), "std": np.std(vals),
                "min": np.min(vals), "max": np.max(vals)
            }

    hough_detect_rate = sum(r["hough_detected"] for r in results) / len(results) * 100

    print()
    print("=" * 70)
    print("  SUMMARY  (mean +/- std)")
    print("=" * 70)
    header = f"  {'Metric':<12}"
    for label in methods:
        header += f"  {label:>22}"
    print(header)
    print("-" * 70)
    for mk in metric_keys:
        line = f"  {mk:<12}"
        for label in methods:
            s = summary[label][mk]
            line += f"  {s['mean']:.4f} +/- {s['std']:.4f}  "
        print(line)
    print("-" * 70)
    print(f"  Hough detection rate: {hough_detect_rate:.1f}%")
    print(f"  Evaluated {len(results)} images")
    print("=" * 70)

    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "mean", "std", "min", "max"])
        for label, prefix in methods.items():
            for mk in metric_keys:
                s = summary[label][mk]
                writer.writerow([label, mk, f"{s['mean']:.6f}", f"{s['std']:.6f}",
                                 f"{s['min']:.6f}", f"{s['max']:.6f}"])
        writer.writerow(["Hough", "detection_rate", f"{hough_detect_rate:.2f}", "", "", ""])

    print(f"\n[EVAL] Per-frame CSV  : {csv_path}")
    print(f"[EVAL] Summary CSV    : {summary_path}")
    print(f"[EVAL] Comparisons    : {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
