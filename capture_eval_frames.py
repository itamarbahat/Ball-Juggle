"""
Captures frames from a video for evaluation.
SPACE to start/stop auto-capture, W/S to adjust interval, ESC to exit.
"""

import cv2
import os
import sys
import csv
from datetime import datetime
from config import CONFIG

EVAL_DIR = "evaluation"
FRAMES_DIR = os.path.join(EVAL_DIR, "frames")
GT_DIR = os.path.join(EVAL_DIR, "ground_truth")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

MAX_CAPTURE_FRAMES = None  # computed from FPS at runtime


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    source = sys.argv[1] if len(sys.argv) > 1 else CONFIG["camera"]["side_source"]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target_w = CONFIG["camera"]["width"]
    target_h = CONFIG["camera"]["height"]

    max_capture_duration = 60  # seconds
    max_capture_frames = int(fps * max_capture_duration)

    capture_interval = 15  # save one frame every N frames
    capturing = False
    capture_start_frame = 0
    frames_since_start = 0

    session_num = 0
    session_saved = 0
    session_dir = ""
    saved_entries = []
    frame_idx = 0

    print("=" * 55)
    print("  Frame Capture Tool")
    print("=" * 55)
    print(f"  Source       : {source}")
    print(f"  Output       : {FRAMES_DIR}/")
    print(f"  Resolution   : {target_w}x{target_h}")
    print(f"  Total frames : {total_frames}  ({fps:.0f} FPS)")
    print(f"  Capture limit: {max_capture_duration}s per session")
    print()
    print("  SPACE = start/stop capture session (each session = new folder)")
    print("  W / S = increase/decrease interval")
    print("  ESC   = exit")
    print("=" * 55)

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            if capturing:
                print(f"  Auto-capture stopped (video ended). Saved {saved_count} frames.")
                capturing = False
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame = cv2.resize(frame_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if capturing:
            frames_since_start = frame_idx - capture_start_frame
            if frames_since_start >= max_capture_frames:
                print(f"  Session {session_num} stopped ({max_capture_duration}s limit). "
                      f"Saved {session_saved} frames to {session_dir}/")
                capturing = False
            elif frames_since_start % capture_interval == 0:
                filename = f"frame_{session_saved:04d}.png"
                cv2.imwrite(os.path.join(session_dir, filename), frame)
                saved_entries.append({
                    "session": session_tag,
                    "id": session_saved,
                    "video_frame_idx": frame_idx,
                    "filename": filename
                })
                session_saved += 1

        display = frame.copy()
        total_saved = sum(1 for e in saved_entries)
        line1 = f"Frame: {frame_idx}/{total_frames}  |  Interval: {capture_interval}  |  Session: {session_num}  |  Total: {total_saved}"
        cv2.putText(display, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if capturing:
            elapsed_sec = frames_since_start / fps
            remaining = max_capture_duration - elapsed_sec
            status = f"CAPTURING  {elapsed_sec:.1f}s / {max_capture_duration}s  ({remaining:.0f}s left)"
            cv2.putText(display, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Press SPACE to start capture  |  W/S = adjust interval",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Capture Tool", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == 32:  # SPACE
            if not capturing:
                session_num += 1
                session_saved = 0
                session_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                session_dir = os.path.join(FRAMES_DIR, session_tag)
                os.makedirs(session_dir, exist_ok=True)
                capturing = True
                capture_start_frame = frame_idx
                frames_since_start = 0
                est_count = max_capture_frames // capture_interval
                print(f"  Session {session_num} started at frame {frame_idx} "
                      f"-> {session_dir}/  (interval={capture_interval}, ~{est_count} frames)")
            else:
                print(f"  Session {session_num} stopped. Saved {session_saved} frames to {session_dir}/")
                capturing = False

        elif key in (ord('w'), ord('W')):
            capture_interval = min(capture_interval + 5, 200)
            print(f"  Interval: every {capture_interval} frames (~{fps / capture_interval:.1f} per sec)")

        elif key in (ord('s'), ord('S')):
            capture_interval = max(capture_interval - 5, 1)
            print(f"  Interval: every {capture_interval} frames (~{fps / capture_interval:.1f} per sec)")

    index_path = os.path.join(EVAL_DIR, "frame_index.csv")
    with open(index_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["session", "id", "video_frame_idx", "filename"])
        writer.writeheader()
        writer.writerows(saved_entries)

    total_saved = len(saved_entries)
    print(f"\nDone. {total_saved} frames across {session_num} sessions.")
    print(f"Index written to {index_path}")
    sessions = sorted(set(e["session"] for e in saved_entries))
    print(f"\nSession folders:")
    for s in sessions:
        count = sum(1 for e in saved_entries if e["session"] == s)
        print(f"  {FRAMES_DIR}/{s}/  ({count} frames)")
    print(f"\nNext steps:")
    print(f"  1. Review each session folder, delete frames you don't want")
    print(f"  2. Annotate kept frames with SAM3, save masks to {GT_DIR}/")
    print(f"     (same folder structure + filenames, white=ball, black=background)")
    print(f"  3. Run: python evaluate_segmentation.py <video_path>")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
