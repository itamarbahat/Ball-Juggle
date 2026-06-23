import cv2
import numpy as np
import time
import os
import json
import subprocess
import pygame
from camera_manager import DualCameraManager
from ball_processor import BallProcessor
from juggling_logic import JugglingCounter
from config import CONFIG
from config_utils import load_hsv_config, load_floor_points, save_floor_points
from floor_finding import FloorFinder
from dashboard import Dashboard

# Path to this script's directory, so the HSV helper subprocess and configs
# resolve correctly regardless of the working directory.
HERE = os.path.dirname(os.path.abspath(__file__))


def _check_hsv_calibrated():
    """True when both camera HSV profiles exist in ball_config.json."""
    try:
        with open(os.path.join(HERE, "ball_config.json")) as f:
            d = json.load(f)
        return "side" in d and "top" in d
    except Exception:
        return False


def draw_tracking_info(frame, ball_data, status_msg, color=(0, 255, 0)):
    """Draws ball circle, center dot, and status bar on the frame."""
    if ball_data:
        x, y, r = ball_data
        cv2.circle(frame, (x, y), r, color, 2)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Pos: {x},{y}", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.rectangle(frame, (5, 5), (320, 25), (0, 0, 0), -1)
    cv2.putText(frame, f"{status_msg}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def build_mask_view(mask_side, mask_top, panel_h=300):
    """Builds a side-by-side BGR view of the two black/white detection masks, marking the
    largest detected blob. Read-only on the masks — does not affect detection."""
    def _panel(mask, label):
        if mask is None:
            panel = np.zeros((panel_h, panel_h * 2, 3), dtype=np.uint8)
            cv2.putText(panel, "WARMING UP...", (20, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2, cv2.LINE_AA)
            return panel

        bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Mark the largest white blob (what detection treats as the ball candidate)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 5:
                (bx, by), br = cv2.minEnclosingCircle(c)
                cv2.circle(bgr, (int(bx), int(by)), int(br), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(bgr, (int(bx), int(by)), 3, (0, 0, 255), -1, cv2.LINE_AA)

        mh, mw = bgr.shape[:2]
        scale = panel_h / mh
        bgr = cv2.resize(bgr, (int(mw * scale), panel_h), interpolation=cv2.INTER_NEAREST)

        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
        return bgr

    return np.hstack([_panel(mask_side, "SIDE-A - DETECTION"),
                      _panel(mask_top, "SIDE-B - DETECTION")])


def draw_cal_hud(frame, idx, total, name, world_pt, captured_count, warn_text, flash_ok):
    """On-screen heads-up display for guided floor calibration: instructions, the 12-point
    progress dots, plus warning / capture-confirmation flashes."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FLOOR CAL  Point {idx + 1}/{total}: {name}", (10, 22),
                font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"world ({int(world_pt[0])},{int(world_pt[1])}) cm   "
                       f"SPACE=capture   BACKSPACE=undo   ESC=cancel",
                (10, 44), font, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    for k in range(total):
        cx = 14 + k * 18
        filled = k < captured_count
        is_cur = (k == idx)
        color = (0, 200, 0) if filled else ((0, 255, 255) if is_cur else (120, 120, 120))
        cv2.circle(frame, (cx, 56), 6, color, -1 if (filled or is_cur) else 1, cv2.LINE_AA)

    if warn_text:
        sz = cv2.getTextSize(warn_text, font, 0.7, 2)[0]
        cv2.putText(frame, warn_text, ((w - sz[0]) // 2, h - 20), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    elif flash_ok:
        sz = cv2.getTextSize("CAPTURED", font, 0.8, 2)[0]
        cv2.putText(frame, "CAPTURED", ((w - sz[0]) // 2, h - 20), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def perform_flash_calibration(dual_cam, game_logic):
    """
    Captures 15 frames using pure HSV (MOG2 bypassed) to find the median
    ball radius while the ball is sitting still on the floor.
    """
    print("[CAL] Starting radius calibration (15-frame HSV sample)...")

    lower_side, upper_side = load_hsv_config("side")
    lower_top, upper_top = load_hsv_config("top")

    collected_radii_side = []
    collected_radii_top = []

    for _ in range(15):
        frame_top_raw, frame_side_raw = dual_cam.read()
        if frame_top_raw is None: break

        target_w = CONFIG["camera"]["width"]
        target_h = CONFIG["camera"]["height"]
        frame_top = cv2.resize(frame_top_raw, (target_w, target_h))
        frame_side = cv2.resize(frame_side_raw, (target_w, target_h))

        hsv_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2HSV)
        mask_side = cv2.inRange(hsv_side, lower_side, upper_side)
        cnts_side, _ = cv2.findContours(mask_side, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_side:
            valid_side = [c for c in cnts_side if 10 < cv2.minEnclosingCircle(c)[1] < 100]
            if valid_side:
                c = max(valid_side, key=cv2.contourArea)
                ((x, y), r) = cv2.minEnclosingCircle(c)
                collected_radii_side.append((int(x), int(y), int(r)))

        hsv_top = cv2.cvtColor(frame_top, cv2.COLOR_BGR2HSV)
        mask_top = cv2.inRange(hsv_top, lower_top, upper_top)
        cnts_top, _ = cv2.findContours(mask_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_top:
            valid_top = [c for c in cnts_top if 10 < cv2.minEnclosingCircle(c)[1] < 100]
            if valid_top:
                c = max(valid_top, key=cv2.contourArea)
                ((x, y), r) = cv2.minEnclosingCircle(c)
                collected_radii_top.append((int(x), int(y), int(r)))

        time.sleep(0.01)

    final_side = None
    final_top = None

    if collected_radii_side:
        collected_radii_side.sort(key=lambda x: x[2])
        final_side = collected_radii_side[len(collected_radii_side)//2]

    if collected_radii_top:
        collected_radii_top.sort(key=lambda x: x[2])
        final_top = collected_radii_top[len(collected_radii_top)//2]

    if final_side:
        game_logic.set_baseline(final_top, final_side)
        print(f"[CAL] Radius calibration OK. Median radius: {final_side[2]}")
    else:
        print("[CAL] Radius calibration FAILED — ball not found in side view.")


def perform_flash_head_calibration(dual_cam, game_logic):
    """Pure HSV capture for the top camera to set the head-height radius baseline."""
    print("[CAL] Starting header calibration (15-frame HSV sample)...")

    lower_top, upper_top = load_hsv_config("top")
    collected_radii_top = []

    for _ in range(15):
        frame_top_raw, _ = dual_cam.read()
        if frame_top_raw is None: break

        target_w = CONFIG["camera"]["width"]
        target_h = CONFIG["camera"]["height"]
        frame_top = cv2.resize(frame_top_raw, (target_w, target_h))

        hsv_top = cv2.cvtColor(frame_top, cv2.COLOR_BGR2HSV)
        mask_top = cv2.inRange(hsv_top, lower_top, upper_top)

        cnts_top, _ = cv2.findContours(mask_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_top:
            valid_top = [c for c in cnts_top if 10 < cv2.minEnclosingCircle(c)[1] < 100]
            if valid_top:
                c = max(valid_top, key=cv2.contourArea)
                ((x, y), r) = cv2.minEnclosingCircle(c)
                collected_radii_top.append((int(x), int(y), int(r)))

        time.sleep(0.01)

    if collected_radii_top:
        collected_radii_top.sort(key=lambda x: x[2])
        final_top = collected_radii_top[len(collected_radii_top)//2]
        game_logic.set_head_baseline(final_top)
        print(f"[CAL] Header calibration OK. Head radius: {final_top[2]}")
    else:
        print("[CAL] Header calibration FAILED — ball not seen by top camera.")


def perform_floor_calibration(dual_cam, processor_top, processor_side, game_logic):
    """
    Interactive 12-point floor calibration using pure HSV detection.
    Captures ball positions from both cameras, computes homography,
    and updates the FloorFinder.
    """
    NUM_POINTS = 12

    world_points = np.array([
        # Row 1 — Front (y=0): 4 points across
        [0, 0],       # Point 1:  Front-Left
        [33, 0],      # Point 2:  Front-Center-Left
        [67, 0],      # Point 3:  Front-Center-Right
        [100, 0],     # Point 4:  Front-Right
        # Row 2 — Middle (y=50): 4 points across
        [0, 50],      # Point 5:  Middle-Left
        [33, 50],     # Point 6:  Middle-Center-Left
        [67, 50],     # Point 7:  Middle-Center-Right
        [100, 50],    # Point 8:  Middle-Right
        # Row 3 — Back (y=100): 4 points across
        [0, 100],     # Point 9:  Back-Left
        [33, 100],    # Point 10: Back-Center-Left
        [67, 100],    # Point 11: Back-Center-Right
        [100, 100]    # Point 12: Back-Right
    ], dtype=np.float32)

    point_names = [
        "FRONT-LEFT",
        "FRONT-CENTER-LEFT",
        "FRONT-CENTER-RIGHT",
        "FRONT-RIGHT",
        "MIDDLE-LEFT",
        "MIDDLE-CENTER-LEFT",
        "MIDDLE-CENTER-RIGHT",
        "MIDDLE-RIGHT",
        "BACK-LEFT",
        "BACK-CENTER-LEFT",
        "BACK-CENTER-RIGHT",
        "BACK-RIGHT"
    ]

    captured_top = []
    captured_side = []

    target_w = CONFIG["camera"]["width"]
    target_h = CONFIG["camera"]["height"]

    print()
    print("=" * 50)
    print(f"  FLOOR CALIBRATION ({NUM_POINTS} Points)")
    print("=" * 50)
    print("  Place the ball on the floor at each position.")
    print("  Detection uses pure HSV color (MOG2 disabled).")
    print("  SPACE = capture point   |   ESC = cancel")
    print("=" * 50)

    processor_top.disable_mog_temporarily(99999)
    processor_side.disable_mog_temporarily(99999)

    lower_side, upper_side = load_hsv_config("side")
    lower_top, upper_top = load_hsv_config("top")

    # Guided capture loop. A `while` (rather than `for`) lets BACKSPACE step back for undo.
    i = 0
    warn_until = 0.0
    flash_until = 0.0

    while i < NUM_POINTS:
        frame_top_raw, frame_side_raw = dual_cam.read()
        if frame_top_raw is None or frame_side_raw is None:
            break

        frame_top = cv2.resize(frame_top_raw, (target_w, target_h))
        frame_side = cv2.resize(frame_side_raw, (target_w, target_h))

        ball_top = _detect_ball_hsv(frame_top, lower_top, upper_top)
        ball_side = _detect_ball_hsv(frame_side, lower_side, upper_side)

        display_top = frame_top.copy()
        display_side = frame_side.copy()

        now = time.time()
        warn = "BALL NOT VISIBLE IN BOTH CAMERAS" if now < warn_until else None
        flash_ok = now < flash_until

        draw_cal_hud(display_top, i, NUM_POINTS, point_names[i], world_points[i],
                     len(captured_top), warn, flash_ok)
        draw_cal_hud(display_side, i, NUM_POINTS, point_names[i], world_points[i],
                     len(captured_side), warn, flash_ok)

        if ball_top:
            cv2.circle(display_top, (ball_top[0], ball_top[1]), ball_top[2], (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(display_top, (ball_top[0], ball_top[1]), 5, (0, 0, 255), -1)

        if ball_side:
            cv2.circle(display_side, (ball_side[0], ball_side[1]), ball_side[2], (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(display_side, (ball_side[0], ball_side[1]), 5, (0, 0, 255), -1)

        for j, (pt, ps) in enumerate(zip(captured_top, captured_side)):
            cv2.circle(display_top, (int(pt[0]), int(pt[1])), 8, (255, 0, 255), -1)
            cv2.putText(display_top, str(j+1), (int(pt[0])+10, int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(display_side, (int(ps[0]), int(ps[1])), 8, (255, 0, 255), -1)
            cv2.putText(display_side, str(j+1), (int(ps[0])+10, int(ps[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Floor Cal - Camera Side A", display_side)
        cv2.imshow("Floor Cal - Camera Side B", display_top)

        key = cv2.waitKey(33) & 0xFF

        if key == 27:  # ESC - cancel
            print("[FLOOR_CAL] Cancelled by user.")
            processor_top.disable_mog_temporarily(0)
            processor_side.disable_mog_temporarily(0)
            return None

        if key == 8:  # BACKSPACE - undo last captured point
            if captured_top:
                captured_top.pop()
                captured_side.pop()
                i -= 1
                print(f"[FLOOR_CAL] Undo — back to point {i+1}.")
            continue

        if key == 32:  # SPACE - capture
            if ball_top and ball_side:
                captured_top.append([ball_top[0], ball_top[1]])
                captured_side.append([ball_side[0], ball_side[1]])
                print(f"[FLOOR_CAL] Point {i+1} captured: TOP=({ball_top[0]},{ball_top[1]})  SIDE=({ball_side[0]},{ball_side[1]})")
                flash_until = time.time() + CONFIG["ui"]["capture_flash_sec"]
                i += 1
            else:
                missing = []
                if not ball_top: missing.append("SIDE-B")
                if not ball_side: missing.append("SIDE-A")
                print(f"[FLOOR_CAL] Ball not detected in: {', '.join(missing)}. Reposition and retry.")
                warn_until = time.time() + CONFIG["ui"]["warn_sec"]

    processor_top.disable_mog_temporarily(0)
    processor_side.disable_mog_temporarily(0)

    # Stream ended before all points were captured — abort without saving.
    if len(captured_top) < NUM_POINTS:
        print("[FLOOR_CAL] Incomplete — calibration aborted.")
        return None

    pts_cam_top = np.array(captured_top, dtype=np.float32)
    pts_cam_side = np.array(captured_side, dtype=np.float32)

    save_floor_points(world_points, pts_cam_top, pts_cam_side)
    new_floor_finder = FloorFinder(world_points, pts_cam_top, pts_cam_side)

    if new_floor_finder.calibrated:
        game_logic.floor_finder = new_floor_finder
        print("[FLOOR_CAL] Floor calibration complete and active.")
        result_text, result_color = "FLOOR CALIBRATED", (0, 220, 0)
    else:
        print("[FLOOR_CAL] WARNING — Homography failed. Check point positions.")
        result_text, result_color = "HOMOGRAPHY FAILED - RECALIBRATE", (0, 0, 255)

    # Show the outcome on-screen (instead of console only) for a couple of seconds.
    result_end = time.time() + CONFIG["ui"]["result_sec"]
    while time.time() < result_end:
        ft_raw, fs_raw = dual_cam.read()
        if ft_raw is None or fs_raw is None:
            break
        fs = cv2.resize(fs_raw, (target_w, target_h))
        ft = cv2.resize(ft_raw, (target_w, target_h))
        for disp in (fs, ft):
            ov = disp.copy()
            cv2.rectangle(ov, (0, 0), (target_w, target_h), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.45, disp, 0.55, 0, disp)
            sz = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            cv2.putText(disp, result_text, ((target_w - sz[0]) // 2, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 3, cv2.LINE_AA)
        cv2.imshow("Floor Cal - Camera Side A", fs)
        cv2.imshow("Floor Cal - Camera Side B", ft)
        cv2.waitKey(33)

    return new_floor_finder


def _detect_ball_hsv(frame, lower_hsv, upper_hsv):
    """Pure HSV ball detection. Returns (x, y, r) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        valid = [c for c in cnts if 10 < cv2.minEnclosingCircle(c)[1] < 100]
        if valid:
            c = max(valid, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            return (int(x), int(y), int(r))
    return None


def draw_floor_overlay(frame, pts, color=(0, 180, 0), alpha=0.3):
    """Draws the calibrated 12-point floor grid with semi-transparent fill."""
    if pts is None or len(pts) < 12:
        return

    pts_int = pts.astype(int)
    overlay = frame.copy()

    corners = np.array([pts_int[0], pts_int[3], pts_int[11], pts_int[8]], dtype=np.int32)
    cv2.fillPoly(overlay, [corners], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for row_start in (0, 4, 8):
        for i in range(row_start, row_start + 3):
            pt1 = tuple(pts_int[i])
            pt2 = tuple(pts_int[i + 1])
            cv2.line(frame, pt1, pt2, color, 1)

    for col in range(4):
        for row in range(2):
            pt1 = tuple(pts_int[row * 4 + col])
            pt2 = tuple(pts_int[(row + 1) * 4 + col])
            cv2.line(frame, pt1, pt2, color, 1)

    for i, pt in enumerate(pts_int):
        cv2.circle(frame, tuple(pt), 4, (0, 255, 255), -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)


def print_welcome_instructions():
    """Prints calibration steps and keyboard reference."""
    print("=" * 58)
    print("        JUGGLING COUNTER SYSTEM — User Guide")
    print("=" * 58)
    print()
    print("  CALIBRATION (run these before playing):")
    print("  -----------------------------------------")
    print("  0. HSV COLOR     Press '1' (opens color helper).")
    print("  1. BACKGROUND    Step out of frame, press 'B'.")
    print("  2. RADIUS        Place ball on floor, press 'S'.")
    print("  3. FLOOR (12pt)  Press 'F', place ball at 12 spots,")
    print("                   press SPACE at each. ESC to cancel.")
    print("  4. FLOOR LINE    Adjust with A/Z, UP/DOWN, [/], '/\\")
    print()
    print("  GAMEPLAY:")
    print("  -----------------------------------------")
    print("  T          Start 3-second countdown & play")
    print("  N          Switch player (saves score)")
    print("  R          Reset score")
    print()
    print("  FLOOR ADJUSTMENT:")
    print("  -----------------------------------------")
    print("  A / Z      Raise / lower LEFT anchor")
    print("  UP / DOWN  Raise / lower RIGHT anchor")
    print("  K / M      Raise / lower RIGHT anchor (fallback)")
    print("  [ / ]      Move LEFT boundary")
    print("  ' / \\      Move RIGHT boundary")
    print("  - / +      Floor epsilon  -/+ 0.5 cm")
    print()
    print("  OTHER:")
    print("  -----------------------------------------")
    print("  1          Launch HSV color calibration helper")
    print("  B          Re-capture background (reset MOG2)")
    print("  S          Recalibrate ball radius")
    print("  F          Floor calibration (12-point)")
    print("  D          Toggle color-only detection")
    print("  G          Toggle floor grid overlay")
    print("  V          Toggle B/W detection-mask window")
    print("  C          Clear evaluation log")
    print("  ESC        Exit program")
    print("=" * 58)


def main():
    """Main loop: reads dual-camera frames, processes, detects events, and displays."""

    print_welcome_instructions()

    dual_cam = DualCameraManager().start()
    processor_top = BallProcessor(camera_profile="top")
    processor_side = BallProcessor(camera_profile="side")
    floor_finder = FloorFinder()
    game_logic = JugglingCounter(history_len=10, floor_finder=floor_finder)
    print(f"[INFO] Floor epsilon: {game_logic.floor_epsilon_cm:.2f} cm")

    target_w = CONFIG["camera"]["width"]
    target_h = CONFIG["camera"]["height"]

    is_video_top = isinstance(CONFIG["camera"]["top_source"], str)
    is_video_side = isinstance(CONFIG["camera"]["side_source"], str)

    _, floor_pts_top, floor_pts_side = load_floor_points()
    show_floor_overlay = True
    floor_flash_until = 0.0
    color_only_mode = False

    active_player = 1
    score_p1 = 0
    score_p2 = 0

    ui = CONFIG["ui"]

    # --- Dashboard (pure presentation + input; no game logic) ---
    dash = Dashboard()
    dash.fps = 30 if (is_video_top or is_video_side) else 60
    dash.hit_popup_sec = ui["hit_popup_sec"]
    dash.cal["hsv"] = _check_hsv_calibrated()
    dash.cal["floor"] = getattr(floor_finder, "calibrated", False)
    dash.cal["radius"] = game_logic.baseline.get("is_set", False)

    # B/W detection-mask debug window (toggled with V). Lives as a separate cv2
    # window; needs cv2.waitKey to pump its GUI loop, handled in the main loop.
    show_mask_window = ui["show_mask_window"]
    mask_window_placed = False

    # game-state timers read by the state machine below
    gameover_until = 0.0
    go_until = 0.0
    prev_countdown = False

    running = True
    while running:
        frame_top_raw, frame_side_raw = dual_cam.read()

        if frame_top_raw is None or frame_side_raw is None:
            print("[WARNING] End of stream or camera failure.")
            break

        frame_top = cv2.resize(frame_top_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frame_side = cv2.resize(frame_side_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if color_only_mode:
            processor_top.mog_bypass_frames = max(processor_top.mog_bypass_frames, 2)
            processor_side.mog_bypass_frames = max(processor_side.mog_bypass_frames, 2)

        # Side camera uses strict (Hough only); top camera relaxes if side sees the ball
        data_side, mask_side, status_side = processor_side.process(frame_side, detection_mode="strict")
        top_mode = "relaxed" if data_side is not None else "strict"
        data_top, mask_top, status_top = processor_top.process(frame_top, detection_mode=top_mode)
        status_top += f" [{top_mode.upper()}]"

        current_count, logic_feedback = game_logic.update(data_top, data_side, frame_width=target_w, frame_height=target_h)

        now = time.time()

        # --- event timers (read the logic's outputs; do not change them) ---
        if logic_feedback and "DROP" in logic_feedback:
            floor_flash_until = now + 0.5
            dash.drop_alert_until = now + ui["drop_flash_sec"]
            gameover_until = now + ui["gameover_sec"]

        if logic_feedback and "+1" in logic_feedback:
            dash.hit_popup_text = logic_feedback
            dash.hit_popup_time = now

        # Detect the countdown -> live transition to flash "GO!"
        if prev_countdown and not game_logic.countdown_active and game_logic.game_active:
            go_until = now + 0.6
            dash.go_until = go_until
        prev_countdown = game_logic.countdown_active

        # --- tracking overlays on both feeds (drawn on the numpy frames) ---
        draw_tracking_info(frame_side, data_side, f"SIDE-A: {status_side}", (255, 255, 0))
        draw_tracking_info(frame_top, data_top, f"SIDE-B: {status_top}", (0, 255, 0))

        if color_only_mode:
            cv2.putText(frame_side, "COLOR-ONLY (D)", (8, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_top, "COLOR-ONLY (D)", (8, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)

        if show_floor_overlay and not color_only_mode:
            floor_is_red = now < floor_flash_until
            top_color = (0, 0, 220) if floor_is_red else (0, 180, 0)
            side_color = (0, 0, 220) if floor_is_red else (180, 100, 0)
            top_alpha = 0.45 if floor_is_red else 0.3
            side_alpha = 0.45 if floor_is_red else 0.3
            draw_floor_overlay(frame_top, floor_pts_top, color=top_color, alpha=top_alpha)
            draw_floor_overlay(frame_side, floor_pts_side, color=side_color, alpha=side_alpha)

        # --- game-state machine -> dashboard badge + countdown text ---
        if game_logic.countdown_active:
            elapsed = now - game_logic.countdown_start_time
            dash.game_state = "COUNTDOWN"
            dash.countdown_text = str(max(1, 3 - int(elapsed)))
        elif now < go_until:
            dash.game_state = "LIVE"
            dash.countdown_text = ""
        elif game_logic.game_active:
            dash.game_state = "LIVE"
            dash.countdown_text = ""
        elif now < gameover_until:
            dash.game_state = "GAME OVER"
            dash.countdown_text = ""
        elif not game_logic.baseline["is_set"]:
            dash.game_state = "WAITING"
            dash.countdown_text = ""
        else:
            dash.game_state = "READY"
            dash.countdown_text = ""

        # --- sync plain display state to the dashboard (no logic here) ---
        dash.score_p1 = score_p1
        dash.score_p2 = score_p2
        dash.current_count = current_count
        dash.active_player = active_player
        dash.cal["radius"] = game_logic.baseline.get("is_set", False)
        dash.floor_epsilon_cm = game_logic.floor_epsilon_cm
        dash.color_only = color_only_mode

        # --- render the unified dashboard (Side A = main, Side B = secondary) ---
        dash.draw(frame_side, frame_top)

        # --- optional B/W mask debug window (separate cv2 window) ---
        if show_mask_window:
            cv2.imshow("Detection (B/W Mask)", build_mask_view(mask_side, mask_top))
            if not mask_window_placed:
                cv2.moveWindow("Detection (B/W Mask)", 40, 600)
                mask_window_placed = True
            cv2.waitKey(1)   # pump the cv2 GUI loop so the window refreshes

        # ------------------------------------------------------------------ #
        #  Input — all keys now come from pygame                             #
        # ------------------------------------------------------------------ #
        for key in dash.poll_events():
            if key == pygame.K_ESCAPE:
                running = False
                break

            elif key == pygame.K_1:
                print("[INPUT] Launching HSV Calibration Helper...")
                dash.set_toast("Releasing cameras for HSV helper...",
                               time.time() + ui["toast_sec"])
                dash.draw(frame_side, frame_top)
                # Free the camera handles so the helper subprocess can open them,
                # then reacquire afterwards (the HSV GUI needs exclusive access).
                dual_cam.stop()
                try:
                    subprocess.call(["python", "Calibration Helper script.py"], cwd=HERE)
                except Exception as e:
                    print(f"[ERROR] Could not launch HSV helper: {e}")
                dual_cam = DualCameraManager().start()
                # Warm up the reopened cameras before the main loop's None-check.
                for _ in range(30):
                    wt, ws = dual_cam.read()
                    if wt is not None and ws is not None:
                        break
                    time.sleep(0.03)
                dash.cal["hsv"] = _check_hsv_calibrated()
                dash.set_toast("HSV calibration updated", time.time() + ui["toast_sec"])
                pygame.event.clear()

            elif key == pygame.K_b:
                print("[INPUT] Resetting MOG2 background models...")
                processor_top.set_instant_background(frame_top)
                processor_side.set_instant_background(frame_side)
                dash.cal["background"] = True
                dash.set_toast("Background reset", time.time() + ui["toast_sec"])

            elif key == pygame.K_s:
                processor_top.disable_mog_temporarily(15)
                processor_side.disable_mog_temporarily(15)
                perform_flash_calibration(dual_cam, game_logic)
                dash.cal["radius"] = game_logic.baseline.get("is_set", False)
                dash.set_toast("Ball radius calibrated", time.time() + ui["toast_sec"])

            elif key == pygame.K_r:
                print("[INPUT] Resetting counter.")
                game_logic.reset()
                dash.set_toast("Counter reset", time.time() + ui["toast_sec"])

            elif key == pygame.K_h:
                print("[INPUT] Calibrating header radius...")
                processor_top.disable_mog_temporarily(15)
                perform_flash_head_calibration(dual_cam, game_logic)
                dash.set_toast("Header calibrated", time.time() + ui["toast_sec"])

            elif key == pygame.K_c:
                game_logic.clear_evaluation_log()
                dash.set_toast("Evaluation log cleared", time.time() + ui["toast_sec"])

            elif key == pygame.K_f:
                print("[INPUT] Starting floor calibration...")
                new_ff = perform_floor_calibration(dual_cam, processor_top, processor_side, game_logic)
                if new_ff and new_ff.calibrated:
                    floor_finder = new_ff
                    _, floor_pts_top, floor_pts_side = load_floor_points()
                    dash.cal["floor"] = True
                # Close the temporary floor-cal cv2 windows and discard queued keys.
                for wname in ("Floor Cal - Camera Side A", "Floor Cal - Camera Side B"):
                    try:
                        cv2.destroyWindow(wname)
                    except Exception:
                        pass
                cv2.waitKey(1)
                pygame.event.clear()

            elif key == pygame.K_d:
                color_only_mode = not color_only_mode
                if not color_only_mode:
                    processor_top.mog_bypass_frames = 0
                    processor_side.mog_bypass_frames = 0
                state = "ON" if color_only_mode else "OFF"
                print(f"[INPUT] Color-only detection: {state}")
                dash.set_toast(f"Color-only detection: {state}", time.time() + ui["toast_sec"])

            elif key == pygame.K_g:
                show_floor_overlay = not show_floor_overlay
                state = "ON" if show_floor_overlay else "OFF"
                print(f"[INPUT] Floor overlay: {state}")
                dash.set_toast(f"Floor grid: {state}", time.time() + ui["toast_sec"])

            elif key == pygame.K_v:
                show_mask_window = not show_mask_window
                if not show_mask_window:
                    cv2.destroyWindow("Detection (B/W Mask)")
                    cv2.waitKey(1)
                    mask_window_placed = False
                state = "ON" if show_mask_window else "OFF"
                print(f"[INPUT] Detection mask window: {state}")
                dash.set_toast(f"Mask window: {state}", time.time() + ui["toast_sec"])

            elif key == pygame.K_n:
                if active_player == 1:
                    score_p1 = game_logic.count
                    active_player = 2
                    print("[GAME] Switched to Player 2")
                    dash.set_toast("Player 2's turn", time.time() + ui["toast_sec"])
                else:
                    score_p2 = game_logic.count
                    active_player = 1
                    print("[GAME] Switched to Player 1")
                    dash.set_toast("Player 1's turn", time.time() + ui["toast_sec"])

                    # Trigger the (non-blocking) winner overlay after both players have played
                    if score_p1 > 0 or score_p2 > 0:
                        if score_p1 > score_p2:
                            winner_text = f"P1 WINS!  {score_p1} - {score_p2}"
                        elif score_p2 > score_p1:
                            winner_text = f"P2 WINS!  {score_p2} - {score_p1}"
                        else:
                            winner_text = f"DRAW!  {score_p1} - {score_p2}"
                        print(f"[GAME] {winner_text}")
                        dash.winner_text = winner_text
                        dash.winner_until = time.time() + ui["winner_sec"]

                game_logic.reset()

            elif key == pygame.K_z:
                game_logic.adjust_left_anchor(5)
            elif key == pygame.K_a:
                game_logic.adjust_left_anchor(-5)

            elif key == pygame.K_DOWN:
                game_logic.adjust_right_anchor(5)
            elif key == pygame.K_UP:
                game_logic.adjust_right_anchor(-5)
            elif key == pygame.K_k:
                game_logic.adjust_right_anchor(-5)
            elif key == pygame.K_m:
                game_logic.adjust_right_anchor(5)

            elif key == pygame.K_LEFTBRACKET:
                game_logic.adjust_floor_start_x(-10)
            elif key == pygame.K_RIGHTBRACKET:
                game_logic.adjust_floor_start_x(10)

            elif key == pygame.K_QUOTE:
                game_logic.adjust_floor_end_x(-10)
            elif key == pygame.K_BACKSLASH:
                game_logic.adjust_floor_end_x(10)

            elif key == pygame.K_t:
                if game_logic.start_game():
                    dash.set_toast("Get ready!", time.time() + ui["toast_sec"])
                else:
                    dash.set_toast("Calibrate first (S)", time.time() + ui["toast_sec"])

            elif key == pygame.K_MINUS or key == pygame.K_KP_MINUS:
                game_logic.adjust_floor_epsilon(-0.5)
            elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                game_logic.adjust_floor_epsilon(0.5)

    dual_cam.stop()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
