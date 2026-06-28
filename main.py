import cv2
import numpy as np
import time
from camera_manager import DualCameraManager
from ball_processor import BallProcessor
from juggling_logic import JugglingCounter
from config import CONFIG
from Utils.config_utils import load_hsv_config, load_floor_points, save_floor_points, generate_world_points, generate_point_names
from floor_finding import FloorFinder

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

def draw_game_score(frame, current_count, p1_score, p2_score, active_player, feedback):
    """Overlays multiplayer scores and non-hit event feedback (DROP, countdown, etc.)."""
    h, w = frame.shape[:2]
    
    COLOR_P1_ACTIVE = (180, 0, 0)
    COLOR_P2_ACTIVE = (0, 130, 0)
    COLOR_INACTIVE = (80, 80, 80)

    c1 = COLOR_P1_ACTIVE if active_player == 1 else COLOR_INACTIVE
    s1 = current_count if active_player == 1 else p1_score
    pref1 = ">" if active_player == 1 else " "
    txt1 = f"{pref1}P1:{s1}"

    c2 = COLOR_P2_ACTIVE if active_player == 2 else COLOR_INACTIVE
    s2 = current_count if active_player == 2 else p2_score
    pref2 = ">" if active_player == 2 else " "
    txt2 = f"{pref2}P2:{s2}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thick = 3
    sz1 = cv2.getTextSize(txt1, font, scale, thick)[0]
    sz2 = cv2.getTextSize(txt2, font, scale, thick)[0]
    box_w = max(sz1[0], sz2[0]) + 6
    line_h = sz1[1]
    gap = 8
    box_h = line_h * 2 + gap + 6
    pad = 3

    cv2.rectangle(frame, (w - box_w - pad, pad), (w - pad, pad + box_h), (0, 0, 0), -1)
    cv2.putText(frame, txt1, (w - box_w, pad + line_h + 2), font, scale, c1, thick)
    cv2.putText(frame, txt2, (w - box_w, pad + line_h * 2 + gap + 2), font, scale, c2, thick)
    
    # Only show non-hit feedback (DROP, countdown, game over) as persistent center text
    if feedback and "+1" not in feedback and feedback != "Tracking":
        text_color = (0, 0, 255)
        text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(frame, feedback, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, text_color, 4)


def draw_hit_popup(frame, text, elapsed, duration=1.0):
    """Draws a fading '+1' popup that drifts upward and fades out over `duration` seconds."""
    if elapsed >= duration:
        return
    
    h, w = frame.shape[:2]
    progress = elapsed / duration          # 0.0 → 1.0
    alpha = 1.0 - progress                 # 1.0 → 0.0
    drift = int(30 * progress)             # text rises ~30px over the duration
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.5
    thick = 5
    sz = cv2.getTextSize(text, font, scale, thick)[0]
    tx = (w - sz[0]) // 2
    ty = (h + sz[1]) // 2 - drift
    
    overlay = frame.copy()
    cv2.putText(overlay, text, (tx, ty), font, scale, (0, 255, 0), thick)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

def perform_flash_calibration(dual_cam, game_logic):
    """
    Captures 15 frames using pure HSV (MOG2 bypassed) to find the median
    ball radius while the ball is sitting still on the floor.
    """
    print("[CAL] Starting radius calibration (15-frame HSV sample)...")
    
    lower_side, upper_side = load_hsv_config("side")
    lower_main, upper_main = load_hsv_config("main")
    
    collected_radii_side = []
    collected_radii_main = []
    
    for _ in range(15):
        frame_main_raw, frame_side_raw = dual_cam.read()
        if frame_main_raw is None: break
        
        target_w = CONFIG["camera"]["width"]
        target_h = CONFIG["camera"]["height"]
        frame_main = cv2.resize(frame_main_raw, (target_w, target_h))
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

        hsv_main = cv2.cvtColor(frame_main, cv2.COLOR_BGR2HSV)
        mask_main = cv2.inRange(hsv_main, lower_main, upper_main)
        cnts_main, _ = cv2.findContours(mask_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_main:
            valid_main = [c for c in cnts_main if 10 < cv2.minEnclosingCircle(c)[1] < 100]
            if valid_main:
                c = max(valid_main, key=cv2.contourArea)
                ((x, y), r) = cv2.minEnclosingCircle(c)
                collected_radii_main.append((int(x), int(y), int(r)))
            
        time.sleep(0.01)

    final_side = None
    final_main = None
    
    if collected_radii_side:
        collected_radii_side.sort(key=lambda x: x[2])
        final_side = collected_radii_side[len(collected_radii_side)//2]
        
    if collected_radii_main:
        collected_radii_main.sort(key=lambda x: x[2])
        final_main = collected_radii_main[len(collected_radii_main)//2]
        
    if final_side:
        print(f"[CAL] Radius calibration OK. Median radius: {final_side[2]}")
    else:
        print("[CAL] Radius calibration FAILED — ball not found in side view.")


def perform_flash_header_calibration(header_cap, game_logic):
    """Pure HSV capture for the optional header camera to set the head-height radius baseline."""
    if header_cap is None:
        print("[WARNING] No header camera available for header calibration.")
        return

    print("[CAL] Starting header calibration (15-frame HSV sample)...")
    lower_header, upper_header = load_hsv_config("header")
    collected_radii_header = []
    
    for _ in range(15):
        ret, frame_header_raw = header_cap.read()
        if not ret or frame_header_raw is None:
            break
        
        target_w = CONFIG["camera"]["width"]
        target_h = CONFIG["camera"]["height"]
        frame_header = cv2.resize(frame_header_raw, (target_w, target_h))
        
        hsv_header = cv2.cvtColor(frame_header, cv2.COLOR_BGR2HSV)
        mask_header = cv2.inRange(hsv_header, lower_header, upper_header)
        
        cnts_header, _ = cv2.findContours(mask_header, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_header:
            valid_header = [c for c in cnts_header if 10 < cv2.minEnclosingCircle(c)[1] < 100]
            if valid_header:
                c = max(valid_header, key=cv2.contourArea)
                ((x, y), r) = cv2.minEnclosingCircle(c)
                collected_radii_header.append((int(x), int(y), int(r)))
            
        time.sleep(0.01)

    if collected_radii_header:
        collected_radii_header.sort(key=lambda x: x[2])
        final_header = collected_radii_header[len(collected_radii_header)//2]
        game_logic.calibrate_head_height(final_header[2])
        print(f"[CAL] Header calibration OK. Head radius: {final_header[2]}")
    else:
        print("[CAL] Header calibration FAILED — ball not seen by header camera.")


def perform_floor_calibration(dual_cam, processor_main, processor_side, game_logic):
    """
    Interactive floor calibration using pure HSV detection.
    Dynamically generates calibration grid points based on configuration.
    Captures ball positions from both cameras, computes homography,
    and updates the FloorFinder.
    """
    # Dynamically generate world points and their descriptive names
    world_points = generate_world_points(CONFIG)
    point_names = generate_point_names(CONFIG)
    NUM_POINTS = len(world_points)
    
    captured_main = []
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
    
    processor_main.disable_mog_temporarily(99999)
    processor_side.disable_mog_temporarily(99999)
    
    lower_side, upper_side = load_hsv_config("side")
    lower_main, upper_main = load_hsv_config("main")
    
    for i in range(NUM_POINTS):
        print(f"[FLOOR_CAL] Point {i+1}/{NUM_POINTS}: {point_names[i]}  (world: {world_points[i].tolist()} cm)")
        print(f"            Press SPACE when ball is in position...")
        
        while True:
            frame_main_raw, frame_side_raw = dual_cam.read()
            if frame_main_raw is None or frame_side_raw is None:
                break
                
            frame_main = cv2.resize(frame_main_raw, (target_w, target_h))
            frame_side = cv2.resize(frame_side_raw, (target_w, target_h))
            
            ball_main = _detect_ball_hsv(frame_main, lower_main, upper_main)
            ball_side = _detect_ball_hsv(frame_side, lower_side, upper_side)

            display_main = frame_main.copy()
            display_side = frame_side.copy()
            
            cv2.putText(display_main, f"FLOOR CAL: Point {i+1}/{NUM_POINTS} - {point_names[i]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_side, f"FLOOR CAL: Point {i+1}/{NUM_POINTS} - {point_names[i]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if ball_main:
                cv2.circle(display_main, (ball_main[0], ball_main[1]), ball_main[2], (0, 255, 0), 2)
                cv2.circle(display_main, (ball_main[0], ball_main[1]), 5, (0, 0, 255), -1)
                cv2.putText(display_main, f"({ball_main[0]}, {ball_main[1]})", 
                           (ball_main[0]+10, ball_main[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            if ball_side:
                cv2.circle(display_side, (ball_side[0], ball_side[1]), ball_side[2], (0, 255, 0), 2)
                cv2.circle(display_side, (ball_side[0], ball_side[1]), 5, (0, 0, 255), -1)
                cv2.putText(display_side, f"({ball_side[0]}, {ball_side[1]})", 
                           (ball_side[0]+10, ball_side[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            for j, (pt, ps) in enumerate(zip(captured_main, captured_side)):
                cv2.circle(display_main, (int(pt[0]), int(pt[1])), 8, (255, 0, 255), -1)
                cv2.putText(display_main, str(j+1), (int(pt[0])+10, int(pt[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.circle(display_side, (int(ps[0]), int(ps[1])), 8, (255, 0, 255), -1)
                cv2.putText(display_side, str(j+1), (int(ps[0])+10, int(ps[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            cv2.imshow("Main - SIDE View (Master)", display_side)
            cv2.imshow("Main - FRONT View (Main)", display_main)
            
            key = cv2.waitKey(33) & 0xFF
            
            if key == 27:  # ESC - cancel
                print("[FLOOR_CAL] Cancelled by user.")
                processor_main.disable_mog_temporarily(0)
                processor_side.disable_mog_temporarily(0)
                return None
            
            if key == 32:  # SPACE - capture
                if ball_main and ball_side:
                    captured_main.append([ball_main[0], ball_main[1]])
                    captured_side.append([ball_side[0], ball_side[1]])
                    print(f"[FLOOR_CAL] Point {i+1} captured: MAIN=({ball_main[0]},{ball_main[1]})  SIDE=({ball_side[0]},{ball_side[1]})")
                    break
                else:
                    missing = []
                    if not ball_main: missing.append("MAIN")
                    if not ball_side: missing.append("SIDE")
                    print(f"[FLOOR_CAL] Ball not detected in: {', '.join(missing)}. Reposition and retry.")
    
    processor_main.disable_mog_temporarily(0)
    processor_side.disable_mog_temporarily(0)
    
    pts_cam_main = np.array(captured_main, dtype=np.float32)
    pts_cam_side = np.array(captured_side, dtype=np.float32)
    
    save_floor_points(world_points, pts_cam_main, pts_cam_side)
    new_floor_finder = FloorFinder(world_points, pts_cam_main, pts_cam_side)
    
    if new_floor_finder.calibrated:
        game_logic.floor_finder = new_floor_finder
        print("[FLOOR_CAL] Floor calibration complete and active.")
        print("[FLOOR_CAL] Main camera points:")
        for i, pt in enumerate(pts_cam_main.tolist(), start=1):
            print(f"  P{i}: {pt}")
        print("[FLOOR_CAL] Side camera points:")
        for i, pt in enumerate(pts_cam_side.tolist(), start=1):
            print(f"  P{i}: {pt}")
    else:
        print("[FLOOR_CAL] WARNING — Homography failed. Check point positions.")
    
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
    print("  1. BACKGROUND   Step out of frame, press 'B'.")
    print("  2. RADIUS        Place ball on floor, press 'S'.")
    print("  3. FLOOR (12pt)  Press 'F', place ball at 12 spots,")
    print("                   press SPACE at each. ESC to cancel.")
    print("  4. FLOOR LINE    Adjust with A/Z, UP/DOWN, [/], '/\\")
    print("  5. HEADER (opt)  Hold ball on head, press 'H'.")
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
    print("  B          Re-capture background (reset MOG2)")
    print("  S          Recalibrate ball radius")
    print("  F          Floor calibration (12-point)")
    print("  H          Calibrate header height")
    print("  D          Toggle color-only detection")
    print("  G          Toggle floor grid overlay")
    print("  C          Clear evaluation log")
    print("  ESC        Exit program")
    print("=" * 58)

def main():
    """Main loop: reads dual-camera frames, processes, detects events, and displays."""
    
    print_welcome_instructions()

    dual_cam = DualCameraManager().start()
    processor_main = BallProcessor(camera_profile="main")
    processor_side = BallProcessor(camera_profile="side")
    processor_header = None
    header_cap = None
    has_header_cam = False
    floor_finder = FloorFinder()
    game_logic = JugglingCounter(history_len=10, floor_finder=floor_finder)
    print(f"[INFO] Floor epsilon: {game_logic.floor_epsilon_cm:.2f} cm")
    
    target_w = CONFIG["camera"]["width"]
    target_h = CONFIG["camera"]["height"]
    
    header_source = CONFIG["camera"].get("header_source")
    if header_source is not None:
        header_cap = cv2.VideoCapture(header_source)
        header_cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        header_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        if header_cap.isOpened():
            processor_header = BallProcessor(camera_profile="header")
            has_header_cam = True
            print(f"[INFO] Header camera opened from {header_source}")
        else:
            print(f"[WARNING] Header camera failed to open: {header_source}")
            header_cap.release()
            header_cap = None
    
    main_src = CONFIG["camera"].get("main_source", CONFIG["camera"].get("top_source"))
    is_video_main = isinstance(main_src, str)
    is_video_side = isinstance(CONFIG["camera"]["side_source"], str)
    delay_ms = 33 if (is_video_main or is_video_side) else 1

    _, floor_pts_main, floor_pts_side = load_floor_points()
    show_floor_overlay = True
    floor_flash_until = 0.0
    color_only_mode = False
    
    hit_popup_text = ""
    hit_popup_time = 0.0

    active_player = 1
    score_p1 = 0
    score_p2 = 0

    while True:
        frame_main_raw, frame_side_raw = dual_cam.read()

        if frame_main_raw is None or frame_side_raw is None:
            print("[WARNING] End of stream or camera failure.")
            break

        frame_main = cv2.resize(frame_main_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frame_side = cv2.resize(frame_side_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)

        frame_header = None
        data_header = None
        status_header = "HEADER: NO CAMERA"
        if has_header_cam and header_cap is not None:
            ret_header, frame_header_raw = header_cap.read()
            if not ret_header:
                if isinstance(header_source, str):
                    header_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_header, frame_header_raw = header_cap.read()
            if not ret_header or frame_header_raw is None:
                print("[WARNING] Header camera lost. Disabling header support.")
                has_header_cam = False
                header_cap.release()
                header_cap = None
            else:
                frame_header = cv2.resize(frame_header_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if color_only_mode:
            processor_main.mog_bypass_frames = max(processor_main.mog_bypass_frames, 2)
            processor_side.mog_bypass_frames = max(processor_side.mog_bypass_frames, 2)
            if processor_header is not None:
                processor_header.mog_bypass_frames = max(processor_header.mog_bypass_frames, 2)
        
        # Side camera uses strict (Hough only); main camera relaxes if side sees the ball
        data_side, mask_side, status_side = processor_side.process(frame_side, detection_mode="strict")
        main_mode = "relaxed" if data_side is not None else "strict"
        data_main, mask_main, status_main = processor_main.process(frame_main, detection_mode=main_mode)
        status_main += f" [{main_mode.upper()}]"

        if has_header_cam and frame_header is not None and processor_header is not None:
            data_header, mask_header, status_header = processor_header.process(frame_header, detection_mode="strict")

        current_count, logic_feedback = game_logic.update(main_data=data_main, side_data=data_side, header_data=data_header, frame_width=target_w, frame_height=target_h)

        if logic_feedback and "DROP" in logic_feedback:
            floor_flash_until = time.time() + 0.5

        if logic_feedback and "+1" in logic_feedback:
            hit_popup_text = logic_feedback
            hit_popup_time = time.time()

        draw_tracking_info(frame_side, data_side, f"SIDE: {status_side}", (255, 255, 0)) 
        draw_game_score(frame_side, current_count, score_p1, score_p2, active_player, logic_feedback)
        draw_tracking_info(frame_main, data_main, f"MAIN: {status_main}", (0, 255, 0))
        if has_header_cam and frame_header is not None:
            draw_tracking_info(frame_header, data_header, f"HEADER: {status_header}", (255, 0, 255))
        
        if hit_popup_text:
            draw_hit_popup(frame_side, hit_popup_text, time.time() - hit_popup_time)

        if color_only_mode:
            cv2.putText(frame_side, "COLOR-ONLY (D)", (8, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            cv2.putText(frame_main, "COLOR-ONLY (D)", (8, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if show_floor_overlay and not color_only_mode:
            floor_is_red = time.time() < floor_flash_until
            main_color = (0, 0, 220) if floor_is_red else (0, 180, 0)
            side_color = (0, 0, 220) if floor_is_red else (180, 100, 0)
            main_alpha = 0.45 if floor_is_red else 0.3
            side_alpha = 0.45 if floor_is_red else 0.3
            draw_floor_overlay(frame_main, floor_pts_main, color=main_color, alpha=main_alpha)
            draw_floor_overlay(frame_side, floor_pts_side, color=side_color, alpha=side_alpha)

        cv2.imshow("Main - SIDE View (Master)", frame_side)
        cv2.imshow("Main - FRONT View (Main)", frame_main)
        if has_header_cam and frame_header is not None:
            cv2.imshow("Header View (Ceiling)", frame_header)
        
        key = cv2.waitKey(delay_ms) & 0xFF
        
        if key == 27:
            break
            
        elif key == ord('b') or key == ord('B'):
            print("[INPUT] Resetting MOG2 background models...")
            processor_main.set_instant_background(frame_main)
            processor_side.set_instant_background(frame_side)

        elif key == ord('s') or key == ord('S'):
            processor_main.disable_mog_temporarily(15)
            processor_side.disable_mog_temporarily(15)
            perform_flash_calibration(dual_cam, game_logic)

        elif key == ord('r') or key == ord('R'):
            print("[INPUT] Resetting counter.")
            game_logic.reset()

        elif key == ord('h') or key == ord('H'):
            print("[INPUT] Calibrating header height...")
            if has_header_cam and processor_header is not None and header_cap is not None:
                processor_header.disable_mog_temporarily(15)
                perform_flash_header_calibration(header_cap, game_logic)
            else:
                print("[WARNING] No optional header camera detected.")

        elif key == ord('c') or key == ord('C'):
            game_logic.clear_evaluation_log()

        elif key == ord('f') or key == ord('F'):
            print("[INPUT] Starting floor calibration...")
            new_ff = perform_floor_calibration(dual_cam, processor_main, processor_side, game_logic)
            if new_ff and new_ff.calibrated:
                floor_finder = new_ff
                _, floor_pts_main, floor_pts_side = load_floor_points()

        elif key == ord('d') or key == ord('D'):
            color_only_mode = not color_only_mode
            if not color_only_mode:
                processor_main.mog_bypass_frames = 0
                processor_side.mog_bypass_frames = 0
            state = "ON" if color_only_mode else "OFF"
            print(f"[INPUT] Color-only detection: {state}")

        elif key == ord('g') or key == ord('G'):
            show_floor_overlay = not show_floor_overlay
            state = "ON" if show_floor_overlay else "OFF"
            print(f"[INPUT] Floor overlay: {state}")

        elif key == ord('n') or key == ord('N'):
            if active_player == 1:
                score_p1 = game_logic.count
                active_player = 2
                print("[GAME] Switched to Player 2")
            else:
                score_p2 = game_logic.count
                active_player = 1
                print("[GAME] Switched to Player 1")
                
                # Show winner after both players have played
                if score_p1 > 0 or score_p2 > 0:
                    if score_p1 > score_p2:
                        winner_text = f"P1 WINS!  {score_p1} - {score_p2}"
                    elif score_p2 > score_p1:
                        winner_text = f"P2 WINS!  {score_p2} - {score_p1}"
                    else:
                        winner_text = f"DRAW!  {score_p1} - {score_p2}"
                    print(f"[GAME] {winner_text}")
                    
                    overlay = frame_side.copy()
                    cv2.rectangle(overlay, (0, 0), (target_w, target_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame_side, 0.4, 0, frame_side)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    sz = cv2.getTextSize(winner_text, font, 1.8, 4)[0]
                    tx = (target_w - sz[0]) // 2
                    ty = (target_h + sz[1]) // 2
                    cv2.putText(frame_side, winner_text, (tx, ty), font, 1.8, (0, 255, 255), 4)
                    cv2.imshow("Main - SIDE View (Master)", frame_side)
                    cv2.waitKey(3000)
                    
            game_logic.reset()

        elif key == ord('z') or key == ord('Z'):
            game_logic.adjust_left_anchor(5)
        elif key == ord('a') or key == ord('A'):
            game_logic.adjust_left_anchor(-5)
        
        elif key == 84:
            game_logic.adjust_right_anchor(5)
        elif key == 82:
            game_logic.adjust_right_anchor(-5)
        elif key == ord('k'):
             game_logic.adjust_right_anchor(-5)
        elif key == ord('m'):
             game_logic.adjust_right_anchor(5)
             
        elif key == ord('['): 
            game_logic.adjust_floor_start_x(-10)
        elif key == ord(']'): 
            game_logic.adjust_floor_start_x(10)
            
        elif key == ord("'"): 
            game_logic.adjust_floor_end_x(-10)
        elif key == ord('\\'): 
            game_logic.adjust_floor_end_x(10)
            
        elif key == ord('t') or key == ord('T'):
            game_logic.start_game()  

        elif key == ord('-'):
            game_logic.adjust_floor_epsilon(-0.5)
        elif key == ord('=') or key == ord('+'):
            game_logic.adjust_floor_epsilon(0.5)

    dual_cam.stop()
    if header_cap is not None:
        header_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()