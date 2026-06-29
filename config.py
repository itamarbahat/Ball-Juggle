CONFIG = {
    "camera": {
        "main_source": 0,
        "top_source": 1,     # Camera Side B (secondary) — PREFERRED index.
        "side_source": 2,    # Camera Side A (main)       — PREFERRED index.
                             # Webcam indices drift between runs on this PC, so main.py
                             # verifies this pair at startup and auto-detects a working
                             # pair if it fails. Run  python list_cameras.py  to inspect.
        "header_source": None,
        "width": 640,
        "height": 360
    },

    "processing": {
        "blur_kernel": (11, 11),
        "morph_open_kernel": (3, 3),
        "morph_close_kernel": (11, 11)
    },

    "background": {
        "history_frames": 30,
        "threshold": 25,
        "mog2_history": 250,
        "mog2_threshold": 25,
        "detect_shadows": False
    },

    "roi_top": {
        "top": 0.0, "bottom": 1.0, "left": 0.0, "right": 1.0
    },
    
    "roi_side": {
        "top": 0.0, "bottom": 1.0, "left": 0.0, "right": 1.0
    },

    "detection": {
        "min_area": 100,
        "max_area": 5000,
        "min_circularity": 0.65,
        "min_aspect_ratio": 0.7,
        "max_aspect_ratio": 1.3,
        "hough_param1": 50,
        "hough_param2": 20,
        "relaxed_min_area": 150,     
    },

    "logic": {
        "jitter_threshold": 5,
        "floor_x_start": 0,
        "floor_x_end": 640,
        "min_hit_velocity": -1.0,
        "inertia_frames": 2,
        "floor_left_y": 350,
        "floor_right_y": 350,
        "floor_tolerance": 15,
        "radius_correction_factor": 0.5
    },

    "calibration": {
        "grid_width_cm": 100.0,
        "grid_length_cm": 100.0,
        "grid_width_intervals": 3,
        "grid_length_intervals": 2
    },

    # Presentation / UX layer only — consumed exclusively by the drawing helpers in
    # main.py. Does NOT affect detection, scoring, or any logic values above.
    "ui": {
        # window toggles (default state)
        "show_mask_window": False,   # B/W detection-mask window (key V). Off by default:
                                     # a separate cv2 window steals keyboard focus from
                                     # the pygame dashboard, which blocks all hotkeys.
        "show_pip": True,            # TOP feed inset into the SIDE stage window (key P)

        # event/overlay durations (seconds)
        "toast_sec": 2.0,            # on-screen confirmation messages for control keys
        "winner_sec": 4.0,           # non-blocking winner overlay
        "gameover_sec": 3.0,         # GAME OVER banner hold after a drop
        "hit_popup_sec": 1.4,        # "+1" popup lifetime
        "drop_flash_sec": 0.6,       # red drop alert flash
        "drop_alpha": 0.45,          # drop flash strength

        # floor-calibration UX timings (seconds)
        "result_sec": 2.5,           # on-screen success/failure message
        "capture_flash_sec": 0.4,    # "CAPTURED" confirmation flash
        "warn_sec": 1.2,             # "ball not visible" warning

        # picture-in-picture
        "pip_scale": 0.30,           # inset size as a fraction of the stage frame

        # semantic banner colors (BGR)
        "color_waiting":   (90, 90, 90),
        "color_ready":     (0, 170, 0),
        "color_countdown": (0, 210, 235),
        "color_live":      (0, 200, 0),
        "color_gameover":  (0, 0, 220)
    }
}