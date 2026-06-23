CONFIG = {
    "camera": {
        "top_source": 0,
        "side_source": 0,
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
    }
}