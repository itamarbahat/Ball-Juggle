import cv2
import numpy as np
from config import CONFIG
from Utils.config_utils import load_hsv_config

class BallProcessor:
    """
    Per-camera ball detection pipeline: MOG2 motion mask + HSV color mask,
    fused into a hybrid mask, then detected via Hough circles (strict) or
    contour analysis (relaxed). Kalman filter smooths tracking across frames.
    """

    def __init__(self, camera_profile="main"):
        self.profile = camera_profile
        self.lower_hsv, self.upper_hsv = load_hsv_config(camera_profile)
        print(f"[INFO] Processor '{self.profile}' initialized.")
        
        self.target_w = CONFIG["camera"]["width"]
        self.target_h = CONFIG["camera"]["height"]

        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=CONFIG["background"].get("mog2_history", 500),
            varThreshold=CONFIG["background"].get("mog2_threshold", 25),
            detectShadows=False
        )

        self.min_hough_r = int(np.sqrt(CONFIG["detection"]["min_area"] / np.pi))
        self.max_hough_r = int(np.sqrt(CONFIG["detection"]["max_area"] / np.pi))

        self.kernel_open = np.ones(CONFIG["processing"]["morph_open_kernel"], np.uint8)
        self.kernel_close = np.ones(CONFIG["processing"]["morph_close_kernel"], np.uint8)

        self.mog_bypass_frames = 0

        self.kf = cv2.KalmanFilter(4, 2)  # state: [x, y, dx, dy], measurement: [x, y]
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        self.kalman_active = False
        self.last_known_r = 10

    def update_hough_params(self, calibrated_radius):
        """Narrows Hough radius range to +/-50% of the calibrated ball radius."""
        self.min_hough_r = int(calibrated_radius * 0.5)
        self.max_hough_r = int(calibrated_radius * 1.5)
        print(f"[INFO] Hough params updated for '{self.profile}': R=[{self.min_hough_r}, {self.max_hough_r}]")

    def crop_to_roi(self, frame):
        h, w = frame.shape[:2]
        roi_key = f"roi_{self.profile}"
        roi = CONFIG.get(roi_key, CONFIG.get("roi_top")) 
        
        y1 = int(h * roi["top"])
        y2 = int(h * roi["bottom"])
        x1 = int(w * roi["left"])
        x2 = int(w * roi["right"])
        return frame[y1:y2, x1:x2]

    def set_instant_background(self, frame):
        """Resets the MOG2 model so it relearns the background."""
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=CONFIG["background"].get("mog2_history", 500),
            varThreshold=CONFIG["background"].get("mog2_threshold", 25),
            detectShadows=False
        )
        print(f"[INFO] MOG2 background reset for '{self.profile}'.")
    
    def disable_mog_temporarily(self, frames=15):
        """Bypasses MOG2 for the given number of frames (color-only detection)."""
        self.mog_bypass_frames = frames
        print(f"[INFO] MOG2 bypassed for {frames} frames on '{self.profile}'.")
        

    def process(self, frame, detection_mode="strict"):
        """
        Runs the full detection pipeline on one frame.
        Returns (ball_tuple, mask, status_string).
        detection_mode: 'strict' = Hough only, 'relaxed' = fallback to contours.
        """
        if frame is None: return None, None, "No Frame"

        frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        frame_roi = self.crop_to_roi(frame)
        blurred = cv2.GaussianBlur(frame_roi, CONFIG["processing"]["blur_kernel"], 0)

        # Motion mask (MOG2) + color mask (HSV) → hybrid
        fgmask = self.fgbg.apply(blurred)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        if self.mog_bypass_frames > 0:
            final_mask = color_mask
            self.mog_bypass_frames -= 1
        else:
            final_mask = cv2.bitwise_and(color_mask, color_mask, mask=fgmask)

        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

        best_ball = None
        status = "Searching..."
        
        # Strict: Hough circle detection
        mask_blurred_for_hough = cv2.GaussianBlur(final_mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            mask_blurred_for_hough, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2,
            minDist=100,
            param1=CONFIG["detection"].get("hough_param1", 50),
            param2=CONFIG["detection"].get("hough_param2", 20),
            minRadius=self.min_hough_r,
            maxRadius=self.max_hough_r
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            cx, cy, r = circles[0]
            best_ball = self._remap_to_global(cx, cy, r, frame)
            status = "Tracking (Hough)"
        
        # Relaxed: contour/blob fallback (only when side cam confirmed ball is in play)
        elif detection_mode == "relaxed":
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > CONFIG["detection"].get("relaxed_min_area", 100):
                    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                    best_ball = self._remap_to_global(int(x), int(y), int(radius), frame)
                    status = "Tracking (Blob)"

        # Kalman filter: smooth tracked position or predict during occlusion
        pred_x, pred_y = 0, 0
        if self.kalman_active:
            prediction = self.kf.predict()
            pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])

        if best_ball is not None:
            x, y, r = best_ball
            measurement = np.array([[x], [y]], dtype=np.float32)
            
            if not self.kalman_active:
                self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
                self.kalman_active = True
            else:
                self.kf.correct(measurement)
            
            self.last_known_r = r
            
        else:
            if self.kalman_active:
                best_ball = (pred_x, pred_y, self.last_known_r)
                status = "Tracking (Kalman Prediction)"
                
                if pred_x < 0 or pred_x > self.target_w or pred_y < 0 or pred_y > self.target_h:
                    self.kalman_active = False
                    best_ball = None
                    status = "Lost Ball (Kalman Reset)"

        return best_ball, final_mask, status


    def _remap_to_global(self, cx, cy, r, frame):
        roi_key = f"roi_{self.profile}"
        roi = CONFIG.get(roi_key, CONFIG.get("roi_top"))
        
        global_y = cy + int(frame.shape[0] * roi["top"])
        global_x = cx + int(frame.shape[1] * roi["left"])
        return (int(global_x), int(global_y), int(r))