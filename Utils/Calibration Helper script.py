import cv2
import numpy as np
import time
import sys
import os

# we add the root directory to the path so we can import from utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# עכשיו אנחנו מייבאים מתוך utils בהנחה ששני הקבצים שם
from Utils.config_utils import load_hsv_config, save_hsv_config 

def nothing(x):
    pass

def calibrate_ball_color(source=1, camera_name="default"):
    """
    Interactive HSV calibration GUI. Shows live feed + binary mask side by side.
    Adjust sliders until the ball is white and background is black, then press ESC to save.
    """
    print(f"[INFO] Opening source: {source} for profile: '{camera_name}'...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    window_name = f"HSV Calibration - {camera_name.upper()}"
    cv2.namedWindow(window_name)

    lower_saved, upper_saved = load_hsv_config(profile_name=camera_name)
    
    cv2.createTrackbar("Lower Hue", window_name, int(lower_saved[0]), 179, nothing)
    cv2.createTrackbar("Lower Sat", window_name, int(lower_saved[1]), 255, nothing) 
    cv2.createTrackbar("Lower Val", window_name, int(lower_saved[2]), 255, nothing)
    cv2.createTrackbar("Upper Hue", window_name, int(upper_saved[0]), 179, nothing)
    cv2.createTrackbar("Upper Sat", window_name, int(upper_saved[1]), 255, nothing)
    cv2.createTrackbar("Upper Val", window_name, int(upper_saved[2]), 255, nothing)

    print("[INFO] Calibration Started.")
    print("[INFO] Adjust sliders until the ball is pure WHITE and background is BLACK.")
    print("[INFO] Press 'ESC' to SAVE settings and EXIT.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("Lower Hue", window_name)
        l_s = cv2.getTrackbarPos("Lower Sat", window_name)
        l_v = cv2.getTrackbarPos("Lower Val", window_name)
        u_h = cv2.getTrackbarPos("Upper Hue", window_name)
        u_s = cv2.getTrackbarPos("Upper Sat", window_name)
        u_v = cv2.getTrackbarPos("Upper Val", window_name)

        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack([frame, mask_bgr])
        cv2.imshow(window_name, combined_view)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            print("\n" + "="*40)
            print(f"[INFO] Calibration finished for '{camera_name}'.")
            print(f"[RESULT] Lower HSV: {lower_bound}")
            print(f"[RESULT] Upper HSV: {upper_bound}")
            save_hsv_config(lower_bound, upper_bound, profile_name=camera_name)
            print("="*40 + "\n")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_ball_color(source=0, camera_name="top")
    calibrate_ball_color(source=0, camera_name="side")