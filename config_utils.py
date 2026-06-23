import json
import os
import numpy as np

CONFIG_FILE = "ball_config.json"
RUNTIME_CONFIG_FILE = "runtime_config.json"

def save_hsv_config(lower_hsv, upper_hsv, profile_name="default"):
    """Saves HSV bounds for a camera profile, preserving other profiles in the file."""
    data = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("[WARNING] Config file was corrupted. Creating a new one.")
            data = {}

    data[profile_name] = {
        "lower_hsv": lower_hsv.tolist(),
        "upper_hsv": upper_hsv.tolist()
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"[INFO] HSV config for '{profile_name}' saved.")

def load_hsv_config(profile_name="default"):
    """Loads HSV bounds for a camera profile; falls back to broad defaults."""
    default_lower = np.array([0, 30, 50])
    default_upper = np.array([30, 255, 255])

    if not os.path.exists(CONFIG_FILE):
        print(f"[WARNING] Config file not found. Using defaults.")
        return default_lower, default_upper
    
    with open(CONFIG_FILE, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("[ERROR] Failed to decode JSON. Using defaults.")
            return default_lower, default_upper
    
    if profile_name not in data:
        print(f"[WARNING] Profile '{profile_name}' not found. Using defaults.")
        return default_lower, default_upper
    
    profile_data = data[profile_name]
    lower = np.array(profile_data["lower_hsv"])
    upper = np.array(profile_data["upper_hsv"])
    
    print(f"[INFO] HSV config for '{profile_name}' loaded.")
    return lower, upper


def save_floor_epsilon(epsilon_cm):
    """Persists the floor-agreement epsilon to runtime_config.json."""
    data = {}
    if os.path.exists(RUNTIME_CONFIG_FILE):
        try:
            with open(RUNTIME_CONFIG_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("[WARNING] Runtime config was corrupted. Creating a new one.")
            data = {}

    data["floor_epsilon_cm"] = float(round(epsilon_cm, 2))

    with open(RUNTIME_CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[INFO] Floor epsilon saved: {data['floor_epsilon_cm']:.2f} cm")


def load_floor_epsilon(default_value=5.0):
    if not os.path.exists(RUNTIME_CONFIG_FILE):
        return float(default_value)

    try:
        with open(RUNTIME_CONFIG_FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("[WARNING] Runtime config corrupted. Using default floor epsilon.")
        return float(default_value)

    epsilon = data.get("floor_epsilon_cm", default_value)
    try:
        return float(epsilon)
    except (TypeError, ValueError):
        print("[WARNING] Invalid floor epsilon value in runtime config. Using default.")
        return float(default_value)


FLOOR_CONFIG_FILE = "floor_calibration.json"

def save_floor_points(pts_world, pts_cam_top, pts_cam_side):
    data = {
        "pts_world": pts_world.tolist(),
        "pts_cam_top": pts_cam_top.tolist(),
        "pts_cam_side": pts_cam_side.tolist()
    }
    with open(FLOOR_CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Floor calibration saved.")


def load_floor_points():
    if not os.path.exists(FLOOR_CONFIG_FILE):
        print(f"[WARNING] Floor calibration file not found.")
        return None, None, None

    try:
        with open(FLOOR_CONFIG_FILE, 'r') as f:
            data = json.load(f)
        pts_world = np.array(data["pts_world"], dtype=np.float32)
        pts_cam_top = np.array(data["pts_cam_top"], dtype=np.float32)
        pts_cam_side = np.array(data["pts_cam_side"], dtype=np.float32)
        print(f"[INFO] Floor calibration loaded ({len(pts_world)} points).")
        return pts_world, pts_cam_top, pts_cam_side
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Floor calibration load failed: {e}")
        return None, None, None