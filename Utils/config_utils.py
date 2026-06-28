import json
import os
import numpy as np

# formatting the config file paths to be relative to the script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
# defining the configs directory path
CONFIG_DIR = os.path.join(BASE_DIR, "Configs")

# ensuring the configs directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)

# this defines the paths for the configuration files,
# ensuring they are stored in the configs directory
CONFIG_FILE = os.path.join(CONFIG_DIR, "ball_config.json")
RUNTIME_CONFIG_FILE = os.path.join(CONFIG_DIR, "runtime_config.json")
FLOOR_CONFIG_FILE = os.path.join(CONFIG_DIR, "floor_calibration.json")


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
    
    print(f"[INFO] HSV config for '{profile_name}' saved to {CONFIG_FILE}.")


def load_hsv_config(profile_name="default"):
    """Loads HSV bounds for a camera profile; falls back to broad defaults."""
    default_lower = np.array([0, 30, 50])
    default_upper = np.array([30, 255, 255])

    if not os.path.exists(CONFIG_FILE):
        print(f"[WARNING] Config file not found at {CONFIG_FILE}. Using defaults.")
        return default_lower, default_upper
    
    with open(CONFIG_FILE, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("[ERROR] Failed to decode JSON. Using defaults.")
            return default_lower, default_upper
    
    if profile_name not in data:
        alias_map = {"main": "top", "front": "top"}
        alias_name = alias_map.get(profile_name)
        if alias_name and alias_name in data:
            profile_name = alias_name
        else:
            print(f"[WARNING] Profile '{profile_name}' not found at {CONFIG_FILE}. Using defaults.")
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


def save_floor_points(pts_world, pts_cam_main, pts_cam_side):
    data = {
        "pts_world": pts_world.tolist(),
        "pts_cam_main": pts_cam_main.tolist(),
        "pts_cam_top": pts_cam_main.tolist(),
        "pts_cam_side": pts_cam_side.tolist()
    }
    with open(FLOOR_CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Floor calibration saved to {FLOOR_CONFIG_FILE}.")


def load_floor_points():
    if not os.path.exists(FLOOR_CONFIG_FILE):
        print(f"[WARNING] Floor calibration file not found at {FLOOR_CONFIG_FILE}.")
        return None, None, None

    try:
        with open(FLOOR_CONFIG_FILE, 'r') as f:
            data = json.load(f)
        pts_world = np.array(data["pts_world"], dtype=np.float32)
        if "pts_cam_main" in data:
            pts_cam_main = np.array(data["pts_cam_main"], dtype=np.float32)
        else:
            pts_cam_main = np.array(data["pts_cam_top"], dtype=np.float32)
        pts_cam_side = np.array(data["pts_cam_side"], dtype=np.float32)
        print(f"[INFO] Floor calibration loaded ({len(pts_world)} points).")
        return pts_world, pts_cam_main, pts_cam_side
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Floor calibration load failed: {e}")
        return None, None, None


def generate_world_points(config):
    """
    Dynamically generates a 2D grid of world calibration points based on configuration.
    
    Args:
        config: CONFIG dictionary containing 'calibration' section with:
            - grid_width_cm: total width of the grid in centimeters
            - grid_length_cm: total length (depth) of the grid in centimeters
            - grid_width_intervals: number of intervals along width (creates n+1 points)
            - grid_length_intervals: number of intervals along length (creates n+1 points)
    
    Returns:
        numpy array of shape (num_points, 2) with world coordinates [x, y] in cm.
        For example, with 3 width intervals and 2 length intervals: 4x3 grid = 12 points.
    """
    cal_config = config.get("calibration", {})
    
    grid_width = float(cal_config.get("grid_width_cm", 100.0))
    grid_length = float(cal_config.get("grid_length_cm", 100.0))
    #The next tow lines determine how many intervals to divide the grid into, which affects the number of points generated.
    width_intervals = int(cal_config.get("grid_width_intervals", 3))
    length_intervals = int(cal_config.get("grid_length_intervals", 2))
    
    # Generate evenly spaced points along each dimension
    x_coords = np.linspace(0, grid_width, width_intervals + 1)
    y_coords = np.linspace(0, grid_length, length_intervals + 1)
    
    # Create grid points by iterating through y first (rows), then x (columns)
    world_points = []
    for y in y_coords:
        for x in x_coords:
            world_points.append([x, y])
    
    return np.array(world_points, dtype=np.float32)


def generate_point_names(config):
    """
    Dynamically generates descriptive names for all calibration points based on grid configuration.
    
    Args:
        config: CONFIG dictionary containing 'calibration' section with interval counts.
    
    Returns:
        List of strings describing each point's position in the grid.
    """
    cal_config = config.get("calibration", {})
    width_intervals = int(cal_config.get("grid_width_intervals", 3))
    length_intervals = int(cal_config.get("grid_length_intervals", 2))
    
    # Dimension descriptors
    length_names = ["FRONT", "MIDDLE", "BACK"]
    width_names = ["LEFT", "CENTER-LEFT", "CENTER-RIGHT", "RIGHT"]
    
    # Adjust names if grid dimensions differ
    if length_intervals + 1 != len(length_names):
        length_names = [f"ROW-{i}" for i in range(length_intervals + 1)]
    if width_intervals + 1 != len(width_names):
        width_names = [f"COL-{i}" for i in range(width_intervals + 1)]
    
    point_names = []
    for length_desc in length_names:
        for width_desc in width_names:
            point_names.append(f"{length_desc}-{width_desc}")
    
    return point_names
