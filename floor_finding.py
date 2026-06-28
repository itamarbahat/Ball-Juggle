import cv2
import numpy as np
from Utils.config_utils import load_floor_points


class FloorFinder:
    """
    Dual-camera homography floor detector. Projects both cameras' ball
    positions onto a shared world plane; if they agree the ball is on the floor.
    """

    def __init__(self, pts_world=None, pts_cam1=None, pts_cam2=None):
        if pts_world is None or pts_cam1 is None or pts_cam2 is None:
            saved_world, saved_top, saved_side = load_floor_points()
            if saved_world is not None:
                pts_world = saved_world
                pts_cam1 = saved_top
                pts_cam2 = saved_side
            else:
                print("[FLOOR] No saved calibration found. Using hardcoded defaults.")
                pts_world = DEFAULT_PTS_WORLD
                pts_cam1 = DEFAULT_PTS_CAM1
                pts_cam2 = DEFAULT_PTS_CAM2

        if len(pts_world) < 4:
            print("[FLOOR] Need at least 4 calibration points. Floor detection disabled.")
            self.H1 = None
            self.H2 = None
            self.calibrated = False
            return

        self.H1, _ = cv2.findHomography(pts_cam1, pts_world, cv2.RANSAC)
        self.H2, _ = cv2.findHomography(pts_cam2, pts_world, cv2.RANSAC)
        self.calibrated = self.H1 is not None and self.H2 is not None

        if self.calibrated:
            print("[FLOOR] Homography matrices calculated successfully.")
        else:
            print("[FLOOR] Calibration failed. Floor detection disabled.")

    def is_ball_on_floor(self, pixel_cam1, pixel_cam2, epsilon=5.0):
        """Returns (on_ground, distance, world_p1, world_p2)."""
        if not self.calibrated:
            return False, float('inf'), None, None

        p1 = np.array([[[pixel_cam1[0], pixel_cam1[1]]]], dtype=np.float32)
        p2 = np.array([[[pixel_cam2[0], pixel_cam2[1]]]], dtype=np.float32)

        world_p1 = cv2.perspectiveTransform(p1, self.H1)[0][0]
        world_p2 = cv2.perspectiveTransform(p2, self.H2)[0][0]

        distance = np.linalg.norm(world_p1 - world_p2)
        on_ground = distance <= epsilon

        return on_ground, distance, world_p1, world_p2


# Default calibration data (fallback when no saved calibration exists)
DEFAULT_PTS_WORLD = np.array([
    # Row 1 — Front (y=0): 4 across
    [0, 0],        # Point 1:  Front-Left
    [83, 0],       # Point 2:  Front-Center-Left
    [166, 0],       # Point 3:  Front-Center-Right
    [249, 0],      # Point 4:  Front-Right
    # Row 2 — Middle (y=96): 4 across
    [0, 92],       # Point 5:  Middle-Left
    [83, 92],      # Point 6:  Middle-Center-Left
    [166, 92],      # Point 7:  Middle-Center-Right
    [249, 92],     # Point 8:  Middle-Right
    # Row 3 — Back (y=192): 4 across
    [0, 184],      # Point 9:  Back-Left
    [83, 184],     # Point 10: Back-Center-Left
    [166, 184],     # Point 11: Back-Center-Right
    [249, 184]     # Point 12: Back-Right
], dtype=np.float32)

DEFAULT_PTS_CAM1 = np.array([
    [100, 400],    # Point 1
    [170, 400],    # Point 2
    [230, 400],    # Point 3
    [300, 400],    # Point 4
    [120, 320],    # Point 5
    [175, 320],    # Point 6
    [225, 320],    # Point 7
    [280, 320],    # Point 8
    [150, 250],    # Point 9
    [185, 250],    # Point 10
    [215, 250],    # Point 11
    [250, 250]     # Point 12
], dtype=np.float32)

DEFAULT_PTS_CAM2 = np.array([
    [500, 400],    # Point 1
    [530, 370],    # Point 2
    [560, 340],    # Point 3
    [600, 300],    # Point 4
    [440, 350],    # Point 5
    [475, 320],    # Point 6
    [510, 290],    # Point 7
    [560, 250],    # Point 8
    [400, 300],    # Point 9
    [430, 275],    # Point 10
    [460, 250],    # Point 11
    [500, 200]     # Point 12
], dtype=np.float32)

