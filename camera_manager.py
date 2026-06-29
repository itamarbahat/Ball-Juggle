import cv2
import sys
from threading import Thread
import time
from config import CONFIG

class CameraManager:
    """
    Threaded camera reader. A background thread continuously grabs the latest
    frame so the main loop never blocks on I/O.
    """

    def __init__(self, src=0, name="Camera"):
        self.src = src
        self.name = name
        # On Windows the default MSMF backend opens very slowly (~5 s per camera) and
        # can appear to hang on the first .set()/read(); DirectShow opens in ~1 s and
        # allows the same device index to be opened by both cameras. Webcam indices
        # only — string sources (video files) keep the default backend.
        if isinstance(self.src, int) and sys.platform == "win32":
            self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        else:
            self.stream = cv2.VideoCapture(self.src)

        target_w = CONFIG["camera"]["width"]
        target_h = CONFIG["camera"]["height"]
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        
        if not self.stream.isOpened():
            print(f"[ERROR] Cannot open camera: {self.src}")
            self.stopped = True
            self.grabbed = False
            self.frame = None
            return

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        if self.stopped: 
            return self
        print(f"[INFO] Camera thread '{self.name}' started.")
        t = Thread(target=self._update, name=self.name, daemon=True)
        t.start()
        return self

    def _update(self):
        while True:
            if self.stopped:
                return
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                if isinstance(self.src, str): 
                    print(f"[INFO] Video '{self.name}' ended — looping.")
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print(f"[WARNING] '{self.name}' failed to grab frame. Stopping.")
                    self.stopped = True
                    return
            self.grabbed = grabbed
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.stream.release()
        print(f"[INFO] Camera '{self.name}' released.")


class DualCameraManager:
    """Manages main and side CameraManager instances together."""

    def __init__(self):
        main_src = CONFIG["camera"].get("main_source", CONFIG["camera"].get("top_source"))
        side_src = CONFIG["camera"]["side_source"]
        self.cam_main = CameraManager(src=main_src, name="MainCam")
        self.cam_side = CameraManager(src=side_src, name="SideCam")

    def start(self):
        self.cam_main.start()
        self.cam_side.start()
        return self

    def read(self):
        return self.cam_main.read(), self.cam_side.read()

    def stop(self):
        self.cam_main.stop()
        self.cam_side.stop()