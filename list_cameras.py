"""
list_cameras.py — quick camera diagnostic.

Webcam indices on Windows are not always stable between runs (USB re-enumeration,
virtual-camera drivers, etc.). Run this to see which indices currently deliver
frames and which PAIRS work simultaneously, then set the working pair in
config.py -> CONFIG["camera"]["top_source"] / ["side_source"].

    python list_cameras.py
"""

import sys
import time
import itertools
import cv2

W, H = 640, 360
MAX_INDEX = 5          # scan indices 0..MAX_INDEX-1
READS = 8              # frames to sample per camera


def _open(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    return cap


def _sustained(cap, n=READS):
    ok = 0
    for _ in range(n):
        r, f = cap.read()
        if r and f is not None:
            ok += 1
        time.sleep(0.03)
    return ok


def main():
    print("Scanning camera indices (this can take a few seconds per index)...\n")

    working = []
    for idx in range(MAX_INDEX):
        cap = _open(idx)
        got = _sustained(cap) if cap.isOpened() else 0
        cap.release()
        mark = "OK " if got > 0 else "-- "
        print(f"  [{mark}] index {idx}: {got}/{READS} frames")
        if got > 0:
            working.append(idx)
        time.sleep(0.3)

    print(f"\nIndices that deliver frames individually: {working or 'NONE'}")

    good_pairs = []
    for a, b in itertools.combinations(working, 2):
        ca, cb = _open(a), _open(b)
        ga, gb = _sustained(ca), _sustained(cb)
        ca.release(); cb.release()
        ok = ga > 0 and gb > 0
        print(f"  pair ({a},{b}): {ga}/{READS} + {gb}/{READS} -> {'WORKS together' if ok else 'conflict'}")
        if ok:
            good_pairs.append((a, b))
        time.sleep(0.3)

    print("\n" + "=" * 56)
    if good_pairs:
        a, b = good_pairs[0]
        print(f"Two cameras work together: indices {a} and {b}.")
        print("Set in config.py:")
        print(f'    "top_source": {a},')
        print(f'    "side_source": {b},')
    elif working:
        only = working[0]
        print(f"Only one usable camera (index {only}).")
        print("For two live panels on this machine, set BOTH to the same index:")
        print(f'    "top_source": {only},')
        print(f'    "side_source": {only},')
    else:
        print("No working camera found. Check connections / privacy settings.")
    print("=" * 56)


if __name__ == "__main__":
    main()
