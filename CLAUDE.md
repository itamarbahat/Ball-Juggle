# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A **real-time juggling counter**: classical computer vision (no deep learning at
runtime, CPU-only) that detects, tracks, and counts soccer-ball juggles from a
**dual-camera** feed, with live feedback, drop detection, and two-player scoring.
See `README.md` for the full write-up.

Two cameras:
- **Side (Master)** — primary tracking & counting; runs **strict** detection (Hough only).
- **Top (Slave)** — floor-agreement + optional headers; runs **relaxed** detection
  (Hough + contour fallback), enabled only when the side camera already sees the ball.

Core detection pipeline (per camera, in `ball_processor.py`):
`Gaussian blur → MOG2 motion mask → HSV color mask → bitwise-AND fusion →
morphology → Hough circles (contour fallback) → Kalman filter` → `(x, y, radius)`.

Counting (in `juggling_logic.py`): a kick = a **V-flip** (ball was FALLING, then
rises 2+ frames). Drops use a multi-layer check (homography floor agreement +
boundary + velocity-freeze + tracking-loss timeout + retroactive kinematic sanity).

## Calibration model (important — most user-facing pain lives here)

Calibration runs from inside `main.py` via single keystrokes; HSV color is a
separate helper. During calibration, MOG2 is bypassed so detection uses **pure HSV**
(lets a stationary ball be detected). Order matters:

0. **HSV color** — `Calibration Helper script.py` (standalone GUI, 6 sliders) →
   saves per-camera bounds to `ball_config.json`.
1. **Background** (`B`) — rebuild MOG2 model from the empty scene.
2. **Radius** (`S`) — 15-frame median ball radius → radius gate at ±50%.
3. **Floor** (`F`) — **12-point homography**: place ball at a 4×3 world grid
   (X ∈ {0,33,67,100} cm, Y ∈ {0,50,100} cm), SPACE at each, detected in **both**
   cameras → two RANSAC homographies → `floor_calibration.json`. A drop is confirmed
   when both cameras' world-plane projections agree within `floor_epsilon_cm`.
4. **Floor fine-tune** — live anchor/boundary/epsilon nudges (keys A/Z, arrows, [ ], ' \, - +).
5. **Header** (`H`, optional) — top-camera radius baseline for head juggles.

## Files that matter for running the project

| File | Role |
|------|------|
| `main.py` | Main loop, UI overlays, keyboard input, calibration orchestration |
| `ball_processor.py` | Per-camera detection pipeline |
| `juggling_logic.py` | Kick counting (V-flip), drop logic, scoring |
| `floor_finding.py` | Homography ground-plane agreement |
| `camera_manager.py` | Threaded dual-camera capture |
| `config.py` | Static config (kernels, thresholds, **camera sources**, logic params) |
| `config_utils.py` | Load/save HSV, floor points, runtime config |
| `Calibration Helper script.py` | Standalone HSV calibration GUI |
| `ball_config.json`, `floor_calibration.json`, `runtime_config.json` | Persisted calibration |

Evaluation tooling (not needed to run the game): `capture_eval_frames.py`,
`generate_ground_truth.py`, `run_pipeline.py`, `evaluate_segmentation.py`, `evaluation/`.

## How to run

```bash
pip install opencv-python numpy pygame   # or: pip install .
python main.py
```
Set camera sources in `config.py → CONFIG["camera"]` (`top_source`/`side_source`:
webcam index, or a string video path for looping test playback). There is **no
test suite** — verification is done by running the app live and observing behavior.

## Focus for upcoming sessions

The project is being prepared for presentation at a **prestigious exhibition**.
Priorities, in order:

1. **Testing / robustness** — exercise the system under varied lighting, backgrounds,
   and camera setups; harden against false hits/drops and tracking loss. Prefer
   real runs (`python main.py`) to confirm behavior.
2. **User interface** — improve the on-screen UX: scores, hit/drop feedback,
   countdown, winner screen, and overall polish so it reads well to an audience.
3. **Accessible calibration** — the biggest friction point. Make the
   calibration flow (HSV, radius, 12-point floor, header) simpler, more guided, and
   harder to get wrong for a non-expert operator at a live booth.

## Working conventions

- Match the existing code style: small single-responsibility functions, OpenCV
  drawing helpers in `main.py`, config-driven thresholds (avoid hardcoding — add to
  `config.py` or the JSON configs).
- Keep it **CPU-only and real-time**; don't introduce GPU/deep-learning runtime deps.
- When changing detection/logic thresholds, remember they're tuned empirically —
  explain the trade-off and verify live.
- Calibration writes to JSON via `config_utils.py`; reuse those helpers rather than
  reading/writing files ad hoc.
