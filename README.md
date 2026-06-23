# Real-Time Juggling Counter

A real-time computer-vision system that automatically detects, tracks, and counts
soccer-ball juggles ("keepie-uppies") from a live **dual-camera** feed. The system
gives immediate on-screen feedback, detects drops to the floor, and supports
two-player gameplay — all built entirely on **classical image-processing** techniques
(no deep learning at runtime, no GPU required).

> Course: *Introduction to Image Processing* — Final Project
> Team: Bar Megidish · Eylon Oren · Itamar Bahat · Yosef Shalev

---

## Table of Contents
- [Overview](#overview)
- [How It Works (Pipeline)](#how-it-works-pipeline)
- [Calibration](#calibration)
  - [0. HSV Color Calibration](#0-hsv-color-calibration-ball)
  - [1. Background (MOG2)](#1-background-mog2--key-b)
  - [2. Ball Radius](#2-ball-radius--key-s)
  - [3. Floor (12-point homography)](#3-floor-calibration--12-point-homography--key-f)
  - [4. Floor-line fine-tuning](#4-floor-line-fine-tuning)
  - [5. Header height (optional)](#5-header-height-optional--key-h)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Keyboard Controls](#keyboard-controls)
- [Configuration Files](#configuration-files)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Assumptions](#assumptions)

---

## Overview

The goal is to count juggles automatically and reliably under realistic indoor
conditions — changing lighting, cluttered backgrounds, and frequent occlusions
(the ball disappearing behind a leg). Rather than imitating human perception, the
system reframes each challenge into something a 2D camera and basic math can
*measure*:

- A **kick** is not "seen" — it is detected as a **velocity reversal** (a V-flip)
  in the ball's vertical motion.
- The **floor** is not "recognized" — two cameras project the ball onto a shared
  world plane via **homography**, and a touch is confirmed when both projections
  agree.

Two cameras are used:

| Camera | Role | View |
|--------|------|------|
| **Side** (Master) | Primary tracking & counting | Side-angled, facing the floor area |
| **Top** (Slave)   | Floor agreement & optional headers | Front-facing, straight at the player |

The side camera runs **strict** detection (Hough only). The top camera runs
**relaxed** detection (Hough with a contour fallback) but only when the side camera
already sees the ball — this suppresses false positives while keeping recall high.

---

## How It Works (Pipeline)

Each frame from both cameras passes through the same stages before a **HIT** or
**DROP** event is produced.

### 1. Ball Detection → `(x, y, radius)` per camera
1. **Gaussian Blur** (11×11) — suppress high-frequency sensor noise.
2. **MOG2 Background Subtraction** — adaptive Gaussian Mixture Model isolates
   moving pixels; adapts to gradual lighting changes without manual recalibration.
3. **HSV Color Thresholding** — convert BGR→HSV and threshold with calibrated
   bounds; isolating Hue makes detection robust to shadows and lighting.
4. **Bitwise-AND Fusion** — keep only pixels that are *both* moving (MOG2) *and*
   the right color (HSV). Eliminates both the "static same-color object" and the
   "moving body part" failure modes.
5. **Morphology** — Opening (3×3) removes speckle noise; Closing (11×11) fills
   holes inside the ball silhouette.
6. **Hough Circle Transform** — accept only truly round shapes within the
   calibrated radius range (rejects shoes, hands, clutter). `param2` is tuned via
   config for the sensitivity/robustness trade-off.
7. **Contour Analysis** (fallback) — if Hough fails on the secondary camera, take
   the largest contour and fit a minimum enclosing circle.
8. **Kalman Filter** — a 4D `[x, y, dx, dy]` constant-velocity filter per camera
   smooths the track and bridges short occlusions (up to ~10 frames) by predicting
   from last known velocity.

### 2. Movement Analysis & Impact Detection
- **Radius Gating** — reject detections deviating >±50% from the calibrated
  baseline radius (a fast sanity check against non-ball objects).
- **Finite-Difference Velocity** — `ΔY = Y_current − Y_previous` (positive =
  falling, negative = rising).
- **Inertia State Machine** — requires 2+ consecutive consistent frames before
  locking a `FALLING`/`RISING` state (a temporal low-pass filter against jitter).
- **V-Flip (Velocity Reversal) Detection** — a HIT is registered when the ball was
  `FALLING` and then rises for 2+ frames. This is a direct physical model of a
  kick (ball falls → contacts foot → rises).
- **First-Kick Protection** — the opening flick-up uses stricter thresholds
  (3+ rising frames, ≥15 px displacement).

### 3. Floor Detection & Drop Logic (multi-layered, redundant)
- **Dual-Camera Homography** *(primary)* — each camera's pixels map to real-world
  floor coordinates via a homography matrix (RANSAC). Both cameras project the
  ball center onto the shared ground plane; if the Euclidean distance between the
  two projections is below `floor_epsilon_cm`, the ball is confirmed **on the floor → DROP**.
- **Boundary Check** — ball X outside the playing area → DROP.
- **Velocity Freeze** — near-zero velocity for ~1 second (≈30 frames) → DROP.
- **Tracking-Loss Timeout** — both cameras lose the ball for ~1 s → game ends;
  extended to ~3 s if the ball was last seen high with upward velocity (high-kick
  immunity).
- **Retroactive Kinematic Analysis** *(safety wrapper)* — examines intervals
  between recent hits; physically impossible rapid patterns (e.g. floor-bounce
  "ghost hits", intervals < 0.35 s, or shrinking intervals from energy decay) are
  identified and subtracted from the score.

---

## Calibration

Calibration is the heart of the system. Run the steps below **once** at the start
of each session (or whenever lighting / camera position changes). Most steps are
triggered by a single keystroke in the running app; HSV color calibration is a
separate helper script.

> **Note:** During calibration, MOG2 motion detection is temporarily bypassed so
> detection runs on **pure HSV color** — this lets a stationary ball be detected
> while it sits still on the floor or head.

### 0. HSV Color Calibration (ball)
Run **once** with the standalone helper `Calibration Helper script.py`. It opens a
GUI with six sliders (Lower/Upper × Hue/Sat/Val) and a live feed beside its binary
mask. Adjust until the ball is pure **white** and the background is fully **black**,
then press **ESC** to save. Results are written to `ball_config.json` under a
separate profile per camera (`side`, `top`).

### 1. Background (MOG2) — key **`B`**
Step out of frame and press **`B`** to reset and rebuild the MOG2 background model
from the current scene.

### 2. Ball Radius — key **`S`**
Place the ball on the floor and press **`S`**. The system captures 15 pure-HSV
frames, takes the **median** detected radius as the baseline, and configures the
radius gate to ±50% around it. (Median, not mean — robust to outlier frames.)

### 3. Floor Calibration — 12-point homography — key **`F`**
This is the most critical calibration step. Press **`F`** and place the ball at
**12 marked positions** forming a 4×3 grid on the real floor. Press **SPACE** at
each position (or **ESC** to cancel). The world coordinates are:

```
X ∈ {0, 33, 67, 100} cm   (left → right)
Y ∈ {0, 50, 100} cm       (front → back)
```

For each point the ball must be detected in **both** cameras simultaneously. After
all 12 points, two homography matrices (one per camera) are computed via RANSAC and
saved to `floor_calibration.json`. From then on, a drop is confirmed when both
cameras' world-plane projections of the ball agree within `floor_epsilon_cm`.

### 4. Floor-line fine-tuning
After automatic calibration the floor line can be nudged live:
- **`A` / `Z`** — raise / lower the **left** anchor
- **`UP` / `DOWN`** (or **`K` / `M`**) — raise / lower the **right** anchor
- **`[` / `]`** — move the **left** boundary
- **`'` / `\`** — move the **right** boundary
- **`-` / `+`** — decrease / increase floor epsilon by 0.5 cm (saved to `runtime_config.json`)

### 5. Header height (optional) — key **`H`**
Hold the ball on the player's head and press **`H`** to capture a separate radius
baseline for the **top** camera, enabling head-juggle ("header") detection via
relative size change.

---

## Installation

Requires **Python 3.9+** and two connected cameras (USB webcams or video files).

```bash
# Using the standard toolchain
pip install opencv-python numpy pygame

# or with the included pyproject.toml
pip install .
```

Dependencies: `opencv-python`, `numpy`, `pygame` (for the drop whistle sound).

---

## Running the System

1. **Configure camera sources** in `config.py` → `CONFIG["camera"]`:
   - `top_source` and `side_source` — webcam indices (e.g. `0`, `1`) or paths to
     video files (a string path enables looping playback for testing).
   - `width` / `height` — processing resolution (default 640×360).

2. **Run the HSV helper once** to set the ball color (see [step 0](#0-hsv-color-calibration-ball)).

3. **Launch the main app:**
   ```bash
   python main.py
   ```

4. **Calibrate** in order: `B` → `S` → `F` → (fine-tune) → optionally `H`.

5. Press **`T`** to start a 3-second countdown and begin a counted game.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `B` | Re-capture background (reset MOG2) |
| `S` | Calibrate ball radius (15-frame median) |
| `F` | Floor calibration (12-point homography) |
| `H` | Calibrate header height (top camera) |
| `T` | Start 3-second countdown & play |
| `N` | Switch player (saves score; shows winner after both play) |
| `R` | Reset score / counter |
| `D` | Toggle color-only detection (MOG2 bypass) |
| `G` | Toggle floor-grid overlay |
| `C` | Clear evaluation log |
| `A`/`Z`, `UP`/`DOWN`, `K`/`M`, `[`/`]`, `'`/`\`, `-`/`+` | Floor adjustment (see above) |
| `ESC` | Exit |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config.py` | Static configuration (kernels, thresholds, camera sources, logic params) |
| `ball_config.json` | Saved HSV color bounds per camera profile (`side`, `top`) |
| `floor_calibration.json` | 12 world points + per-camera pixel points for homography |
| `runtime_config.json` | Runtime-tunable params persisted live (e.g. `floor_epsilon_cm`) |
| `evaluation_log.csv` | Game-result log: score, drops, mask quality, clean-play % |

---

## Evaluation

The classical pipeline was benchmarked against **SAM 3** (Segment Anything Model 3)
masks used as ground truth on 12 diverse frames. The raw HSV mask achieves a mean
**DICE of 0.953** and **IoU of 0.912** — accuracy approaching a deep-learning model
while running at real-time speed on CPU. The Hough-circle localization is
geometrically approximate (DICE 0.797) but achieves a 100% detection rate and
provides the center-and-radius representation the game logic needs.

| Metric | Raw HSV Mask | Hough Circle |
|--------|--------------|--------------|
| DICE | 0.953 ± 0.011 | 0.797 ± 0.085 |
| IoU | 0.912 ± 0.021 | 0.665 ± 0.112 |
| Precision | 0.955 ± 0.030 | 0.756 ± 0.176 |
| Recall | 0.953 ± 0.025 | 0.878 ± 0.093 |
| F1 | 0.953 ± 0.011 | 0.797 ± 0.085 |

**Evaluation tooling:**
- `capture_eval_frames.py` — capture a frame set for evaluation.
- `generate_ground_truth.py` — produce/organize ground-truth masks.
- `run_pipeline.py` — run the detection pipeline over the evaluation frame set.
- `evaluate_segmentation.py` — compute DICE / IoU / Precision / Recall vs. ground truth.

---

## Project Structure

```
main.py                     # Main loop: dual-camera read, UI, keyboard input, calibration orchestration
camera_manager.py           # Threaded dual-camera capture (non-blocking I/O)
ball_processor.py           # Per-camera detection pipeline (blur→MOG2→HSV→fusion→morph→Hough→Kalman)
juggling_logic.py           # Game logic: V-flip kick counting, drop detection, scoring
floor_finding.py            # Homography-based floor / ground-plane agreement
config.py                   # Static configuration
config_utils.py             # Load/save HSV, floor points, runtime config
Calibration Helper script.py# Standalone HSV color-calibration GUI
ball_config.json            # Saved HSV bounds (per camera)
floor_calibration.json      # Saved floor calibration points
runtime_config.json         # Live-tunable runtime params

# Evaluation
capture_eval_frames.py      # Capture frames for benchmarking
generate_ground_truth.py    # Build ground-truth masks
run_pipeline.py             # Run pipeline over eval frames
evaluate_segmentation.py    # Compute segmentation metrics
evaluation/                 # Frames, ground truth, and pipeline outputs
```

---

## Assumptions

- **Static cameras** — both cameras are stationary throughout a session to keep
  the spatial calibration valid.
- **Object sphericity** — the ball stays near-spherical in flight (radius gating).
- **Color contrast** — a distinct color difference exists between ball and
  background for effective HSV thresholding.
- **Roughly constant lighting** — minimal ambient-light variation so the
  pre-calibrated HSV bounds remain valid.
- **Kinematic mimicry** — human juggling follows a predictable velocity-reversal
  pattern, allowing kicks to be counted via V-flips instead of expensive collision
  modeling.
