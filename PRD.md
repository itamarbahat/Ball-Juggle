# Dashboard Rebuild — Product Requirements Document

**Project:** Live Juggling Counter  
**Scope:** Replace raw OpenCV windows with a unified `pygame` dashboard; no changes to detection or game logic.  
**Target resolution:** 1920 × 1080 (configurable via constants in `dashboard.py`)  
**Language:** English only  
**Constraint:** `juggling_logic.py`, `ball_processor.py`, `floor_finding.py`, `camera_manager.py`, `config_utils.py` must NOT be modified.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  main.py  (modified — display + input only)             │
│  ├── Reads frames from DualCameraManager                │
│  ├── Calls BallProcessor / JugglingCounter (unchanged)  │
│  ├── Draws OpenCV overlays on numpy frames              │
│  │   (tracking circle, floor grid, cal HUD)             │
│  ├── Passes annotated frames → Dashboard.draw()         │
│  └── Reads input from Dashboard.poll_events()           │
│                                                         │
│  dashboard.py  (NEW)                                    │
│  └── pygame window: header + Camera A + Camera B panel  │
│      + calibration checklist + game scores + overlays   │
└─────────────────────────────────────────────────────────┘
```

**What moves to `dashboard.py`:** game score display, state banner, countdown, hit popup, drop alert, winner screen, toast messages, camera feed display.  
**What stays in `main.py` (drawn on OpenCV frames):** tracking circle/dot, tracking status bar, floor grid overlay, floor calibration HUD (`draw_cal_hud`), B/W mask debug window.

---

## 2. Window Layout (1920 × 1080)

```
┌──────────────────────── HEADER (h=55) ─────────────────────────┐
│  JUGGLING COUNTER           [key hints row]                     │
├─────────────────────────┬──────────────────────────────────────┤
│                         │  ┌── Camera Side B label (h=26) ──┐  │
│   Camera Side A         │  │  Camera Side B feed            │  │
│   (main, master)        │  │  640 × 360                     │  │
│   1270 × 714 feed       │  └────────────────────────────────┘  │
│   + 26px label bar      │  ┌── Info Panel ───────────────────┐  │
│                         │  │  CALIBRATION section            │  │
│                         │  │  [  ] HSV Color        [btn]   │  │
│                         │  │  [  ] Background       key: B  │  │
│                         │  │  [  ] Ball Radius      key: S  │  │
│                         │  │  [  ] Floor Grid       key: F  │  │
│                         │  │  ─────────────────────────────  │  │
│                         │  │  GAME STATE badge               │  │
│                         │  │  ─────────────────────────────  │  │
│                         │  │  SCORES section                 │  │
│                         │  │  > Player 1   [large number]   │  │
│                         │  │    Player 2   [large number]   │  │
│                         │  │  ─────────────────────────────  │  │
│                         │  │  [toast message area]           │  │
│                         │  └────────────────────────────────┘  │
├─────────────────────────┴──────────────────────────────────────┤
│  STATUS BAR (h=35): current status text + floor epsilon        │
└────────────────────────────────────────────────────────────────┘
```

### Pixel constants (all in `dashboard.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `W` | 1920 | Window width |
| `H` | 1080 | Window height |
| `HEADER_H` | 55 | Top header bar |
| `STATUS_H` | 35 | Bottom status bar |
| `RIGHT_W` | 650 | Right panel total width |
| `LEFT_W` | `W - RIGHT_W - 5` | Camera A panel width |
| `FEED_A_W` | `LEFT_W` | Camera A feed width |
| `FEED_A_H` | `int(FEED_A_W * 9 / 16)` | Camera A feed height (16:9) |
| `FEED_A_LABEL_H` | 26 | Label bar above Camera A feed |
| `FEED_B_W` | `RIGHT_W` | Camera B feed width |
| `FEED_B_H` | `int(FEED_B_W * 9 / 16)` | Camera B feed height |
| `FEED_B_LABEL_H` | 26 | Label bar above Camera B feed |
| `INFO_Y` | `HEADER_H + FEED_B_LABEL_H + FEED_B_H + 6` | Info panel top |
| `INFO_H` | `H - INFO_Y - STATUS_H - 4` | Info panel height |

---

## 3. Task Breakdown

---

### TASK 1 — Create `dashboard.py`

#### 1.1 Module skeleton & constants
- Define all layout constants listed in §2 at module level.
- Define colour palette as module-level tuples (RGB, pygame convention):
  - `C_BG`, `C_PANEL`, `C_BORDER`, `C_TEXT`, `C_DIM`, `C_GREEN`, `C_RED`, `C_YELLOW`, `C_CYAN`, `C_WHITE`, `C_ORANGE`
- Define `_STATE_COLOR` dict mapping game-state strings to colours.
- Define `CAL_STEPS` list of `(label, key)` tuples:
  ```python
  CAL_STEPS = [
      ("HSV Color",    "hsv"),
      ("Background",   "background"),
      ("Ball Radius",  "radius"),
      ("Floor Grid",   "floor"),
  ]
  ```

#### 1.2 `Dashboard.__init__`
- Call `pygame.init()` and `pygame.font.init()`.
- Create window: `pygame.display.set_mode((W, H))`.
- Set caption: `"Juggling Counter — Live"`.
- Create `pygame.time.Clock()` for frame-rate control.
- Load fonts via `_sys_font(size, bold)` helper (tries Segoe UI → Arial → DejaVu Sans → fallback):
  - `f_title` (22, bold), `f_head` (17, bold), `f_body` (15), `f_small` (12)
  - `f_score` (80, bold), `f_big` (120, bold), `f_state` (26, bold), `f_hint` (11)
- Initialise state attributes that `main.py` sets each frame:
  ```python
  self.cal = {k: False for _, k in CAL_STEPS}
  self.score_p1 = 0
  self.score_p2 = 0
  self.current_count = 0
  self.active_player = 1
  self.game_state = "WAITING"
  self.winner_text = ""
  self.winner_until = 0.0
  self.hit_popup_text = ""
  self.hit_popup_time = 0.0
  self.drop_alert_until = 0.0
  self.toast_text = ""
  self.toast_until = 0.0
  self.go_until = 0.0
  self.countdown_text = ""
  self.floor_epsilon_cm = 5.0
  ```

#### 1.3 `Dashboard.draw(frame_a, frame_b)` — main draw method
Called once per main-loop iteration. Sequence:
1. `self.screen.fill(C_BG)`
2. `self._draw_header()`
3. `self._draw_feed_a(frame_a, now)`
4. `self._draw_feed_b(frame_b)`
5. `self._draw_info_panel(now)`
6. `self._draw_status_bar()`
7. `pygame.display.flip()`
8. `self.clock.tick(60)`

#### 1.4 `_draw_header()`
- Filled rect `(0, 0, W, HEADER_H)` with `C_PANEL`.
- 1px border bottom with `C_BORDER`.
- Left: title text "JUGGLING COUNTER" in `C_CYAN`.
- Right: key-hint string in `C_DIM`, small font:  
  `"T=Start  N=Next Player  R=Reset  1=HSV  B=Background  S=Radius  F=Floor  G=Grid  V=Mask  ESC=Exit"`

#### 1.5 `_draw_feed_a(frame, now)`
- Draw coloured label bar (26px) above feed: text "Camera Side A  (Main / Master)".
- If `frame is not None`: scale to `(FEED_A_W, FEED_A_H)`, convert BGR→RGB, blit via `pygame.surfarray`.
- Else: draw dark rect + "WARMING UP..." text.
- Draw 1px `C_BORDER` rect around entire panel.
- **Overlay: drop alert** — if `now < self.drop_alert_until`:
  - Semi-transparent red SRCALPHA surface over the feed area.
  - "DROP!" text centred in `C_RED`, `f_state` font.
- **Overlay: hit popup** — if `self.hit_popup_text` and elapsed < `hit_popup_sec` from config:
  - Compute `alpha = int(255 * (1 - elapsed/dur))`.
  - Compute `drift = int(50 * elapsed/dur)` (drifts upward).
  - Render `self.hit_popup_text` in `C_GREEN`, `f_state`, set alpha, blit drifting centre.
- **Overlay: countdown** — if `self.countdown_text`:
  - Render large centred number in `C_CYAN`, `f_big`, alpha=200.
- **Overlay: GO!** — elif `now < self.go_until`:
  - Render "GO!" in `C_GREEN`, `f_big`, alpha=200.
- **Overlay: winner** — if `self.winner_text` and `now < self.winner_until`:
  - Semi-transparent black SRCALPHA over feed.
  - "*  WINNER  *" line above, `self.winner_text` below, both in `C_CYAN`.

#### 1.6 `_draw_feed_b(frame)`
- Draw coloured label bar (26px): "Camera Side B  (Secondary / Slave)".
- Scale frame to `(FEED_B_W, FEED_B_H)`, convert and blit.
- 1px `C_BORDER` border.
- No game overlays (secondary camera).

#### 1.7 `_draw_info_panel(now)`
Panel at `(W - RIGHT_W, INFO_Y, RIGHT_W, INFO_H)` with `C_PANEL` background.

**Calibration section:**
- Section header "CALIBRATION" in `C_CYAN`, `f_head`.
- For each step in `CAL_STEPS`:
  - If done: green `[OK]` + label.
  - If not done: dim `[ ]` + label + right-aligned key hint ("key: B", etc.).
  - For "HSV Color" not done: right-aligned button-style text "[Launch]" in `C_ORANGE`.
- Horizontal divider below.

**Game state section:**
- Section header "GAME STATE" in `f_head`.
- Centred badge with `_STATE_COLOR` colour: state string in `f_state`.
- Horizontal divider below.

**Scores section:**
- Section header "SCORES" in `C_YELLOW`, `f_head`.
- Player 1 row: `"> Player 1"` if active (green) else `"  Player 1"` (dim).
  - Score number right-aligned, `f_score` font.
- Player 2 row: same pattern.
- Score shown = `current_count` for active player; locked score for inactive.
- Horizontal divider below.

**Toast area:**
- If `self.toast_text` and `now < self.toast_until`: centred small white text.

#### 1.8 `_draw_status_bar()`
- Filled rect at `(0, H - STATUS_H, W, STATUS_H)` with `C_PANEL`.
- 1px border top.
- Left: current floor epsilon `f"Floor ε: {self.floor_epsilon_cm:.1f} cm"` in `C_DIM`.
- Right: session tip or empty.

#### 1.9 `Dashboard.poll_events()` → `list[int]`
- Iterate `pygame.event.get()`.
- On `QUIT` event: append `pygame.K_ESCAPE`.
- On `KEYDOWN` event: append `event.key`.
- Return list (may be empty or have multiple keys).

#### 1.10 `_bgr_surf(frame, size)` — module-level helper
```python
def _bgr_surf(frame, size):
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
```

---

### TASK 2 — Modify `main.py`

**Strict constraint:** only change display output, input routing, and dashboard state updates. Do not alter any variable passed to or returned from `BallProcessor`, `JugglingCounter`, or `FloorFinder`.

#### 2.1 Add imports
```python
import pygame
import subprocess
import os
from dashboard import Dashboard
```

#### 2.2 Add `_check_hsv_calibrated()` helper
```python
def _check_hsv_calibrated():
    import json
    try:
        with open("ball_config.json") as f:
            d = json.load(f)
        return "side" in d and "top" in d
    except Exception:
        return False
```

#### 2.3 Remove these functions from `main.py` entirely
(All replaced by `dashboard.py`)
- `draw_game_score`
- `draw_state_banner`
- `draw_countdown`
- `draw_hit_popup`
- `draw_drop_alert`
- `draw_winner`
- `draw_toast`
- `draw_pip`

**Keep** (still annotate OpenCV frames): `draw_tracking_info`, `draw_floor_overlay`, `draw_cal_hud`, `build_mask_view`, `_detect_ball_hsv`, all `perform_*` calibration functions.

#### 2.4 Initialise Dashboard in `main()`
After all game objects are created and before the main loop:
```python
dash = Dashboard()
dash.cal["hsv"]   = _check_hsv_calibrated()
dash.cal["floor"] = getattr(floor_finder, "calibrated", False)
dash.cal["radius"]= game_logic.baseline.get("is_set", False)
```
Remove the `show_pip` / `mask_window_placed` initialisation block (or keep `mask_window_placed` for the V-key mask window).

#### 2.5 Main loop — frame annotation (keep as-is)
The following OpenCV draw calls on numpy frames are **unchanged**:
- `draw_tracking_info(frame_side, ...)` and `draw_tracking_info(frame_top, ...)`
- Color-only mode `cv2.putText` indicators on frames
- `draw_floor_overlay(frame_top, ...)` and `draw_floor_overlay(frame_side, ...)`

Remove all other `draw_*` calls on the frames (they now live in dashboard).

#### 2.6 Dashboard state update block (add after frame annotation)
Insert before the display call:
```python
# --- sync dashboard state (no logic changes) ---
now = time.time()
dash.score_p1      = score_p1
dash.score_p2      = score_p2
dash.current_count = current_count
dash.active_player = active_player
dash.cal["radius"] = game_logic.baseline.get("is_set", False)
dash.floor_epsilon_cm = game_logic.floor_epsilon_cm

if game_logic.countdown_active:
    elapsed_cd = now - game_logic.countdown_start_time
    dash.game_state    = "COUNTDOWN"
    dash.countdown_text = str(max(1, 3 - int(elapsed_cd)))
elif now < go_until:
    dash.game_state     = "LIVE"
    dash.countdown_text = ""
elif game_logic.game_active:
    dash.game_state     = "LIVE"
    dash.countdown_text = ""
elif now < gameover_until:
    dash.game_state     = "GAME OVER"
    dash.countdown_text = ""
elif not game_logic.baseline["is_set"]:
    dash.game_state     = "WAITING"
    dash.countdown_text = ""
else:
    dash.game_state     = "READY"
    dash.countdown_text = ""
```

#### 2.7 Replace `cv2.imshow` + `cv2.waitKey` with dashboard
Remove:
```python
cv2.imshow("Main - SIDE View (Master)", frame_side)
if not show_pip:
    cv2.imshow("Main - TOP View (Slave)", frame_top)
key = cv2.waitKey(delay_ms) & 0xFF
```
Add:
```python
# frame_side = Side A (main), frame_top = Side B (secondary)
dash.draw(frame_side, frame_top)
pressed_keys = dash.poll_events()
```

#### 2.8 Key-handling loop
Replace the single `if/elif key == ...` chain with:
```python
running = True
for key in pressed_keys:
    if key == pygame.K_ESCAPE:
        running = False
        break
    elif key == pygame.K_b:       # Background reset
        ...
    elif key == pygame.K_s:       # Radius calibration
        ...
    elif key == pygame.K_r:       # Reset counter
        ...
    elif key == pygame.K_f:       # Floor calibration
        ...
    elif key == pygame.K_d:       # Color-only toggle
        ...
    elif key == pygame.K_g:       # Floor grid toggle
        ...
    elif key == pygame.K_v:       # B/W mask window toggle
        ...
    elif key == pygame.K_t:       # Start game
        ...
    elif key == pygame.K_n:       # Next player
        ...
    elif key == pygame.K_1:       # Launch HSV calibrator  ← NEW
        ...
    elif key == pygame.K_c:       # Clear eval log
        ...
    elif key == pygame.K_z:       # Left anchor up
        ...
    elif key == pygame.K_a:       # Left anchor down
        ...
    elif key == pygame.K_DOWN:    # Right anchor down  (was cv2 code 84)
        ...
    elif key == pygame.K_UP:      # Right anchor up    (was cv2 code 82)
        ...
    elif key == pygame.K_k:       # Right anchor up (fallback)
        ...
    elif key == pygame.K_m:       # Right anchor down (fallback)
        ...
    elif key == pygame.K_LEFTBRACKET:   # Floor start X left
        ...
    elif key == pygame.K_RIGHTBRACKET:  # Floor start X right
        ...
    elif key == pygame.K_QUOTE:         # Floor end X left
        ...
    elif key == pygame.K_BACKSLASH:     # Floor end X right
        ...
    elif key == pygame.K_MINUS:         # Floor epsilon -0.5
        ...
    elif key == pygame.K_EQUALS:        # Floor epsilon +0.5  (= or + key)
        ...
if not running:
    break
```
The inner logic of each branch is **identical** to the current code; only the condition changes.

#### 2.9 HSV calibrator launch (`pygame.K_1`)
```python
elif key == pygame.K_1:
    print("[INPUT] Launching HSV Calibration Helper...")
    proc = subprocess.Popen(
        ["python", "Calibration Helper script.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    proc.wait()   # block until the helper closes
    dash.cal["hsv"] = _check_hsv_calibrated()
    toast_text, toast_until = "HSV calibration updated", time.time() + ui["toast_sec"]
    dash.toast_text  = toast_text
    dash.toast_until = toast_until
    pygame.event.clear()
```

#### 2.10 Post-calibration dashboard flag updates
After each calibration step, update the corresponding `dash.cal` flag:

| Event | Flag update |
|-------|-------------|
| B pressed successfully | `dash.cal["background"] = True` |
| S (radius) completes | `dash.cal["radius"] = game_logic.baseline.get("is_set", False)` |
| F (floor) completes with `.calibrated == True` | `dash.cal["floor"] = True` |
| Key 1 (HSV) subprocess exits | `dash.cal["hsv"] = _check_hsv_calibrated()` |

#### 2.11 Floor calibration cleanup
After `perform_floor_calibration` returns, add:
```python
for wname in ["Main - SIDE View (Master)", "Main - TOP View (Slave)"]:
    try:
        cv2.destroyWindow(wname)
    except Exception:
        pass
pygame.event.clear()
```

#### 2.12 Toast routing
Whenever `toast_text` and `toast_until` are set, also set:
```python
dash.toast_text  = toast_text
dash.toast_until = toast_until
```

#### 2.13 Event overlay routing
Route transient event state to dashboard:
```python
# DROP event:
dash.drop_alert_until = now + ui["drop_flash_sec"]

# +1 hit event:
dash.hit_popup_text = logic_feedback
dash.hit_popup_time = now

# Countdown → GO! transition:
dash.go_until = now + 0.6

# Winner:
dash.winner_text  = winner_text
dash.winner_until = time.time() + ui["winner_sec"]
```

#### 2.14 Cleanup on exit
Replace `cv2.destroyAllWindows()` with:
```python
cv2.destroyAllWindows()
pygame.quit()
```

#### 2.15 Remove unused variables
- `show_pip` (no longer needed)
- `mask_window_placed` is still needed for the V-key mask window

---

### TASK 3 — Camera label rename

All display strings referring to "top" or "slave" camera are updated to "Side B" or "Secondary".  
Locations to update (in `main.py` only, do not change variable names):

| Current string | Replacement |
|----------------|-------------|
| `"SIDE: {status_side}"` | `"SIDE-A: {status_side}"` |
| `"TOP: {status_top}"` | `"SIDE-B: {status_top}"` |
| `"Main - SIDE View (Master)"` (in perform_floor_calibration imshow) | `"Floor Cal — Camera Side A"` |
| `"Main - TOP View (Slave)"` (in perform_floor_calibration imshow) | `"Floor Cal — Camera Side B"` |

Internal variable names (`frame_top`, `data_top`, etc.) are **not renamed** — this is display-only.

---

### TASK 4 — B/W Mask debug window (V key)

The mask window (`build_mask_view`) remains as a separate `cv2.imshow` window, unchanged.  
It is still toggled by the `V` key and positioned at `(40, 600)`.  
No changes required here beyond ensuring `mask_window_placed` is still tracked.

---

### TASK 5 — Update `config.py` camera label comments

In `config.py`, update the `"camera"` section comment from:
```python
"top_source": 0,    # top camera
"side_source": 0,   # side camera (master)
```
to:
```python
"top_source": 0,    # Camera Side B (secondary)
"side_source": 0,   # Camera Side A (main / master)
```
This is the only change to `config.py`.

---

## 4. Key Mapping Reference

| Action | Old (cv2) | New (pygame) |
|--------|-----------|--------------|
| Quit | `key == 27` | `key == pygame.K_ESCAPE` |
| Background | `ord('b')` or `ord('B')` | `pygame.K_b` |
| Radius | `ord('s')` or `ord('S')` | `pygame.K_s` |
| Reset | `ord('r')` or `ord('R')` | `pygame.K_r` |
| Floor | `ord('f')` or `ord('F')` | `pygame.K_f` |
| Color-only | `ord('d')` or `ord('D')` | `pygame.K_d` |
| Grid toggle | `ord('g')` or `ord('G')` | `pygame.K_g` |
| Mask window | `ord('v')` or `ord('V')` | `pygame.K_v` |
| Start game | `ord('t')` or `ord('T')` | `pygame.K_t` |
| Next player | `ord('n')` or `ord('N')` | `pygame.K_n` |
| **HSV calibrator** | *(new)* | `pygame.K_1` |
| Clear eval log | `ord('c')` or `ord('C')` | `pygame.K_c` |
| Left anchor up | `ord('a')` or `ord('A')` | `pygame.K_a` |
| Left anchor down | `ord('z')` or `ord('Z')` | `pygame.K_z` |
| Right anchor down | `key == 84` | `pygame.K_DOWN` |
| Right anchor up | `key == 82` | `pygame.K_UP` |
| Right anchor up (alt) | `ord('k')` | `pygame.K_k` |
| Right anchor down (alt) | `ord('m')` | `pygame.K_m` |
| Floor start X left | `ord('[')` | `pygame.K_LEFTBRACKET` |
| Floor start X right | `ord(']')` | `pygame.K_RIGHTBRACKET` |
| Floor end X left | `ord("'")` | `pygame.K_QUOTE` |
| Floor end X right | `ord('\\')` | `pygame.K_BACKSLASH` |
| Floor ε − 0.5 | `ord('-')` | `pygame.K_MINUS` |
| Floor ε + 0.5 | `ord('=')` or `ord('+')` | `pygame.K_EQUALS` |

---

## 5. Files Summary

| File | Action | Notes |
|------|--------|-------|
| `dashboard.py` | **CREATE** | ~350 lines; pure pygame; no game logic |
| `main.py` | **MODIFY** | Remove 8 draw functions; add dashboard integration |
| `config.py` | **MINOR EDIT** | Camera label comments only |
| `juggling_logic.py` | **NO TOUCH** | |
| `ball_processor.py` | **NO TOUCH** | |
| `floor_finding.py` | **NO TOUCH** | |
| `camera_manager.py` | **NO TOUCH** | |
| `config_utils.py` | **NO TOUCH** | |
| `Calibration Helper script.py` | **NO TOUCH** | Launched as subprocess |

---

## 6. Acceptance Criteria

1. `python main.py` opens a single pygame window at 1920×1080.
2. Both camera feeds are visible simultaneously (Side A large, Side B smaller).
3. Calibration checklist shows correct ✓/○ for HSV, background, radius, and floor at startup.
4. Pressing `1` launches the HSV helper script; after closing it, the HSV row turns green if `ball_config.json` is updated.
5. Pressing `B`, `S`, `F` marks the corresponding calibration row green upon success.
6. Game scores update live; active player is highlighted.
7. Countdown numbers, GO!, +1 popup, DROP! flash, and winner screen appear correctly over Camera A feed.
8. Toast messages appear in the info panel.
9. The B/W mask debug window (V key) still works as a separate cv2 window.
10. Floor calibration (F key) opens temporary OpenCV windows; after completion, the dashboard resumes normally.
11. All existing keyboard shortcuts behave identically to before (only key codes changed, not logic).
12. No changes to ball detection, tracking, kick counting, or drop detection.
