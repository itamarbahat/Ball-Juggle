"""
dashboard.py — Unified pygame presentation layer for the Live Juggling Counter.

This module is PURE PRESENTATION + INPUT. It contains ZERO game logic:
detection, tracking, kick-counting and drop-detection all stay in their
existing modules. `main.py` feeds this dashboard already-annotated BGR camera
frames plus a handful of plain state values each frame, and reads back the keys
the user pressed.

Layout is computed responsively from the live window size every frame, so the
same code looks correct at 1920x1080, 1920x1200, a portrait 1080x1920 booth
screen, or any resized / full-screen window.
"""

import cv2
import os
import sys
import numpy as np
import pygame
import time


# --------------------------------------------------------------------------- #
#  Palette (RGB — pygame convention)                                          #
# --------------------------------------------------------------------------- #
C_BG      = (16, 18, 24)
C_PANEL   = (26, 30, 40)
C_PANEL2  = (34, 39, 52)
C_BORDER  = (58, 64, 80)
C_TEXT    = (228, 232, 240)
C_DIM     = (128, 136, 154)
C_GREEN   = (62, 208, 120)
C_RED     = (236, 72, 72)
C_YELLOW  = (240, 200, 72)
C_CYAN    = (82, 198, 230)
C_WHITE   = (245, 248, 255)
C_ORANGE  = (242, 162, 60)

# Game-state badge colours
_STATE_COLOR = {
    "WAITING":   C_DIM,
    "READY":     C_GREEN,
    "COUNTDOWN": C_YELLOW,
    "LIVE":      C_GREEN,
    "GAME OVER": C_RED,
}

# Floor-agreement epsilon slider range (cm) used by the EPSILON settings modal.
EPS_MIN = 0.5
EPS_MAX = 20.0
EPS_STEP = 0.5

# Calibration checklist: (display label, state key, key hint)
CAL_STEPS = [
    ("HSV Color",   "hsv",        "press 1"),
    ("Background",  "background", "press B"),
    ("Ball Radius", "radius",     "press S"),
    ("Floor Grid",  "floor",      "press F"),
]


def _bgr_surf(frame, size):
    """Resize a BGR numpy frame and convert it to an RGB pygame surface."""
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def _fit(rect, aspect=16 / 9):
    """Return a (x, y, w, h) sub-rect centred in `rect` that preserves `aspect`."""
    x, y, w, h = rect
    if w / h > aspect:                      # container too wide -> pillarbox
        nh = h
        nw = int(h * aspect)
    else:                                   # container too tall -> letterbox
        nw = w
        nh = int(w / aspect)
    nx = x + (w - nw) // 2
    ny = y + (h - nh) // 2
    return (nx, ny, nw, nh)


class Dashboard:
    """Owns the single pygame window and draws the entire on-screen experience."""

    def __init__(self, width=1920, height=1080):
        # Center the window on the primary monitor so it can't open off-screen.
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

        # On Windows, declare the process DPI-aware BEFORE creating the window so a
        # 1920x1080 window maps to 1920x1080 *physical* pixels. Without this, display
        # scaling (e.g. 125% / 150%) inflates the window well past the screen edges.
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor v2
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass

        pygame.init()
        pygame.font.init()

        # Clamp the window to the usable desktop so it never overflows the screen,
        # leaving headroom for the title bar and taskbar. Capped at the requested
        # 1920x1080. The layout is fully responsive, so a smaller window is fine.
        desktop = pygame.display.Info()
        win_w = min(width, desktop.current_w - 16)
        win_h = min(height, desktop.current_h - 80)

        self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Juggling Counter — Live")
        self.clock = pygame.time.Clock()

        # Fonts (scaled lazily to the window in _ensure_fonts)
        self._font_cache = {}
        self._fonts_for_h = -1
        self._ensure_fonts(win_h)

        # ---- state set by main.py every frame (no logic lives here) ----
        self.cal = {key: False for _, key, _ in CAL_STEPS}
        self.score_p1 = 0
        self.score_p2 = 0
        self.current_count = 0
        self.active_player = 1
        self.game_state = "WAITING"
        self.floor_epsilon_cm = 5.0
        self.color_only = False

        # ---- EPSILON settings modal (its own pane, next to HSV) ----
        # Owns a working copy of the floor-agreement epsilon while open: a slider,
        # - / + buttons and the keys mutate `settings_eps`, and each change queues an
        # ABSOLUTE target value. main.py drains them via take_epsilon_events() and
        # applies each through game_logic.adjust_floor_epsilon() (no logic here).
        self.show_settings = False
        self.settings_eps = self.floor_epsilon_cm
        self._settings_btn_rect = None          # header "EPSILON" pill
        self._settings_slider_rect = None
        self._settings_slider_drag = False
        self._settings_btn_rects = {}
        self._eps_events = []

        # Help / instructions popup (opened by F1 or the "? HELP" header button).
        self.show_help = False
        self._help_btn_rect = None

        # Live B/W detection view (opened by V or the "B/W MASK" header button).
        # main.py drops the latest combined mask image here each frame while active.
        self.show_mask = False
        self.mask_frame = None
        self._mask_btn_rect = None

        # Ball-colour (HSV) calibration modal — opened by `1` or the "HSV COLOR"
        # header button. This dashboard owns only the widgets: it holds the six
        # slider values, draws a live feed + colour-mask preview, and records
        # high-level events (open/profile/save/close). main.py seeds the sliders,
        # feeds `hsv_preview_frame` each frame, and acts on take_hsv_events()
        # (saving to ball_config.json and reloading the processors) — no logic here.
        self.show_hsv = False
        self.hsv_profile = "side"               # which camera is being calibrated
        self.hsv_lower = [0, 30, 50]
        self.hsv_upper = [30, 255, 255]
        self.hsv_preview_frame = None           # clean BGR frame for the selected cam
        self.hsv_saved_until = 0.0              # set by main.py after a real write
        self._hsv_events = []
        self._hsv_slider_drag = None            # (track_rect, kind, idx, vmax) while dragging
        self._hsv_slider_rects = []
        self._hsv_btn_rects = {}
        self._hsv_btn_rect = None               # header "HSV COLOR" pill

        # transient overlays (timestamps in time.time() seconds)
        self.countdown_text = ""
        self.go_until = 0.0
        self.hit_popup_text = ""
        self.hit_popup_time = 0.0
        self.hit_popup_sec = 1.4
        self.drop_alert_until = 0.0
        self.winner_text = ""
        self.winner_until = 0.0
        self.toast_text = ""
        self.toast_until = 0.0

    # ------------------------------------------------------------------ #
    #  Fonts                                                             #
    # ------------------------------------------------------------------ #
    def _sys_font(self, size, bold=False):
        key = (size, bold)
        if key not in self._font_cache:
            self._font_cache[key] = pygame.font.SysFont(
                "Segoe UI,Arial,DejaVu Sans", size, bold=bold)
        return self._font_cache[key]

    def _ensure_fonts(self, win_h):
        """Rebuild font handles when the window height changes materially."""
        if abs(win_h - self._fonts_for_h) < 24:
            return
        self._fonts_for_h = win_h
        s = win_h / 1080.0                       # scale factor vs design height
        self.f_title = self._sys_font(int(24 * s), True)
        self.f_head  = self._sys_font(int(19 * s), True)
        self.f_body  = self._sys_font(int(16 * s))
        self.f_small = self._sys_font(int(13 * s))
        self.f_hint  = self._sys_font(int(12 * s))
        self.f_state = self._sys_font(int(30 * s), True)
        self.f_score = self._sys_font(int(72 * s), True)
        self.f_big   = self._sys_font(int(150 * s), True)
        self.f_label = self._sys_font(int(15 * s), True)

    # ------------------------------------------------------------------ #
    #  Small drawing helpers                                             #
    # ------------------------------------------------------------------ #
    def _text(self, s, font, color, pos, anchor="topleft", alpha=255):
        surf = font.render(s, True, color)
        if alpha < 255:
            surf.set_alpha(alpha)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def _panel(self, rect, fill=C_PANEL, border=C_BORDER, radius=10):
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        if border:
            pygame.draw.rect(self.screen, border, rect, width=1, border_radius=radius)

    def _divider(self, x, y, w):
        pygame.draw.line(self.screen, C_BORDER, (x, y), (x + w, y), 1)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                  #
    # ------------------------------------------------------------------ #
    def draw(self, frame_a, frame_b):
        """Render one full frame. `frame_a` = main (Side A), `frame_b` = Side B."""
        W, H = self.screen.get_size()
        self._ensure_fonts(H)
        now = time.time()

        self.screen.fill(C_BG)

        m = max(6, int(W * 0.005))               # outer margin
        header_h = int(58 * H / 1080)
        status_h = int(36 * H / 1080)

        self._draw_header(W, header_h)
        self._draw_status_bar(W, H, status_h)

        content_top = header_h + m
        content_bot = H - status_h - m
        content_h = content_bot - content_top

        right_w = int(min(max(W * 0.32, 380), 700))
        left_w = W - right_w - m * 3
        left_x = m

        # ---- Left: big primary camera (Side A) ----
        label_h = int(28 * H / 1080)
        a_outer = (left_x, content_top, left_w, content_h)
        self._draw_feed_panel(frame_a, a_outer, label_h,
                              "CAMERA SIDE A   (Main / Master)", C_CYAN, now,
                              overlays=True)

        # ---- Right column: secondary camera (Side B) over the info panel ----
        right_x = left_x + left_w + m
        # Side B feed at 16:9 plus its label bar, capped so the info panel below
        # always keeps room.
        b_h = min(int(right_w * 9 / 16) + label_h, int(content_h * 0.42))
        b_outer = (right_x, content_top, right_w, b_h)
        self._draw_feed_panel(frame_b, b_outer, label_h,
                              "CAMERA SIDE B   (Secondary)", C_DIM, now,
                              overlays=False)

        info_top = content_top + b_h + m
        info_rect = (right_x, info_top, right_w, content_bot - info_top)
        self._draw_info_panel(info_rect, now)

        # Modal overlays, drawn on top of everything else.
        if self.show_mask:
            self._draw_mask_overlay(W, H)
        if self.show_hsv:
            self._draw_hsv_overlay(W, H)
        if self.show_settings:
            self._draw_settings_overlay(W, H)
        if self.show_help:
            self._draw_help_overlay(W, H)

        pygame.display.flip()
        self.clock.tick(60)

    # ------------------------------------------------------------------ #
    #  Header / status                                                   #
    # ------------------------------------------------------------------ #
    def _pill_button(self, rect, label, active=False):
        """Draws a rounded clickable pill. `active` tints it when its popup is open."""
        hover = rect.collidepoint(pygame.mouse.get_pos())
        if active:
            fill = (30, 70, 92)
        elif hover:
            fill = C_BORDER
        else:
            fill = C_PANEL2
        pygame.draw.rect(self.screen, fill, rect, border_radius=rect.height // 2)
        pygame.draw.rect(self.screen, C_CYAN, rect, 1, border_radius=rect.height // 2)
        self._text(label, self.f_hint, C_CYAN, rect.center, anchor="center")

    def _draw_header(self, W, h):
        pygame.draw.rect(self.screen, C_PANEL, (0, 0, W, h))
        pygame.draw.line(self.screen, C_BORDER, (0, h), (W, h), 1)
        self._text("JUGGLING COUNTER", self.f_title, C_CYAN, (16, h // 2),
                   anchor="midleft")

        # Header buttons, laid out right-to-left. Rects are stored so poll_events()
        # can hit-test mouse clicks against them.
        bh = int(h * 0.56)
        by = (h - bh) // 2
        gap = 8
        self._help_btn_rect = pygame.Rect(W - int(bh * 3.4) - 14, by, int(bh * 3.4), bh)
        self._mask_btn_rect = pygame.Rect(
            self._help_btn_rect.left - gap - int(bh * 4.4), by, int(bh * 4.4), bh)
        self._hsv_btn_rect = pygame.Rect(
            self._mask_btn_rect.left - gap - int(bh * 5.0), by, int(bh * 5.0), bh)
        self._settings_btn_rect = pygame.Rect(
            self._hsv_btn_rect.left - gap - int(bh * 5.0), by, int(bh * 5.0), bh)
        self._pill_button(self._settings_btn_rect, "EPSILON", active=self.show_settings)
        self._pill_button(self._hsv_btn_rect, "HSV COLOR", active=self.show_hsv)
        self._pill_button(self._mask_btn_rect, "B/W MASK", active=self.show_mask)
        self._pill_button(self._help_btn_rect, "?  HELP", active=self.show_help)

        hints = ("T Start   N Next   R Reset   |   1 HSV  B Bg  S Radius  "
                 "F Floor   |   G Grid  V Mask  D Color   F1 Help   ESC Exit")
        self._text(hints, self.f_hint, C_DIM,
                   (self._settings_btn_rect.left - 16, h // 2), anchor="midright")

    def _draw_status_bar(self, W, H, h):
        y = H - h
        pygame.draw.rect(self.screen, C_PANEL, (0, y, W, h))
        pygame.draw.line(self.screen, C_BORDER, (0, y), (W, y), 1)
        lbl = self._text(f"Floor agreement epsilon: {self.floor_epsilon_cm:.1f} cm",
                         self.f_small, C_DIM, (14, y + h // 2), anchor="midleft")
        self._text("— click EPSILON to adjust", self.f_small, C_BORDER,
                   (lbl.right + 12, y + h // 2), anchor="midleft")

        mode = "COLOR-ONLY DETECTION" if self.color_only else "Hough + motion fusion"
        self._text(mode, self.f_small, C_DIM, (W - 14, y + h // 2),
                   anchor="midright")

    # ------------------------------------------------------------------ #
    #  Camera feed panels                                                #
    # ------------------------------------------------------------------ #
    def _draw_feed_panel(self, frame, outer, label_h, label, label_color, now,
                         overlays):
        ox, oy, ow, oh = outer
        # label bar
        pygame.draw.rect(self.screen, C_PANEL2, (ox, oy, ow, label_h),
                         border_top_left_radius=8, border_top_right_radius=8)
        self._text(label, self.f_label, label_color,
                   (ox + 10, oy + label_h // 2), anchor="midleft")

        feed_area = (ox, oy + label_h, ow, oh - label_h)
        fx, fy, fw, fh = _fit(feed_area, 16 / 9)

        if frame is not None and fw > 2 and fh > 2:
            surf = _bgr_surf(frame, (fw, fh))
            self.screen.blit(surf, (fx, fy))
        else:
            pygame.draw.rect(self.screen, (10, 12, 16), feed_area)
            self._text("WARMING UP...", self.f_head, C_YELLOW,
                       (fx + fw // 2, fy + fh // 2), anchor="center")

        pygame.draw.rect(self.screen, C_BORDER, (fx, fy, fw, fh), 1)

        if overlays:
            self._draw_feed_overlays((fx, fy, fw, fh), now)

    def _draw_feed_overlays(self, feed, now):
        fx, fy, fw, fh = feed
        cx, cy = fx + fw // 2, fy + fh // 2

        # --- DROP! red flash ---
        if now < self.drop_alert_until:
            ov = pygame.Surface((fw, fh), pygame.SRCALPHA)
            ov.fill((236, 40, 40, 90))
            pygame.draw.rect(ov, (255, 60, 60, 230), ov.get_rect(), 14)
            self.screen.blit(ov, (fx, fy))
            self._text("DROP!", self.f_big, C_RED, (cx, cy), anchor="center")

        # --- countdown number / GO! ---
        if self.countdown_text:
            self._text(self.countdown_text, self.f_big, C_YELLOW, (cx, cy),
                       anchor="center", alpha=235)
        elif now < self.go_until:
            self._text("GO!", self.f_big, C_GREEN, (cx, cy),
                       anchor="center", alpha=235)

        # --- +1 hit popup: rises and fades ---
        if self.hit_popup_text:
            elapsed = now - self.hit_popup_time
            if elapsed < self.hit_popup_sec:
                p = elapsed / self.hit_popup_sec
                alpha = int(255 * (1.0 - p))
                drift = int(fh * 0.18 * p)
                self._text(self.hit_popup_text, self.f_score, C_GREEN,
                           (cx, cy - drift), anchor="center", alpha=alpha)
            else:
                self.hit_popup_text = ""

        # --- winner overlay ---
        if self.winner_text and now < self.winner_until:
            ov = pygame.Surface((fw, fh), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 150))
            self.screen.blit(ov, (fx, fy))
            self._text("WINNER", self.f_head, C_YELLOW,
                       (cx, cy - int(fh * 0.10)), anchor="center")
            self._text(self.winner_text, self.f_state, C_CYAN, (cx, cy),
                       anchor="center")

    # ------------------------------------------------------------------ #
    #  Info panel (calibration checklist + state + scores + toast)       #
    # ------------------------------------------------------------------ #
    def _draw_info_panel(self, rect, now):
        rx, ry, rw, rh = rect
        if rh < 40:
            return
        self._panel(rect, fill=C_PANEL)
        pad = int(rw * 0.05)
        x = rx + pad
        w = rw - pad * 2
        y = ry + pad

        # ---- Calibration checklist ----
        self._text("CALIBRATION", self.f_head, C_CYAN, (x, y))
        y += int(self.f_head.get_height() * 1.4)
        row_h = max(int(self.f_body.get_height() * 1.55), 26)
        for label, key, hint in CAL_STEPS:
            # Tri-state: falsy = not done, "saved" = loaded from a config file,
            # any other truthy value ("session"/True) = calibrated this run.
            state = self.cal.get(key, False)
            if state in ("saved", "SAVED"):
                box_c, hint_c, hint_txt, done = C_CYAN, C_CYAN, "saved", True
            elif state:
                box_c, hint_c, hint_txt, done = C_GREEN, C_GREEN, "ready", True
            else:
                box_c, hint_c, hint_txt, done = C_DIM, C_ORANGE, hint, False
            pygame.draw.rect(self.screen, box_c, (x, y + 2, 18, 18),
                             0 if done else 2, border_radius=4)
            if done:
                pygame.draw.lines(self.screen, C_BG, False,
                                  [(x + 4, y + 11), (x + 8, y + 15),
                                   (x + 15, y + 5)], 2)
            self._text(label, self.f_body, C_TEXT if done else C_DIM,
                       (x + 28, y + 11), anchor="midleft")
            self._text(hint_txt, self.f_small, hint_c, (x + w, y + 11),
                       anchor="midright")
            y += row_h

        y += pad // 2
        self._divider(x, y, w)
        y += pad // 2

        # ---- Game-state badge ----
        state = self.game_state
        color = _STATE_COLOR.get(state, C_DIM)
        badge_h = int(self.f_state.get_height() * 1.4)
        badge = (x, y, w, badge_h)
        pygame.draw.rect(self.screen, C_PANEL2, badge, border_radius=8)
        pygame.draw.rect(self.screen, color, badge, 2, border_radius=8)
        self._text(state, self.f_state, color,
                   (x + w // 2, y + badge_h // 2), anchor="center")
        y += badge_h + pad // 2
        self._divider(x, y, w)
        y += pad // 2

        # ---- Scores ----
        self._text("SCORES", self.f_head, C_YELLOW, (x, y))
        y += int(self.f_head.get_height() * 1.3)
        score_row_h = int(self.f_score.get_height() * 1.05)
        self._draw_score_row(x, y, w, score_row_h, 1)
        y += score_row_h
        self._draw_score_row(x, y, w, score_row_h, 2)
        y += score_row_h + pad // 2

        # ---- Toast ----
        if self.toast_text and now < self.toast_until:
            self._divider(x, y, w)
            y += pad // 2
            self._text(self.toast_text, self.f_body, C_WHITE,
                       (x + w // 2, y + self.f_body.get_height() // 2),
                       anchor="center")

    def _draw_score_row(self, x, y, w, h, player):
        active = (self.active_player == player)
        if player == 1:
            base_c = (120, 170, 240)
            score = self.current_count if active else self.score_p1
        else:
            base_c = (130, 235, 150)
            score = self.current_count if active else self.score_p2
        color = base_c if active else C_DIM

        if active:
            pygame.draw.rect(self.screen, C_PANEL2, (x, y, w, h),
                             border_radius=8)
            pygame.draw.rect(self.screen, color, (x, y, 5, h),
                             border_radius=3)

        prefix = ">" if active else "  "
        self._text(f"{prefix} Player {player}", self.f_body, color,
                   (x + 14, y + h // 2), anchor="midleft")
        self._text(str(score), self.f_score, color,
                   (x + w - 12, y + h // 2), anchor="midright")

    # ------------------------------------------------------------------ #
    #  Live B/W detection view                                           #
    # ------------------------------------------------------------------ #
    def _draw_mask_overlay(self, W, H):
        """Centered live view of the post-morphology detection mask (B/W).

        `self.mask_frame` is a BGR image built by main.py's build_mask_view().
        It refreshes every frame, so this view is live while it stays open.
        """
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 200))
        self.screen.blit(ov, (0, 0))

        pw = int(min(W * 0.86, 1500))
        ph = int(min(H * 0.78, 760))
        px = (W - pw) // 2
        py = (H - ph) // 2
        self._panel((px, py, pw, ph), fill=C_PANEL, border=C_CYAN, radius=14)

        pad = int(pw * 0.025)
        title_h = self.f_head.get_height()
        self._text("LIVE DETECTION  (B/W, after morphology + filters)",
                   self.f_head, C_CYAN, (px + pad, py + pad))
        self._text("press V or click to close", self.f_small, C_DIM,
                   (px + pw - pad, py + pad + title_h // 2), anchor="midright")

        img_area = (px + pad, py + pad + int(title_h * 1.8),
                    pw - pad * 2, ph - pad * 2 - int(title_h * 1.8))
        if self.mask_frame is not None and img_area[2] > 4 and img_area[3] > 4:
            fh, fw = self.mask_frame.shape[:2]
            ix, iy, iw, ih = _fit(img_area, fw / fh)
            self.screen.blit(_bgr_surf(self.mask_frame, (iw, ih)), (ix, iy))
            pygame.draw.rect(self.screen, C_BORDER, (ix, iy, iw, ih), 1)
        else:
            ax, ay, aw, ah = img_area
            self._text("WARMING UP...", self.f_head, C_YELLOW,
                       (ax + aw // 2, ay + ah // 2), anchor="center")

    # ------------------------------------------------------------------ #
    #  Ball-colour (HSV) calibration modal                               #
    # ------------------------------------------------------------------ #
    def open_hsv(self, profile, lower, upper):
        """Open the HSV modal seeded with a profile's saved bounds (called by main.py)."""
        self.set_hsv_values(profile, lower, upper)
        self.show_hsv = True
        self.show_help = False
        self.show_mask = False

    def set_hsv_values(self, profile, lower, upper):
        """Replace the live slider values, e.g. after switching the active camera."""
        self.hsv_profile = profile
        self.hsv_lower = [int(v) for v in lower]
        self.hsv_upper = [int(v) for v in upper]

    def take_hsv_events(self):
        """Drain the queued HSV actions for main.py: ('open',) ('profile', name)
        ('save',) ('close',). main.py owns the file I/O and processor reload."""
        events, self._hsv_events = self._hsv_events, []
        return events

    def confirm_hsv_saved(self):
        """Called by main.py after the bounds are actually written to disk, so the
        modal can flash a truthful 'SAVED' badge (the toast is hidden behind it)."""
        self.hsv_saved_until = time.time() + 2.2

    def _draw_hsv_overlay(self, W, H):
        """Modal: live feed + colour mask preview and six draggable HSV sliders."""
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 205))
        self.screen.blit(ov, (0, 0))

        pw = int(min(W * 0.82, 1320))
        ph = int(min(H * 0.90, 920))
        px = (W - pw) // 2
        py = (H - ph) // 2
        self._panel((px, py, pw, ph), fill=C_PANEL, border=C_CYAN, radius=14)

        pad = int(pw * 0.03)
        x = px + pad
        y = py + pad
        inner_w = pw - pad * 2

        cam_name = "SIDE A (Main)" if self.hsv_profile == "side" else "SIDE B (Secondary)"
        title_h = self.f_state.get_height()
        self._text(f"BALL COLOR CALIBRATION  -  CAMERA {cam_name}",
                   self.f_state, C_CYAN, (x, y))
        self._text("ESC / CLOSE to exit", self.f_small, C_DIM,
                   (px + pw - pad, y + title_h // 2), anchor="midright")
        y += int(title_h * 1.45)
        self._text("Drag the sliders until the ball is WHITE and the background is "
                   "BLACK, then press SAVE.", self.f_body, C_DIM, (x, y))
        y += int(self.f_body.get_height() * 1.7)

        # ---- previews: live feed (left) + resulting colour mask (right) ----
        img_h = int(ph * 0.34)
        gap = pad
        img_w = (inner_w - gap) // 2
        self._draw_hsv_preview((x, y, img_w, img_h), "LIVE FEED", show_mask=False)
        self._draw_hsv_preview((x + img_w + gap, y, img_w, img_h),
                               "COLOR MASK", show_mask=True)
        y += img_h + pad

        # ---- six sliders, two columns (lower / upper) ----
        self._hsv_slider_rects = []
        specs = [
            ("Lower Hue", "lower", 0, 179), ("Upper Hue", "upper", 0, 179),
            ("Lower Sat", "lower", 1, 255), ("Upper Sat", "upper", 1, 255),
            ("Lower Val", "lower", 2, 255), ("Upper Val", "upper", 2, 255),
        ]
        col_gap = pad
        col_w = (inner_w - col_gap) // 2
        row_h = int(self.f_body.get_height() * 2.2)
        for i, (label, kind, idx, vmax) in enumerate(specs):
            col = i % 2
            row = i // 2
            sx = x + col * (col_w + col_gap)
            sy = y + row * row_h
            self._draw_hsv_slider(sx, sy, col_w, label, kind, idx, vmax)
        y += row_h * 3 + pad // 2

        # ---- action buttons: profile toggles (left), save / close (right) ----
        self._hsv_btn_rects = {}
        bh = int(self.f_body.get_height() * 2.0)
        bw = int(inner_w * 0.22)
        bgap = pad // 2
        self._draw_hsv_button((x, y, bw, bh), "side", "CAMERA SIDE A",
                              active=self.hsv_profile == "side")
        self._draw_hsv_button((x + bw + bgap, y, bw, bh), "top", "CAMERA SIDE B",
                              active=self.hsv_profile == "top")
        self._draw_hsv_button((px + pw - pad - bw, y, bw, bh), "close", "CLOSE")
        save_rect = (px + pw - pad - bw * 2 - bgap, y, bw, bh)
        self._draw_hsv_button(save_rect, "save", "SAVE", kind="primary")

        # Truthful save confirmation: only shown after main.py actually wrote the
        # file (confirm_hsv_saved). Centred above the SAVE button so it's visible
        # on top of the modal, unlike the info-panel toast which the modal hides.
        if time.time() < self.hsv_saved_until:
            cam = "SIDE A" if self.hsv_profile == "side" else "SIDE B"
            self._text(f"✓ SAVED  ({cam})", self.f_label, C_GREEN,
                       (save_rect[0] + bw // 2, y - int(bh * 0.55)), anchor="center")

    def _draw_hsv_preview(self, rect, label, show_mask):
        rx, ry, rw, rh = rect
        label_h = int(self.f_small.get_height() * 1.7)
        pygame.draw.rect(self.screen, C_PANEL2, (rx, ry, rw, label_h),
                         border_top_left_radius=8, border_top_right_radius=8)
        self._text(label, self.f_small, C_CYAN, (rx + 8, ry + label_h // 2),
                   anchor="midleft")

        area = (rx, ry + label_h, rw, rh - label_h)
        frame = self.hsv_preview_frame
        if frame is not None:
            if show_mask:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                m = cv2.inRange(hsv, np.array(self.hsv_lower, dtype=np.uint8),
                                np.array(self.hsv_upper, dtype=np.uint8))
                img = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            else:
                img = frame
            fh, fw = img.shape[:2]
            ix, iy, iw, ih = _fit(area, fw / fh)
            self.screen.blit(_bgr_surf(img, (iw, ih)), (ix, iy))
            pygame.draw.rect(self.screen, C_BORDER, (ix, iy, iw, ih), 1)
        else:
            ax, ay, aw, ah = area
            pygame.draw.rect(self.screen, (10, 12, 16), area)
            self._text("WAITING FOR CAMERA...", self.f_small, C_YELLOW,
                       (ax + aw // 2, ay + ah // 2), anchor="center")

    def _draw_hsv_slider(self, sx, sy, w, label, kind, idx, vmax):
        val = (self.hsv_lower if kind == "lower" else self.hsv_upper)[idx]
        self._text(label, self.f_small, C_TEXT, (sx, sy))
        self._text(str(int(val)), self.f_small, C_CYAN, (sx + w, sy),
                   anchor="topright")

        track_y = sy + int(self.f_small.get_height() * 1.6)
        track_h = 6
        track = pygame.Rect(sx, track_y, w, track_h)
        pygame.draw.rect(self.screen, C_BORDER, track, border_radius=3)
        frac = (val / vmax) if vmax else 0.0
        pygame.draw.rect(self.screen, C_CYAN, (sx, track_y, int(w * frac), track_h),
                         border_radius=3)
        knob_x = sx + int(w * frac)
        knob_c = (track_y + track_h // 2)
        pygame.draw.circle(self.screen, C_WHITE, (knob_x, knob_c), 9)
        pygame.draw.circle(self.screen, C_CYAN, (knob_x, knob_c), 9, 2)
        self._hsv_slider_rects.append((track, kind, idx, vmax))

    def _draw_hsv_button(self, rect, name, label, active=False, kind="toggle"):
        r = pygame.Rect(rect)
        hover = r.collidepoint(pygame.mouse.get_pos())
        if kind == "primary":
            fill = (90, 225, 150) if hover else C_GREEN
            txt, border = C_BG, C_GREEN
        elif active:
            fill, txt, border = (30, 70, 92), C_CYAN, C_CYAN
        else:
            fill = C_BORDER if hover else C_PANEL2
            txt, border = C_TEXT, C_CYAN
        pygame.draw.rect(self.screen, fill, r, border_radius=8)
        pygame.draw.rect(self.screen, border, r, 1, border_radius=8)
        self._text(label, self.f_small, txt, r.center, anchor="center")
        self._hsv_btn_rects[name] = r

    def _poll_hsv(self):
        """Owns all input while the HSV modal is open; returns no game keys so the
        rest of the app stays inert. Records semantic events for main.py to act on."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._hsv_events.append(("close",))
                self.show_hsv = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._hsv_events.append(("close",))
                    self.show_hsv = False
                elif event.key == pygame.K_RETURN:
                    self._hsv_events.append(("save",))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._hsv_mouse_down(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                if self._hsv_slider_drag is not None and event.buttons[0]:
                    self._hsv_drag_to(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._hsv_slider_drag = None
        return []

    def _hsv_mouse_down(self, pos):
        for name, rect in self._hsv_btn_rects.items():
            if rect.collidepoint(pos):
                if name in ("side", "top"):
                    self._hsv_events.append(("profile", name))
                elif name == "save":
                    self._hsv_events.append(("save",))
                elif name == "close":
                    self._hsv_events.append(("close",))
                    self.show_hsv = False
                return
        for track, kind, idx, vmax in self._hsv_slider_rects:
            if track.inflate(0, 18).collidepoint(pos):     # generous vertical hit area
                self._hsv_slider_drag = (track, kind, idx, vmax)
                self._hsv_drag_to(pos)
                return

    def _hsv_drag_to(self, pos):
        track, kind, idx, vmax = self._hsv_slider_drag
        frac = (pos[0] - track.x) / max(1, track.width)
        frac = min(1.0, max(0.0, frac))
        arr = self.hsv_lower if kind == "lower" else self.hsv_upper
        arr[idx] = int(round(frac * vmax))

    # ------------------------------------------------------------------ #
    #  EPSILON settings modal (drop-sensitivity tuner)                   #
    # ------------------------------------------------------------------ #
    def open_settings(self):
        """Open the EPSILON pane, seeded with the current live value."""
        self.settings_eps = self.floor_epsilon_cm
        self.show_settings = True
        self.show_help = False
        self.show_mask = False
        self.show_hsv = False

    def take_epsilon_events(self):
        """Drain queued ABSOLUTE target epsilon values (cm) from the EPSILON pane.
        main.py applies each via game_logic.adjust_floor_epsilon() (delta computed
        sequentially at apply-time, so rapid clicks never desync)."""
        events, self._eps_events = self._eps_events, []
        return events

    def _set_settings_eps(self, value, queue=True):
        """Clamp + store the working epsilon, and (optionally) queue it for main.py."""
        self.settings_eps = min(EPS_MAX, max(EPS_MIN, round(value, 2)))
        if queue:
            self._eps_events.append(self.settings_eps)

    def _draw_settings_overlay(self, W, H):
        """Modal: a big live value, a draggable slider and - / + buttons that tune
        the floor-agreement epsilon (how close the two cameras must agree the ball
        is on the floor before a drop is confirmed)."""
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 205))
        self.screen.blit(ov, (0, 0))

        pw = int(min(W * 0.58, 880))
        ph = int(min(H * 0.66, 640))
        px = (W - pw) // 2
        py = (H - ph) // 2
        self._panel((px, py, pw, ph), fill=C_PANEL, border=C_CYAN, radius=14)

        pad = int(pw * 0.05)
        x = px + pad
        y = py + pad
        inner_w = pw - pad * 2
        title_h = self.f_state.get_height()

        self._text("DROP SENSITIVITY  -  FLOOR EPSILON", self.f_state, C_CYAN, (x, y))
        self._text("ESC / CLOSE to exit", self.f_small, C_DIM,
                   (px + pw - pad, y + title_h // 2), anchor="midright")
        y += int(title_h * 1.6)

        for line in (
            "A drop is confirmed only when BOTH cameras agree the ball is on the",
            "floor within this distance. Wider = drops caught sooner; too wide =",
            "false drops mid-juggle. Tune live until drops register cleanly.",
        ):
            self._text(line, self.f_body, C_DIM, (x, y))
            y += int(self.f_body.get_height() * 1.32)
        y += pad // 2

        # ---- big live value ----
        self._text(f"{self.settings_eps:.1f} cm", self.f_score, C_GREEN,
                   (px + pw // 2, y + self.f_score.get_height() // 2), anchor="center")
        y += int(self.f_score.get_height() * 1.15)

        # ---- slider ----
        track_h = 8
        track = pygame.Rect(x, y, inner_w, track_h)
        self._settings_slider_rect = track
        pygame.draw.rect(self.screen, C_BORDER, track, border_radius=4)
        frac = (self.settings_eps - EPS_MIN) / (EPS_MAX - EPS_MIN)
        frac = min(1.0, max(0.0, frac))
        pygame.draw.rect(self.screen, C_CYAN, (x, y, int(inner_w * frac), track_h),
                         border_radius=4)
        knob_x = x + int(inner_w * frac)
        knob_y = y + track_h // 2
        pygame.draw.circle(self.screen, C_WHITE, (knob_x, knob_y), 12)
        pygame.draw.circle(self.screen, C_CYAN, (knob_x, knob_y), 12, 2)
        self._text(f"{EPS_MIN:.1f}", self.f_small, C_DIM, (x, y + 18))
        self._text(f"{EPS_MAX:.0f} cm", self.f_small, C_DIM,
                   (x + inner_w, y + 18), anchor="topright")
        y += track_h + int(self.f_small.get_height() * 2.6)

        # ---- buttons: -0.5 , +0.5 , close ----
        self._settings_btn_rects = {}
        bh = int(self.f_body.get_height() * 2.2)
        bw = int(inner_w * 0.2)
        bgap = pad // 2
        self._draw_settings_button((x, y, bw, bh), "minus", f"-  {EPS_STEP:.1f}")
        self._draw_settings_button((x + bw + bgap, y, bw, bh), "plus",
                                   f"+  {EPS_STEP:.1f}", kind="primary")
        self._draw_settings_button((px + pw - pad - bw, y, bw, bh), "close", "CLOSE")

    def _draw_settings_button(self, rect, name, label, kind="toggle"):
        r = pygame.Rect(rect)
        hover = r.collidepoint(pygame.mouse.get_pos())
        if kind == "primary":
            fill = (90, 225, 150) if hover else C_GREEN
            txt, border = C_BG, C_GREEN
        else:
            fill = C_BORDER if hover else C_PANEL2
            txt, border = C_TEXT, C_CYAN
        pygame.draw.rect(self.screen, fill, r, border_radius=8)
        pygame.draw.rect(self.screen, border, r, 1, border_radius=8)
        self._text(label, self.f_body, txt, r.center, anchor="center")
        self._settings_btn_rects[name] = r

    def _poll_settings(self):
        """Owns all input while the EPSILON modal is open; returns no game keys."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.show_settings = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.show_settings = False
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._set_settings_eps(self.settings_eps - EPS_STEP)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self._set_settings_eps(self.settings_eps + EPS_STEP)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._settings_mouse_down(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                if self._settings_slider_drag and event.buttons[0]:
                    self._settings_drag_to(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self._settings_slider_drag:
                    self._settings_slider_drag = False
                    self._eps_events.append(self.settings_eps)   # commit on release
        return []

    def _settings_mouse_down(self, pos):
        for name, rect in self._settings_btn_rects.items():
            if rect.collidepoint(pos):
                if name == "minus":
                    self._set_settings_eps(self.settings_eps - EPS_STEP)
                elif name == "plus":
                    self._set_settings_eps(self.settings_eps + EPS_STEP)
                elif name == "close":
                    self.show_settings = False
                return
        if self._settings_slider_rect and \
                self._settings_slider_rect.inflate(0, 22).collidepoint(pos):
            self._settings_slider_drag = True
            self._settings_drag_to(pos)            # jump-to-click; commit on release

    def _settings_drag_to(self, pos):
        track = self._settings_slider_rect
        frac = (pos[0] - track.x) / max(1, track.width)
        frac = min(1.0, max(0.0, frac))
        # Snap to the step grid so the displayed value matches what gets saved.
        raw = EPS_MIN + frac * (EPS_MAX - EPS_MIN)
        snapped = round(raw / EPS_STEP) * EPS_STEP
        self._set_settings_eps(snapped, queue=False)   # preview only while dragging

    # ------------------------------------------------------------------ #
    #  Help / instructions popup                                         #
    # ------------------------------------------------------------------ #
    def _draw_help_overlay(self, W, H):
        """Centered modal that explains the calibration steps and hotkeys."""
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 185))
        self.screen.blit(ov, (0, 0))

        pw = int(min(W * 0.74, 1120))
        ph = int(min(H * 0.88, 840))
        px = (W - pw) // 2
        py = (H - ph) // 2
        self._panel((px, py, pw, ph), fill=C_PANEL, border=C_CYAN, radius=14)

        pad = int(pw * 0.045)
        x = px + pad
        y = py + pad
        title_h = self.f_state.get_height()

        self._text("HOW TO CALIBRATE & PLAY", self.f_state, C_CYAN, (x, y))
        self._text("press any key or click to close", self.f_small, C_DIM,
                   (px + pw - pad, y + title_h // 2), anchor="midright")
        y += int(title_h * 1.7)

        H1, B, S = self.f_head, self.f_body, self.f_small
        lines = [
            ("CALIBRATION  -  do these in order:", H1, C_YELLOW),
            ("1.  HSV COLOR    Press 1 (or click HSV COLOR). Drag the 6 sliders so the", B, C_TEXT),
            ("                 ball is WHITE and the background BLACK, then click SAVE.", B, C_DIM),
            ("2.  BACKGROUND   Step out of the frame and press B.", B, C_TEXT),
            ("3.  BALL RADIUS  Put the ball on the floor and press S.", B, C_TEXT),
            ("4.  FLOOR GRID   Press F. Put the ball on each of the 12 floor marks", B, C_TEXT),
            ("                 and press SPACE at each (the ball must be visible in", B, C_DIM),
            ("                 BOTH cameras).   BACKSPACE = undo,   ESC = cancel.", B, C_DIM),
            ("", B, C_DIM),
            ("FINE-TUNE THE FLOOR LINE", H1, C_CYAN),
            ("A / Z  left anchor      UP / DOWN  right anchor      [  ]  left edge", B, C_TEXT),
            ("'  \\  right edge        - / +  floor-agreement epsilon", B, C_TEXT),
            ("                 (or click EPSILON in the top bar for a slider + buttons)", S, C_DIM),
            ("                 A wider epsilon catches drops sooner; too wide = false drops.", S, C_DIM),
            ("", B, C_DIM),
            ("GAMEPLAY", H1, C_CYAN),
            ("T  start (3-second countdown)     N  next player     R  reset score", B, C_TEXT),
            ("", B, C_DIM),
            ("The HSV and FLOOR checks show CYAN \"saved\" when loaded from a file,", S, C_DIM),
            ("and GREEN \"ready\" after you calibrate them in this session.", S, C_DIM),
        ]
        for text, font, color in lines:
            if not text:
                y += int(B.get_height() * 0.5)
                continue
            self._text(text, font, color, (x, y))
            y += int(font.get_height() * 1.34)

    # ------------------------------------------------------------------ #
    #  Input                                                             #
    # ------------------------------------------------------------------ #
    def poll_events(self):
        """Return a list of pygame key codes pressed since the last call.

        The help popup and the live B/W mask view are handled here
        (presentation-only): F1 / V or the header buttons open them, and while
        one is open it is modal — any key or click dismisses it and that input is
        swallowed so it can't trigger an action.
        """
        # The HSV calibration modal is fully interactive (draggable sliders,
        # buttons), so it takes over input entirely while open.
        if self.show_hsv:
            return self._poll_hsv()
        if self.show_settings:
            return self._poll_settings()

        keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keys.append(pygame.K_ESCAPE)
            elif event.type == pygame.KEYDOWN:
                if self.show_help or self.show_mask:   # modal: any key closes it
                    self.show_help = False
                    self.show_mask = False
                    continue
                if event.key == pygame.K_F1:
                    self.show_help = True
                    continue
                if event.key == pygame.K_v:
                    self.show_mask = True
                    continue
                keys.append(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                p = event.pos
                if self._hsv_btn_rect and self._hsv_btn_rect.collidepoint(p):
                    # main.py seeds the sliders from the saved profile, then opens.
                    self._hsv_events.append(("open",))
                elif self._settings_btn_rect and self._settings_btn_rect.collidepoint(p):
                    self.open_settings()
                elif self._help_btn_rect and self._help_btn_rect.collidepoint(p):
                    self.show_help = not self.show_help
                    self.show_mask = False
                elif self._mask_btn_rect and self._mask_btn_rect.collidepoint(p):
                    self.show_mask = not self.show_mask
                    self.show_help = False
                elif self.show_help or self.show_mask:
                    self.show_help = False
                    self.show_mask = False
        return keys

    def set_toast(self, text, until):
        self.toast_text = text
        self.toast_until = until

    def close(self):
        pygame.quit()
