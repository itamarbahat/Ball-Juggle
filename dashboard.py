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
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Juggling Counter — Live")
        self.clock = pygame.time.Clock()

        # Fonts (scaled lazily to the window in _ensure_fonts)
        self._font_cache = {}
        self._fonts_for_h = -1
        self._ensure_fonts(height)

        # ---- state set by main.py every frame (no logic lives here) ----
        self.cal = {key: False for _, key, _ in CAL_STEPS}
        self.score_p1 = 0
        self.score_p2 = 0
        self.current_count = 0
        self.active_player = 1
        self.game_state = "WAITING"
        self.floor_epsilon_cm = 5.0
        self.color_only = False

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

        pygame.display.flip()
        self.clock.tick(60)

    # ------------------------------------------------------------------ #
    #  Header / status                                                   #
    # ------------------------------------------------------------------ #
    def _draw_header(self, W, h):
        pygame.draw.rect(self.screen, C_PANEL, (0, 0, W, h))
        pygame.draw.line(self.screen, C_BORDER, (0, h), (W, h), 1)
        self._text("JUGGLING COUNTER", self.f_title, C_CYAN, (16, h // 2),
                   anchor="midleft")
        hints = ("T Start   N Next   R Reset   |   1 HSV  B Bg  S Radius  "
                 "F Floor   |   G Grid  V Mask  D Color   ESC Exit")
        self._text(hints, self.f_hint, C_DIM, (W - 14, h // 2), anchor="midright")

    def _draw_status_bar(self, W, H, h):
        y = H - h
        pygame.draw.rect(self.screen, C_PANEL, (0, y, W, h))
        pygame.draw.line(self.screen, C_BORDER, (0, y), (W, y), 1)
        self._text(f"Floor agreement epsilon: {self.floor_epsilon_cm:.1f} cm",
                   self.f_small, C_DIM, (14, y + h // 2), anchor="midleft")
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
            done = self.cal.get(key, False)
            box_c = C_GREEN if done else C_DIM
            pygame.draw.rect(self.screen, box_c, (x, y + 2, 18, 18),
                             0 if done else 2, border_radius=4)
            if done:
                pygame.draw.lines(self.screen, C_BG, False,
                                  [(x + 4, y + 11), (x + 8, y + 15),
                                   (x + 15, y + 5)], 2)
            self._text(label, self.f_body, C_TEXT if done else C_DIM,
                       (x + 28, y + 11), anchor="midleft")
            hint_c = C_GREEN if done else C_ORANGE
            hint_txt = "ready" if done else hint
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
    #  Input                                                             #
    # ------------------------------------------------------------------ #
    def poll_events(self):
        """Return a list of pygame key codes pressed since the last call."""
        keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keys.append(pygame.K_ESCAPE)
            elif event.type == pygame.KEYDOWN:
                keys.append(event.key)
        return keys

    def set_toast(self, text, until):
        self.toast_text = text
        self.toast_until = until

    def close(self):
        pygame.quit()
