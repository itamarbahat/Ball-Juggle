import time
import numpy as np
import csv
import os
import threading
from collections import deque
from config import CONFIG
from Utils.config_utils import load_floor_epsilon, save_floor_epsilon

try:
    import pygame
    pygame.mixer.init()
    _PYGAME_AVAILABLE = True
except Exception:
    _PYGAME_AVAILABLE = False
    print("[WARNING] pygame not available – drop sound will be silent.")

class JugglingCounter:
    """Orchestrates the juggling state machine using velocity-flip inertia-based hit detection,
    dual-camera velocity tracking, homography-based floor verification, and a
    retroactive drop wrapper that removes false bounce hits.
    """

    def __init__(self, history_len=150, floor_finder=None):
        """Initialize the juggling counter with tracking history, floor geometry state,
        and the live tuning parameters used by the hit/drop logic.

        Args:
            history_len (int): Number of prior frames retained for velocity
                smoothing and event history. VALUE: 150 frames (5 seconds at 30 FPS).
            floor_finder (FloorFinder): Optional instance of FloorFinder for homography-based
                floor verification.
        """
        #homography instance for floor detection
        self.floor_finder = floor_finder
        
        # --- State Variables ---
        self.count = 0    
        self.last_hit_time = 0
        self.cooldown = 0.05        # Seconds between valid hits (prevents double counting)
        
        # --- Inertia State Machine Variables ---
        self.is_falling = False     # Boolean flag: Is the ball confirmed to be falling?
        self.fall_counter = 0       # Counter for consecutive falling frames
        self.rise_counter = 0       # Counter for consecutive rising frames (for hit confirmation)
        
        # --- Configuration Loading ---
        self.jitter_thresh = CONFIG["logic"].get("jitter_threshold", 20)
        self.floor_tol = CONFIG["logic"].get("floor_tolerance", 30)
        self.header_ratio = CONFIG["logic"].get("header_height_ratio", 0.35)
        self.required_inertia_frames = CONFIG["logic"].get("inertia_frames", 2)

        # --- 2-Point Anchor System (Perspective Floor Map) ---
        # Instead of a single slope, we hold two absolute Y heights for Left and Right edges.
        self.floor_left_y = CONFIG["logic"].get("floor_left_y", 350)
        self.floor_right_y = CONFIG["logic"].get("floor_right_y", 350)
        # Defines the active playing area width. 
        # Ball drops outside these X coordinates (e.g., hitting a wall) are ignored.
        # Defaults to full frame width (0 to 640).
        self.floor_x_start = 0 
        self.floor_x_end = CONFIG["camera"]["width"]
        # --- Floor Epsilon Configuration --- (World distance tolerance)
        self.floor_epsilon_cm = load_floor_epsilon(default_value=5.0)
        print(f"[INFO] Floor epsilon loaded: {self.floor_epsilon_cm:.2f} cm")
        
        #--- History Buffers ---
        self.history_len = history_len            
        self.side_history = deque(maxlen=history_len)      # Stores (y, timestamp, velocity) tuples for side camera tracking
        self.header_history = deque(maxlen=history_len)    # Stores (radius, timestamp) tuples for header camera tracking

        #--- Prediction state (for occlusion handling) ---  
        self.last_known_pos = None   # (y, timestamp) of the last known ball position
        self.last_velocity = 0
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = 30

        # --- Game State Management ---
        self.game_active = False            # Is the game currently running and counting?
        self.countdown_active = False       # Is the 3-second countdown running?
        self.countdown_start_time = 0       # Timestamp of when 'T' was pressed

        # --- Header / Top-Camera Logic : Extension State ---
        # These fields are kept as a scaffold for a future 3rd-camera or top-down
        # header pipeline that measures radius expansion/contraction (radius-flip)
        # and triggers a header event from a dedicated ceiling stream.
        # In the current dual-camera gameplay loop, the primary floor logic is driven
        # by the side camera plus the calibrated FloorFinder, so this state is not
        # directly linked to the active scoring path.
        self.header_baseline_r = None         # Calibrated minimum radius for a valid header height
        self.is_rising_to_header = False      # Reserved for a future header-camera rise-state machine
        self.rise_counter = 0                 # Reserved counter for header radius growth tracking
        self.is_first_kick = True             # Protects the ball while it rests on the floor before the first flick-up
        self.is_falling_from_header = False   # Reserved flag for a future header fall-state transition
        self.header_growth_counter = 0        # Number of consecutive frames where header-camera radius grows
        self.header_shrink_counter = 0        # Number of consecutive frames where header-camera radius shrinks
        self.is_header_growing = False        # Reserved flag for the future header-camera growth direction state

        # --- Behavioral Death Trackers ---
        self.lost_frames = 0
        self.frozen_frames = 0              # How many frames the ball has been motionless 
        self.dying_start_time = 0           # Timestamp when the ball was first lost or frozen
        self.hit_timestamps = []            # List of timestamps for retroactive drop analysis/for rollback


        self.EDGE_MARGIN_PX = 30            # Pixels from the left/right edges of the frame that are considered "out of bounds"
        self.last_exit_side = None

    def calibrate_head_height(self, radius):
        """Sets the header-camera radius baseline for future header detection."""
        try:
            self.header_baseline_r = float(radius)
            print(f"[CAL] Header baseline set: min radius = {self.header_baseline_r}")
            return True
        except (TypeError, ValueError):
            print("[WARNING] Invalid header radius provided for calibration.")
            return False
    

    def adjust_floor_start_x(self, delta):
        self.floor_x_start += delta
        if self.floor_x_start >= self.floor_x_end: self.floor_x_start = self.floor_x_end - 10
        if self.floor_x_start < 0: self.floor_x_start = 0
        print(f"[INFO] Floor start X: {self.floor_x_start}")

    def adjust_floor_end_x(self, delta):
        self.floor_x_end += delta
        max_width = CONFIG["camera"]["width"]
        if self.floor_x_end <= self.floor_x_start: self.floor_x_end = self.floor_x_start + 10
        if self.floor_x_end > max_width: self.floor_x_end = max_width
        print(f"[INFO] Floor end X: {self.floor_x_end}")
        
    def adjust_left_anchor(self, delta):
        self.floor_left_y += delta
        print(f"[INFO] Left anchor Y: {self.floor_left_y}")

    def adjust_right_anchor(self, delta):
        self.floor_right_y += delta
        print(f"[INFO] Right anchor Y: {self.floor_right_y}")

    def _play_drop_sound(self):
        def _play():
            try:
                if _PYGAME_AVAILABLE:
                    sound_path = os.path.join(os.path.dirname(__file__), "referee-whistle-sound-effect.mp3")
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()
            except Exception as e:
                print(f"[WARNING] Could not play drop sound: {e}")
        threading.Thread(target=_play, daemon=True).start()

    def adjust_floor_epsilon(self, delta_cm):
        self.floor_epsilon_cm = max(0.5, round(self.floor_epsilon_cm + delta_cm, 2))
        save_floor_epsilon(self.floor_epsilon_cm)
        print(f"[INFO] Floor epsilon: {self.floor_epsilon_cm:.2f} cm")

    def _calculate_velocity(self, current_y, current_time):
        """Returns vertical pixel delta: positive = falling, negative = rising."""
        if len(self.side_history) == 0:
            return 0
        prev_y = self.side_history[-1][0]
        return current_y - prev_y


    def save_evaluation_data(self, final_score, deducted, duration):
        """Logs session stats (score, deductions, clean-play %) to CSV."""
        file_path = "evaluation_log.csv"
        file_exists = os.path.isfile(file_path)
        
        total_raw_hits = final_score + deducted
        clean_accuracy = (final_score / total_raw_hits * 100.0) if total_raw_hits > 0 else 0.0
        
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Final Score", "Deducted Hits", "Duration (sec)", "Clean Play %"])
            writer.writerow([time.ctime(), final_score, deducted, round(duration, 2), round(clean_accuracy, 1)])
            
        return round(clean_accuracy, 1)
    
    def clear_evaluation_log(self):
        file_path = "evaluation_log.csv"
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[INFO] Evaluation log '{file_path}' has been cleared.")
        else:
            print("[INFO] No log file found to clear.")

    def _check_retroactive_drop(self, current_time):
        """
        Secondary safety wrapper for drop detection.
        The homography layer is the primary ground-plane check; this retroactive
        analysis catches bounce-like chains and transient floor vibrations when
        the ball appears frozen or briefly lost after the primary logic did not
        classify the event as a drop.
        idea: use time deltas between recent hits to detect floor bounces. If the ball was lost
        Floor bounces produce rapidly shrinking intervals (restitution decay),
        while human juggles keep a steadier rhythm. Walks back recent hits
        that match a bounce pattern and subtracts them from the score.
        
        """
        recent_hits = [t for t in self.hit_timestamps if (current_time - t) <= 5.0]
        if not recent_hits: 
            return "DROP"

        # If ball was clearly going up when lost, no bounce correction needed
        death_velocity = self.last_velocity if self.last_velocity else 0
        if death_velocity < -200: 
            return "DROP"

        if len(recent_hits) == 1:
            if (current_time - recent_hits[0]) < 1.0 and death_velocity > -50:
                return "DROP_MINUS_1"
            return "DROP"

        invalid_hits = 0
        floor_chain_detected = False
        
        for i in range(len(recent_hits) - 1, 0, -1):
            delta_t = recent_hits[i] - recent_hits[i-1]
            
            # Too fast for a human kick — flag as floor bounce
            if delta_t < 0.35:
                invalid_hits += 1
                floor_chain_detected = True
                
            # Gray zone: only flag if interval is shrinking (restitution decay)
            elif delta_t < 0.60:
                if i > 1:
                    delta_t_older = recent_hits[i-1] - recent_hits[i-2]
                    if delta_t < delta_t_older - 0.05:
                        invalid_hits += 1
                        floor_chain_detected = True
                    else:
                        break
                else:
                    break
                    
            else:
                break

        # The first impact that started the bounce chain is also invalid
        if floor_chain_detected:
            invalid_hits += 1
            
        elif invalid_hits == 0 and len(recent_hits) > 0:
            time_since_last = current_time - recent_hits[-1]
            if time_since_last < 1.0 and death_velocity > -50: 
                invalid_hits = 1

        return f"DROP_MINUS_{invalid_hits}" if invalid_hits > 0 else "DROP"

    def _detect_impact(self, current_x, current_y, current_r, current_time, frame_width, frame_height, main_data=None):
        """Detects HIT or DROP using velocity-flip, boundary checks, and floor homography."""

        if len(self.side_history) < 2: 
            return None

        # Out of bounds check
        out_of_x = current_x < self.floor_x_start or current_x > self.floor_x_end

        if out_of_x:
            if self.is_first_kick:
                pass
            else:
                return "DROP"

        # Homography floor check: both cameras must agree the ball is on the ground plane
        if self.floor_finder and self.floor_finder.calibrated and main_data is not None and not self.is_first_kick:
            pixel_main = (main_data[0], main_data[1])
            pixel_side = (current_x, current_y)
            on_ground, distance, _, _ = self.floor_finder.is_ball_on_floor(
                pixel_main,
                pixel_side,
                epsilon=self.floor_epsilon_cm
            )
            if on_ground:
                print(f"[FLOOR] Ball on floor (world distance: {distance:.2f})")
                return "DROP"
        
        # Fallback: linear interpolation between left/right anchors
        if self.floor_finder is None or not self.floor_finder.calibrated:
            if frame_width > 1:
                clamped_x = max(0, min(current_x, frame_width - 1))
                floor_y_at_x = int(
                    self.floor_left_y
                    + (self.floor_right_y - self.floor_left_y) * (clamped_x / (frame_width - 1))
                )
                if current_y > floor_y_at_x and not self.is_first_kick:
                    return "DROP"
            
        # Velocity freeze detection (ball stopped moving / rolling on floor)
        current_vel = self.side_history[-1][2]
        
        if abs(current_vel) < self.jitter_thresh:
            current_vel = 0
            self.frozen_frames += 1
            
            if self.frozen_frames == 1:
                self.dying_start_time = current_time 
                
            if self.frozen_frames >= 30 and not self.is_first_kick:
                # If the frozen ball is low, it may be resting on the floor and
                # should use retroactive bounce analysis as a secondary wrapper.
                # If it is frozen in the upper half of the frame, treat it as a
                # standard drop because it is likely caught or stuck.
                if current_y is not None and current_y > (frame_height / 2):
                    return self._check_retroactive_drop(current_time)
                return "DROP"
        else:
            self.frozen_frames = 0

        # Inertia state machine
        min_hit_vel = CONFIG["logic"].get("min_hit_velocity", -1.0)

        if current_vel > 0:
            self.fall_counter += 1
            self.rise_counter = 0
            if self.fall_counter >= self.required_inertia_frames:
                self.is_falling = True

        elif current_vel <= min_hit_vel:
            self.rise_counter += 1
            self.fall_counter = 0
            
        else:
            pass

        # Hit detection (V-Flip): ball was falling, now rising for 2+ frames
        is_normal_hit = (self.rise_counter >= 2) and self.is_falling

        # First kick from floor: needs 3 rising frames + min 15px upward travel
        is_flick_up_hit = False
        if self.is_first_kick and (self.rise_counter >= 3):
            if len(self.side_history) >= 3:
                past_y = self.side_history[-3][0]
                if past_y is not None and (past_y - current_y) > 15:
                    is_flick_up_hit = True
        
        if is_normal_hit or is_flick_up_hit:
            if (current_time - self.last_hit_time) < self.cooldown:
                return None

            self.is_falling = False 
            self.fall_counter = 0
            self.rise_counter = 0
            self.is_first_kick = False

            return "HIT"

        return None

    def update(self, main_data, side_data, header_data=None, frame_width=640, frame_height=360):
        """Per-frame update: processes side-cam kicks, main-cam floor homography, header-cam hits, and drop conditions."""
        now = time.time()
        current_y = None
        current_x = None 
        current_r = None
        status = "Tracking"
        feedback = status
        event = None

        if header_data:
            self.header_history.append((header_data[2], now))

        if side_data:
            current_x, current_y, current_r = side_data
            self.lost_frames = 0
            
            if current_x <= self.EDGE_MARGIN_PX:
                self.last_exit_side = "LEFT"
            elif current_x >= (frame_width - self.EDGE_MARGIN_PX):
                self.last_exit_side = "RIGHT"
            else:
                self.last_exit_side = None

            velocity = self._calculate_velocity(current_y, now)
            self.last_velocity = velocity
            self.last_known_pos = (current_x, current_y, now)
            self.side_history.append((current_y, now, velocity))
            status = "Tracking Side"

        elif main_data:
            self.lost_frames = 0
            status = "Tracking Main Only"
            current_y = None

        else:
            self.lost_frames += 1
            if self.lost_frames == 1:
                self.dying_start_time = now
            
            status = "Lost Ball"
            current_y = None

        if self.countdown_active:
            elapsed = now - self.countdown_start_time
            remaining = 3 - int(elapsed)
            
            if remaining > 0:
                status = f"STARTING IN: {remaining}..."
                feedback = status
            else:
                self.countdown_active = False
                self.game_active = True
                print("[GAME] START JUGGLING!")
                status = "LIVE - Juggling!"
                feedback = status
                self.last_hit_time = now

        elif self.game_active:
            status = "LIVE - Juggling!"
            feedback = status

            # Lost-ball timeout (extended if ball was high and rising when lost)
            timeout_limit = 30
            if self.last_known_pos:
                last_y = self.last_known_pos[1]
                if last_y < 100 and self.last_velocity < 0:
                    timeout_limit = 90
                    
            if self.lost_frames >= timeout_limit and not side_data and not header_data and not self.is_first_kick:
                margin_x = frame_width * 0.25
                margin_top = frame_height * 0.20

                last_x = self.last_known_pos[0] if self.last_known_pos else (frame_width / 2)
                last_y = self.last_known_pos[1] if self.last_known_pos else (frame_height / 2)
                last_vel = self.last_velocity if hasattr(self, 'last_velocity') else 0

                # Exited horizontally through the left/right margins
                if last_x < margin_x or last_x > (frame_width - margin_x):
                    event = "DROP"
                # Ball flew high out of frame through the ceiling
                elif last_vel < 0 and last_y < margin_top:
                    event = "DROP"
                # Otherwise use retroactive bounce analysis as a secondary safety wrapper
                else:
                    event = self._check_retroactive_drop(now)
            
            elif current_y is not None:
                event = self._detect_impact(current_x, current_y, current_r, now, frame_width, frame_height, main_data=main_data)
                
            if event == "HIT":
                self.count += 1
                self.last_hit_time = now
                self.hit_timestamps.append(now)
                feedback = f"+1  [{self.count}]"
                print(f"[GAME] Kick detected! Total: {self.count}")
            
            elif event and event.startswith("DROP_MINUS_"):
                invalid_hits = int(event.split("_")[2])
                self.count = max(0, self.count - invalid_hits)

                duration = now - self.countdown_start_time
                clean_acc = self.save_evaluation_data(self.count, invalid_hits, duration)

                feedback = "DROP! GAME OVER."
                print(f"[GAME] Drop detected (removed {invalid_hits} floor bounces). Score: {self.count}")
                self.game_active = False 
                self._play_drop_sound()
                status = f"Game Over. Score: {self.count} | Clean: {clean_acc}% | Press T to restart."
            
            elif event == "DROP":
                duration = now - self.countdown_start_time
                clean_acc = self.save_evaluation_data(self.count, 0, duration)
                feedback = "DROP! GAME OVER."
                print(f"[GAME] Drop detected. Score: {self.count}")
                self.game_active = False
                self._play_drop_sound()
                status = f"Game Over. Score: {self.count} | Clean: {clean_acc}% | Press T to restart."

        else:
            if not self.floor_finder or not self.floor_finder.calibrated:
                status = "WAITING: Press 'F' to calibrate floor."
            else:
                status = "READY: Press 'T' to start."
            
            if feedback == "Tracking": 
                feedback = ""

        # Header detection via header-camera radius flip
        if header_data and self.header_baseline_r is not None and len(self.header_history) >= 2:
            status = "Header Tracking..."
            
            header_r = header_data[2]
            prev_header_r = self.header_history[-2][0]
            delta_r = header_r - prev_header_r
            
            if delta_r < -0.5: 
                self.header_shrink_counter += 1
                self.header_growth_counter = 0
            elif delta_r > 0.5: 
                self.header_growth_counter += 1
                self.header_shrink_counter = 0

            if self.header_shrink_counter >= 2:
                self.is_falling_from_header = True

            is_header_impact = self.is_falling_from_header and self.header_growth_counter >= 1

            if is_header_impact:
                if prev_header_r >= self.header_baseline_r:
                    if (now - self.last_hit_time) > 0.25:
                        self.count += 1
                        self.last_hit_time = now
                        self.hit_timestamps.append(now)
                        feedback = f"+1 HEAD  [{self.count}]"
                        print(f"[GAME] Header detected! Total: {self.count}")
                        
                        self.is_falling_from_header = False
                        self.header_shrink_counter = 0
                    
        if feedback == "Tracking":
            feedback = status

        return self.count, feedback

    def reset(self):
        self.count = 0
        self.side_history.clear()
        self.header_history.clear()
        self.is_falling = False
        self.fall_counter = 0
        self.rise_counter = 0
        self.is_first_kick = True
        print("[GAME] Counter reset.")
    
    def start_game(self):
        if not self.floor_finder or not self.floor_finder.calibrated:
            print("[WARNING] Calibrate floor first (press 'F').")
            return False
            
        if not self.countdown_active and not self.game_active:
            print("[GAME] Countdown started — get ready!")
            self.countdown_active = True
            self.countdown_start_time = time.time()
            self.reset()
            return True
        return False

    def force_drop(self, reason=""):
        if self.game_active:
            duration = time.time() - self.countdown_start_time
            clean_acc = self.save_evaluation_data(self.count, 0, duration)
            self.game_active = False
            print(f"[GAME] Forced drop ({reason}). Score: {self.count}")

