import time
import numpy as np
import csv
import os
import threading
from collections import deque
from config import CONFIG
from config_utils import load_floor_epsilon, save_floor_epsilon

try:
    import pygame
    pygame.mixer.init()
    _PYGAME_AVAILABLE = True
except Exception:
    _PYGAME_AVAILABLE = False
    print("[WARNING] pygame not available – drop sound will be silent.")

class JugglingCounter:
    """
    Core game logic: aggregates dual-camera data, tracks ball state via
    a sliding window, detects kicks/drops using velocity-flip analysis,
    and manages calibration parameters.
    """

    def __init__(self, history_len=150, floor_finder=None): 
        self.floor_finder = floor_finder
        
        self.count = 0    
        self.last_hit_time = 0
        self.cooldown = 0.05
        
        self.is_falling = False
        self.fall_counter = 0
        self.rise_counter = 0
        
        self.jitter_thresh = CONFIG["logic"].get("jitter_threshold", 20)
        self.floor_tol = CONFIG["logic"].get("floor_tolerance", 30)
        self.header_ratio = CONFIG["logic"].get("header_height_ratio", 0.35)
        self.required_inertia_frames = CONFIG["logic"].get("inertia_frames", 2)

        self.floor_left_y = CONFIG["logic"].get("floor_left_y", 350)
        self.floor_right_y = CONFIG["logic"].get("floor_right_y", 350)
        self.floor_x_start = 0 
        self.floor_x_end = CONFIG["camera"]["width"]
        
        self.radius_factor = CONFIG["logic"].get("radius_correction_factor", 0.5)
        self.floor_epsilon_cm = load_floor_epsilon(default_value=5.0)
        print(f"[INFO] Floor epsilon loaded: {self.floor_epsilon_cm:.2f} cm")
        
        self.history_len = history_len  
        self.side_history = deque(maxlen=history_len)   # (y, timestamp, velocity)
        self.top_history = deque(maxlen=history_len)    # (r, timestamp)
        
        self.baseline = {
            "floor_y": None,   
            "floor_x": None,   
            "floor_r": None,
            "is_set": False
        }

        self.last_known_pos = None
        self.last_velocity = 0
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = 30

        self.game_active = False
        self.countdown_active = False
        self.countdown_start_time = 0

        self.baseline_head_r = None
        self.is_rising_to_head = False
        self.rise_counter = 0
        self.is_first_kick = True
        self.top_growth_counter = 0
        self.top_shrink_counter = 0
        self.is_growing = False

        self.lost_frames = 0
        self.frozen_frames = 0
        self.dying_start_time = 0
        self.hit_timestamps = []

        self.EDGE_MARGIN_PX = 30
        self.last_exit_side = None

    def set_baseline(self, top_data, side_data):
        """Stores ball position and radius from side camera as the ground-truth reference."""
        if side_data:
            self.baseline["floor_y"] = side_data[1] 
            self.baseline["floor_x"] = side_data[0]
            ref_r = side_data[2]
            self.baseline["floor_r"] = ref_r
            
            self.baseline["is_set"] = True
            print(f"[CAL] Baseline set: radius={ref_r}, L_anchor={self.floor_left_y}, R_anchor={self.floor_right_y}")
            return True
            
        return False
    
    def set_head_baseline(self, top_data):
        """Sets the minimum top-camera radius that qualifies as a header."""
        if top_data:
            self.baseline_head_r = top_data[2]
            print(f"[CAL] Head baseline set: min radius = {self.baseline_head_r}")
            return True
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
        Retroactive drop analysis using time deltas between hits.
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

    def _detect_impact(self, current_x, current_y, current_r, current_time, frame_width, frame_height, top_data=None):
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
        if self.floor_finder and self.floor_finder.calibrated and top_data is not None and not self.is_first_kick:
            pixel_top = (top_data[0], top_data[1])
            pixel_side = (current_x, current_y)
            on_ground, distance, _, _ = self.floor_finder.is_ball_on_floor(
                pixel_top,
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
                return self._check_retroactive_drop(current_time)
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

    def update(self, top_data, side_data, frame_width=640, frame_height=360):
        """Per-frame update: processes side-cam kicks, top-cam headers, and drop conditions."""
        now = time.time()
        current_y = None
        current_x = None 
        current_r = None
        status = "Tracking"
        feedback = status
        event = None

        if top_data:
            self.top_history.append((top_data[2], now))

        # Radius gate: reject detections that are too big/small vs. the calibrated radius
        if side_data and self.baseline["is_set"]:
            _, _, test_r = side_data
            base_r = self.baseline["floor_r"]
            if test_r > (base_r * 1.5) or test_r < (base_r * 0.5):
                side_data = None

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

        elif top_data:
            self.lost_frames = 0
            status = "Tracking Top Only"
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
                    
            if self.lost_frames >= timeout_limit and not side_data and not top_data and not self.is_first_kick:
                margin_x = frame_width * 0.25
                last_x = self.last_known_pos[0] if self.last_known_pos else (frame_width / 2)
                
                if last_x < margin_x or last_x > (frame_width - margin_x):
                    event = "DROP"
                else:
                    event = self._check_retroactive_drop(now)
            
            elif current_y is not None:
                event = self._detect_impact(current_x, current_y, current_r, now, frame_width, frame_height, top_data=top_data)
                
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
            if not self.baseline["is_set"]:
                status = "WAITING: Press 'S' to calibrate."
            else:
                status = "READY: Press 'T' to start."
            
            if feedback == "Tracking": 
                feedback = ""

        # Header detection via top-camera radius flip
        if top_data and self.baseline_head_r is not None and len(self.top_history) >= 2:
            status = "Header Tracking..."
            
            top_r = top_data[2]
            prev_top_r = self.top_history[-2][0]
            delta_r = top_r - prev_top_r
            
            if delta_r < -0.5: 
                self.top_shrink_counter += 1
                self.top_growth_counter = 0
            elif delta_r > 0.5: 
                self.top_growth_counter += 1
                self.top_shrink_counter = 0

            if self.top_shrink_counter >= 2:
                self.is_falling_from_top = True

            is_header_impact = self.is_falling_from_top and self.top_growth_counter >= 1

            if is_header_impact:
                if prev_top_r >= self.baseline_head_r:
                    if (now - self.last_hit_time) > 0.25:
                        self.count += 1
                        self.last_hit_time = now
                        self.hit_timestamps.append(now)
                        feedback = f"+1 HEAD  [{self.count}]"
                        print(f"[GAME] Header detected! Total: {self.count}")
                        
                        self.is_falling_from_top = False
                        self.top_shrink_counter = 0
                    
        if feedback == "Tracking":
            feedback = status

        return self.count, feedback

    def reset(self):
        self.count = 0
        self.side_history.clear()
        self.top_history.clear()
        self.is_falling = False
        self.fall_counter = 0
        self.rise_counter = 0
        self.is_first_kick = True
        print("[GAME] Counter reset.")
    
    def start_game(self):
        if not self.baseline["is_set"]:
            print("[WARNING] Calibrate floor first (press 'S').")
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

