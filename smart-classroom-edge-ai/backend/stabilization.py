import time
from collections import deque
from typing import Dict, Any, Tuple
from backend.config import HysteresisConfig, StabilizationConfig

try:
    import numpy as np
    def get_median(lst):
        return int(np.median(lst)) if lst else 0
except ImportError:
    def get_median(lst):
        if not lst: return 0
        s = sorted(lst)
        n = len(s)
        return s[n // 2]

class OccupancyStabilizer:
    """
    Occupancy Stabilization Pipeline:
    1. Rolling Smoothing: Median count over rolling time window (~5-10s).
    2. Hysteresis Thresholding: Dual thresholds per state boundary to eliminate flickering.
    3. Dwell/Debounce Timer: Sustained candidate state required for configurable dwell seconds before committing.
    """
    def __init__(self, hysteresis: HysteresisConfig, stabilization: StabilizationConfig):
        self.hysteresis = hysteresis
        self.stabilization = stabilization

        # Buffer of (timestamp, raw_count) tuples
        self.rolling_buffer = deque()

        # State machine variables
        self.current_stable_state = "LOW"
        self.candidate_state = "LOW"
        self.candidate_start_time: float = 0.0
        self.last_frame_timestamp: float = time.time()

    def update_config(self, hysteresis: HysteresisConfig, stabilization: StabilizationConfig):
        self.hysteresis = hysteresis
        self.stabilization = stabilization

    def check_idle_timeout(self, max_idle_seconds: float = 5.0):
        """If no frames arrive for max_idle_seconds (e.g. video/webcam stopped), reset state to LOW."""
        if time.time() - self.last_frame_timestamp > max_idle_seconds:
            if self.current_stable_state != "LOW":
                print(f"[STABILIZATION] Idle stream timeout ({max_idle_seconds}s without frames). Resetting occupancy to LOW.")
                self.current_stable_state = "LOW"
                self.candidate_state = "LOW"
                self.candidate_start_time = 0.0
                self.rolling_buffer.clear()

    def process_frame(self, raw_count: int, timestamp: float = None) -> Dict[str, Any]:
        """
        Process new frame's raw person count through stabilization pipeline.
        """
        if timestamp is None:
            timestamp = time.time()
        self.last_frame_timestamp = timestamp

        # 1. Update rolling window buffer
        self.rolling_buffer.append((timestamp, raw_count))

        # Evict entries older than rolling window_seconds
        cutoff_time = timestamp - self.stabilization.window_seconds
        while self.rolling_buffer and self.rolling_buffer[0][0] < cutoff_time:
            self.rolling_buffer.popleft()

        # 2. Compute median smoothed count
        counts = [item[1] for item in self.rolling_buffer]
        smoothed_count = get_median(counts)

        # 3. Determine candidate state using Hysteresis logic
        candidate = self._evaluate_hysteresis(self.current_stable_state, smoothed_count)

        # Immediate walkthrough reset: If raw count drops below threshold during an upgrade candidate, abort candidate immediately
        if self.current_stable_state == "LOW" and raw_count < self.hysteresis.low_to_medium:
            candidate = "LOW"
        elif self.current_stable_state == "MEDIUM" and candidate == "HIGH" and raw_count < self.hysteresis.medium_to_high:
            candidate = "MEDIUM"

        # 4. Apply Dwell / Debounce Timer
        state_changed = False
        dwell_elapsed = 0.0

        if candidate != self.current_stable_state:
            if candidate != self.candidate_state:
                # New candidate state detected, start dwell timer
                self.candidate_state = candidate
                self.candidate_start_time = timestamp
                dwell_elapsed = 0.0
            else:
                # Sustained candidate state
                dwell_elapsed = timestamp - self.candidate_start_time
                if dwell_elapsed >= self.stabilization.dwell_seconds:
                    # Dwell requirement met: commit state transition!
                    print(f"[STABILIZATION] State Transition Committed: {self.current_stable_state} -> {candidate} (Raw: {raw_count}, Smoothed: {smoothed_count}, Dwell: {dwell_elapsed:.1f}s)")
                    self.current_stable_state = candidate
                    state_changed = True
                    self.candidate_start_time = 0.0
        else:
            # Signal matches current stable state, reset candidate timer
            self.candidate_state = self.current_stable_state
            self.candidate_start_time = 0.0
            dwell_elapsed = 0.0

        dwell_progress = min(1.0, dwell_elapsed / self.stabilization.dwell_seconds) if self.stabilization.dwell_seconds > 0 else 1.0

        return {
            "raw_count": raw_count,
            "smoothed_count": smoothed_count,
            "stable_occupancy": self.current_stable_state,
            "candidate_state": self.candidate_state,
            "dwell_elapsed_seconds": round(dwell_elapsed, 1),
            "dwell_progress": round(dwell_progress, 2),
            "state_changed": state_changed
        }

    def _evaluate_hysteresis(self, current_state: str, smoothed_count: int) -> str:
        """
        Evaluate hysteresis rules:
        - LOW -> MEDIUM when smoothed_count >= low_to_medium
        - MEDIUM -> LOW when smoothed_count <= medium_to_low
        - MEDIUM -> HIGH when smoothed_count >= medium_to_high
        - HIGH -> MEDIUM when smoothed_count <= high_to_medium
        """
        cfg = self.hysteresis

        if current_state == "LOW":
            if smoothed_count >= cfg.low_to_medium:
                return "MEDIUM"
            return "LOW"

        elif current_state == "MEDIUM":
            if smoothed_count <= cfg.medium_to_low:
                return "LOW"
            elif smoothed_count >= cfg.medium_to_high:
                return "HIGH"
            return "MEDIUM"

        elif current_state == "HIGH":
            if smoothed_count <= cfg.high_to_medium:
                return "MEDIUM"
            return "HIGH"

        return "LOW"
