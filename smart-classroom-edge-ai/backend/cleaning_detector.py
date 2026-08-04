import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from backend.config import CleaningConfig

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

class JanitorCleaningDetector:
    """
    Janitor Cleaning Detection Service:
    Analyzes motion magnitude (frame differencing) and person centroid displacement.
    Heuristic: person_count == 1 AND sustained high motion/displacement over rolling window
    -> Classifies activity as "CLEANING", distinct from seated/static single student ("NORMAL").
    """
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.prev_gray: Optional[Any] = None

        # Rolling history of motion metrics: (timestamp, motion_ratio, centroid_disp)
        self.history = deque()
        self.last_centroid: Optional[Tuple[float, float]] = None

    def update_config(self, config: CleaningConfig):
        self.config = config

    def process_frame(self, frame_bgr: Any, person_count: int, detections: List[Dict[str, Any]], timestamp: float = None) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = time.time()

        motion_ratio = 0.0
        centroid_disp = 0.0

        if HAS_CV2 and frame_bgr is not None and hasattr(frame_bgr, 'size') and frame_bgr.size > 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
                frame_diff = cv2.absdiff(self.prev_gray, gray)
                _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(thresh)
                total_pixels = gray.shape[0] * gray.shape[1]
                motion_ratio = float(motion_pixels / total_pixels)

            self.prev_gray = gray

        # Centroid displacement tracking for single person
        if person_count == 1 and detections:
            curr_centroid = detections[0].get("centroid")
            if curr_centroid and self.last_centroid:
                dx = curr_centroid[0] - self.last_centroid[0]
                dy = curr_centroid[1] - self.last_centroid[1]
                centroid_disp = float((dx**2 + dy**2)**0.5)
            if curr_centroid:
                self.last_centroid = (curr_centroid[0], curr_centroid[1])
        else:
            self.last_centroid = None

        # Store metric in rolling window
        self.history.append((timestamp, motion_ratio, centroid_disp))

        # Evict metrics older than sustained_seconds
        cutoff = timestamp - self.config.sustained_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        # Evaluate Heuristic
        motions = [item[1] for item in self.history]
        disps = [item[2] for item in self.history]
        avg_motion = float(sum(motions)/len(motions)) if motions else 0.0
        avg_disp = float(sum(disps)/len(disps)) if disps else 0.0

        is_cleaning = False
        if person_count == 1:
            if avg_motion >= self.config.motion_threshold or avg_disp >= self.config.displacement_threshold:
                is_cleaning = True

        activity_state = "CLEANING" if is_cleaning else "NORMAL"

        return {
            "activity_state": activity_state,
            "is_cleaning": is_cleaning,
            "motion_magnitude": round(avg_motion, 4),
            "centroid_displacement": round(avg_disp, 1),
            "person_count": person_count
        }
