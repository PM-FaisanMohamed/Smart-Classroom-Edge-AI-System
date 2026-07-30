import sys
import os
import time
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath("."))

from backend.config import SystemConfig
from backend.inference import YOLOInferenceEngine
from backend.stabilization import OccupancyStabilizer
from backend.cleaning_detector import JanitorCleaningDetector
from backend.ac_simulation import ACSimulator

def run_tests():
    print("=== STARTING EDGE AI SYSTEM UNIT TESTS ===")
    cfg = SystemConfig()

    # 1. Test Inference Engine
    print("\n--- 1. Testing Inference Engine ---")
    engine = YOLOInferenceEngine(model_dir="models")
    blank_frame = np.zeros((400, 640, 3), dtype=np.uint8)
    res = engine.detect(blank_frame)
    print(f"Mock Detection Result: person_count={res['person_count']}, conf={res['confidence_avg']}")
    assert "person_count" in res, "Missing person_count in detection result"

    # 2. Test Stabilization Pipeline
    print("\n--- 2. Testing Occupancy Stabilization Pipeline ---")
    assert cfg.hysteresis.medium_to_low == 2, "Default medium_to_low should be 2"
    assert cfg.hysteresis.high_to_medium == 9, "Default high_to_medium should be 9"
    
    stabilizer = OccupancyStabilizer(cfg.hysteresis, cfg.stabilization)
    
    # Test low noise (walkthrough single frame spike should NOT change state from LOW to MEDIUM immediately)
    now = time.time()
    for i in range(5):
        st = stabilizer.process_frame(0, timestamp=now + i * 0.2)
    assert st["stable_occupancy"] == "LOW", "State should remain LOW"

    # Single spike frame (5 people for 1 frame)
    st = stabilizer.process_frame(5, timestamp=now + 1.2)
    print(f"After 1 spike frame (5 people): Smoothed={st['smoothed_count']}, Stable State={st['stable_occupancy']}")
    assert st["stable_occupancy"] == "LOW", "Single frame spike should be suppressed by median filter!"

    # Sustained 5 people (should change candidate to MEDIUM/HIGH and commit after dwell timer)
    t = now + 2.0
    for i in range(30):
        t += 0.2
        st = stabilizer.process_frame(5, timestamp=t)

    print(f"After sustained 5 people over 6 seconds: Smoothed={st['smoothed_count']}, Stable State={st['stable_occupancy']}")
    assert st["stable_occupancy"] == "MEDIUM", f"State should have committed to MEDIUM after dwell time (got {st['stable_occupancy']})"

    # Test stream idle timeout (no frames for > 5s resets state back to LOW)
    stabilizer.last_frame_timestamp = time.time() - 6.0
    stabilizer.check_idle_timeout(max_idle_seconds=5.0)
    assert stabilizer.current_stable_state == "LOW", "Idle stream timeout should reset occupancy to LOW"
    print("Idle stream timeout successfully verified: occupancy state reset to LOW.")

    # 3. Test Janitor Cleaning Detection
    print("\n--- 3. Testing Janitor Cleaning Detection ---")
    cleaner = JanitorCleaningDetector(cfg.cleaning)
    
    # Static frame with 1 person
    frame1 = np.ones((400, 640, 3), dtype=np.uint8) * 100
    det1 = [{"bbox": [100, 100, 200, 300], "centroid": [150.0, 200.0]}]
    res_clean1 = cleaner.process_frame(frame1, 1, det1, timestamp=now)
    assert res_clean1["activity_state"] == "NORMAL", "Static person should be classified as NORMAL"

    # High motion frame (differencing + displacement)
    frame2 = np.ones((400, 640, 3), dtype=np.uint8) * 200
    det2 = [{"bbox": [300, 100, 400, 300], "centroid": [350.0, 200.0]}]
    res_clean2 = cleaner.process_frame(frame2, 1, det2, timestamp=now + 1.0)
    print(f"Cleaning detector result with motion: activity={res_clean2['activity_state']}, is_cleaning={res_clean2['is_cleaning']}")
    assert res_clean2["activity_state"] == "CLEANING", "High motion with 1 person should be classified as CLEANING"

    # 4. Test AC Simulation
    print("\n--- 4. Testing AC Simulation ---")
    ac = ACSimulator(cfg.ac_rules)
    
    ac_state1 = ac.update_occupancy_state("LOW")
    print(f"LOW Occupancy AC State: AC ON={ac_state1['ac_on']}, Temp={ac_state1['temperature']}")
    assert ac_state1["ac_on"] == False, "LOW occupancy should turn AC OFF"

    ac_state2 = ac.update_occupancy_state("MEDIUM")
    print(f"MEDIUM Occupancy AC State: AC ON={ac_state2['ac_on']}, Temp={ac_state2['temperature']}°C")
    assert ac_state2["ac_on"] == True and ac_state2["temperature"] == 24.0, "MEDIUM occupancy should turn AC ON at 24°C"

    ac_state3 = ac.update_occupancy_state("HIGH")
    print(f"HIGH Occupancy AC State: AC ON={ac_state3['ac_on']}, Temp={ac_state3['temperature']}°C")
    assert ac_state3["ac_on"] == True and ac_state3["temperature"] == 20.0, "HIGH occupancy should turn AC ON at 20°C"

    print("\n=== ALL UNIT TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
