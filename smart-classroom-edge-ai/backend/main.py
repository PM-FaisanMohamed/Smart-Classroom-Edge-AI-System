import os
import io
import time
import base64
import asyncio
import datetime
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from backend.config import current_config, SystemConfig, HysteresisConfig, StabilizationConfig, CleaningConfig, ACRule
from backend.inference import YOLOInferenceEngine
from backend.stabilization import OccupancyStabilizer
from backend.cleaning_detector import JanitorCleaningDetector
from backend.ac_simulation import ACSimulator

app = FastAPI(
    title="Smart Classroom Edge AI System",
    description="Edge AI inference, occupancy stabilization, janitor cleaning detection, and AC simulation service.",
    version="2.0.0"
)

# Enable CORS for cross-origin dashboard requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZE SYSTEM COMPONENTS ---
inference_engine = YOLOInferenceEngine(model_dir="models", model_name="yolo26s.pt")
stabilizer = OccupancyStabilizer(current_config.hysteresis, current_config.stabilization)
cleaning_detector = JanitorCleaningDetector(current_config.cleaning)
ac_simulator = ACSimulator(current_config.ac_rules)

# Global session timeline log
timeline_log: List[Dict[str, Any]] = []

def record_timeline_entry(occupancy: str, raw_count: int, smoothed_count: int, ac_on: bool, temp: Optional[float], activity: str):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "time": now_str,
        "occupancy": occupancy,
        "raw_count": raw_count,
        "smoothed_count": smoothed_count,
        "ac_on": ac_on,
        "temperature": temp,
        "activity": activity
    }
    timeline_log.append(entry)
    if len(timeline_log) > 200:
        timeline_log.pop(0)

# Record initial baseline state
record_timeline_entry("LOW", 0, 0, False, None, "NORMAL")

# --- REQUEST / RESPONSE MODELS ---
class PredictRequest(BaseModel):
    image: str  # Base64 data URL or raw base64 string

class ConfigUpdateRequest(BaseModel):
    hysteresis: Optional[HysteresisConfig] = None
    stabilization: Optional[StabilizationConfig] = None
    cleaning: Optional[CleaningConfig] = None
    ac_rules: Optional[Dict[str, ACRule]] = None

# --- HELPER FUNCTIONS ---
def process_frame_pipeline(frame_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Complete end-to-end processing pipeline:
    Inference -> Stabilization -> Cleaning Detection -> AC Simulation -> Telemetry update
    """
    now = time.time()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Inference Service (YOLO conf=0.33, iou=0.95, imgsz=640)
    inf_res = inference_engine.detect(frame_bgr)
    raw_count = inf_res["person_count"]
    detections = inf_res["detections"]
    confidence = inf_res["confidence_avg"]

    # 2. Occupancy Stabilization Logic (Rolling median + dual hysteresis + dwell timer)
    stab_res = stabilizer.process_frame(raw_count, timestamp=now)
    smoothed_count = stab_res["smoothed_count"]
    stable_occupancy = stab_res["stable_occupancy"]

    # 3. Janitor Cleaning Detection (Motion analysis + centroid displacement)
    clean_res = cleaning_detector.process_frame(frame_bgr, raw_count, detections, timestamp=now)
    activity_state = clean_res["activity_state"]

    # 4. AC Simulation (Driven purely by stable occupancy signal)
    ac_res = ac_simulator.update_occupancy_state(stable_occupancy, timestamp_str=now_str)

    # Record timeline entry
    record_timeline_entry(
        occupancy=stable_occupancy,
        raw_count=raw_count,
        smoothed_count=smoothed_count,
        ac_on=ac_res["ac_on"],
        temp=ac_res["temperature"],
        activity=activity_state
    )

    return get_current_state_snapshot(confidence, raw_count, smoothed_count, detections, clean_res, stab_res)


def get_current_state_snapshot(confidence: float = 1.0, raw_count: int = 0, smoothed_count: int = 0,
                               detections: List[Any] = None, clean_res: Dict[str, Any] = None, stab_res: Dict[str, Any] = None) -> Dict[str, Any]:
    # Reset occupancy to LOW if stream is idle (no video/webcam frame received for > 5 seconds)
    stabilizer.check_idle_timeout(max_idle_seconds=5.0)
    ac_state = ac_simulator.update_occupancy_state(stabilizer.current_stable_state, timestamp_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    current_occupancy = stabilizer.current_stable_state
    candidate_occupancy = stabilizer.candidate_state

    activity_state = clean_res["activity_state"] if clean_res else "NORMAL"
    is_cleaning = clean_res["is_cleaning"] if clean_res else False

    return {
        "occupancy": current_occupancy,
        "candidate_occupancy": candidate_occupancy,
        "raw_count": raw_count,
        "smoothed_count": smoothed_count,
        "confidence": confidence,
        "detections": detections or [],
        "activity_state": activity_state,
        "is_cleaning": is_cleaning,
        "motion_magnitude": clean_res.get("motion_magnitude", 0.0) if clean_res else 0.0,
        "ac_on": ac_state.get("ac_on", False),
        "temperature": ac_state.get("temperature"),
        "last_change": ac_state.get("last_change_timestamp", "—"),
        "last_change_timestamp": ac_state.get("last_change_timestamp", "—"),
        "total_ac_seconds": ac_state.get("total_ac_seconds", 0.0),
        "total_ac_runtime_seconds": ac_state.get("total_ac_seconds", 0.0),
        "total_ac_formatted": ac_state.get("total_ac_formatted", "00:00:00"),
        "dwell_elapsed_seconds": stab_res.get("dwell_elapsed_seconds", 0.0) if stab_res else 0.0,
        "dwell_progress": stab_res.get("dwell_progress", 0.0) if stab_res else 0.0,
        "timeline": timeline_log,
        "timeline_log": timeline_log
    }

# --- REST API ENDPOINTS ---

@app.get("/health")
def health_check():
    """GET /health -> service status, model info, and class labels"""
    ac_simulator.accrue_runtime()
    return {
        "status": "ok",
        "model_loaded": inference_engine.model_loaded,
        "model_path": inference_engine.loaded_model_path,
        "inference_params": {
            "conf": inference_engine.conf,
            "iou": inference_engine.iou,
            "imgsz": inference_engine.imgsz
        },
        "mode": "YOLO Edge Service" if inference_engine.model_loaded else "Mock Fallback Standby"
    }


@app.get("/state")
def get_state():
    """GET /state -> returns full live state snapshot for dashboards"""
    return get_current_state_snapshot()


@app.post("/predict")
def predict_frame(payload: PredictRequest):
    """POST /predict -> body: { "image": "<base64 JPEG>" } -> run pipeline and return snapshot"""
    try:
        image_str = payload.image
        if "," in image_str:
            image_str = image_str.split(",")[1]

        image_bytes = base64.b64decode(image_str)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        snapshot = process_frame_pipeline(frame_bgr)
        return {
            "label": snapshot["occupancy"],
            "raw_label": f"Raw: {snapshot['raw_count']} (Smoothed: {snapshot['smoothed_count']})",
            "confidence": snapshot["confidence"],
            "raw_count": snapshot["raw_count"],
            "smoothed_count": snapshot["smoothed_count"],
            "activity_state": snapshot["activity_state"],
            "is_cleaning": snapshot["is_cleaning"],
            "state": snapshot
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or inference error: {str(e)}")


@app.get("/config")
def get_config():
    """GET /config -> returns current system thresholds and rules"""
    return current_config.dict()


@app.post("/config")
def update_config(update: Dict[str, Any]):
    """POST /config -> updates thresholds, dwell times, and AC rules at runtime"""
    global current_config

    try:
        # Support both new structured format and legacy dashboard format
        if "LOW" in update or "MEDIUM" in update or "HIGH" in update:
            # Legacy format: { "LOW": { "ac_on": false, "temperature": 24 }, ... }
            new_ac_rules = {}
            for lvl in ["LOW", "MEDIUM", "HIGH"]:
                if lvl in update:
                    new_ac_rules[lvl] = ACRule(
                        ac_on=bool(update[lvl].get("ac_on", False)),
                        temperature=update[lvl].get("temperature") or update[lvl].get("temp")
                    )
            current_config.ac_rules = new_ac_rules
            ac_simulator.update_rules(new_ac_rules)

        if "hysteresis" in update:
            h_cfg = HysteresisConfig(**update["hysteresis"])
            current_config.hysteresis = h_cfg
            stabilizer.hysteresis = h_cfg

        if "stabilization" in update:
            s_cfg = StabilizationConfig(**update["stabilization"])
            current_config.stabilization = s_cfg
            stabilizer.stabilization = s_cfg

        if "cleaning" in update:
            c_cfg = CleaningConfig(**update["cleaning"])
            current_config.cleaning = c_cfg
            cleaning_detector.config = c_cfg

        return {"status": "success", "config": current_config.dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update config: {str(e)}")


@app.get("/events")
async def sse_events(request: Request):
    """GET /events -> Server-Sent Events (SSE) live telemetry stream"""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            snapshot = get_current_state_snapshot()
            data_json = JSONResponse(snapshot).body.decode("utf-8")
            yield f"data: {data_json}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- STATIC DASHBOARD SERVING ---
dashboard_dir = os.path.abspath("dashboard")
ac_dir = os.path.abspath("ac-simulation")

if os.path.exists(dashboard_dir):
    app.mount("/dashboard_assets", StaticFiles(directory=dashboard_dir), name="dashboard_assets")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    dash_index = os.path.join(dashboard_dir, "index.html")
    if os.path.exists(dash_index):
        with open(dash_index, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Smart Classroom Edge AI Dashboard</h1><p>Dashboard UI not found.</p>"

@app.get("/ac-simulation", response_class=HTMLResponse)
@app.get("/ac-simulation/", response_class=HTMLResponse)
def serve_ac_simulation():
    ac_index = os.path.join(ac_dir, "index.html")
    if not os.path.exists(ac_index):
        ac_index = os.path.join(dashboard_dir, "ac_simulation.html")
    if os.path.exists(ac_index):
        with open(ac_index, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AC Simulation UI</h1><p>AC Simulation file not found.</p>"
