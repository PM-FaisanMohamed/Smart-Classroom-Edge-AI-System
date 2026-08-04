# Smart Classroom Edge AI System

## 👥 Project Team (Group 8)
- **Product Owner:** Kumaran Vithushan (`kumaran-vithushan`)
- **Project Manager:** PM. Faisan Mohamed (`PM-FaisanMohamed`)
- **Data Scientists:** 
  - E.M. Dulanjana Hirushan (`hirushan26`)
  - R.D. Thakshila Kumari (`KThakshila`)
  - M.M. Mariyam (`MohamedMihlarMaryam`)
- **App Developers:** 
  - Janana Methsara (`jananamethsara`)
  - Danush Kavindaka (`DK0747`)
  - Sivaraja Pratheep (`sivarajapratheep175`)

---

An Edge AI classroom occupancy monitoring, activity classification, and automated building management system (BMS) prototype.

The system processes video streams locally on edge hardware using a **YOLO model (`yolo26s.pt`)**, applies **occupancy stabilization filtering (rolling median + hysteresis + dwell timers + walkthrough abort + idle stream decay)**, detects **janitor cleaning activity**, and dynamically automates classroom Air Conditioning (AC) state and temperature settings without any cloud inference dependency.

---

## 🏗️ Architecture & Data Flow

```text
  [ Video Input Stream ] (Webcam / File / Camera Feed)
           │
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Edge Inference Service (YOLO yolo26s)                │
  │    conf=0.33, iou=0.95, imgsz=640                       │
  │    -> Raw Person Instance Detection Count               │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. Occupancy Stabilization Pipeline                     │
  │    • Rolling Window Median (4.0s smoothing)             │
  │    • Dual Hysteresis Thresholds (prevents flicker)       │
  │    • Walkthrough Immediate Abort (resets transient spikes)│
  │    • Dwell / Debounce Timer (5.0s sustained candidate)  │
  │    • Stream Idle Timeout (Auto-reset to LOW after 5s)   │
  │    -> Stable Occupancy Level (LOW / MEDIUM / HIGH)     │
  └────────────────────────┬────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
  ┌────────────────────────┐  ┌─────────────────────────────┐
  │ 3. Janitor Cleaning    │  │ 4. AC Simulation Engine     │
  │    Activity Detector   │  │    LOW    -> AC OFF           │
  │    (Motion & Centroid  │  │    MEDIUM -> AC ON @ 24°C      │
  │     Displacement)      │  │    HIGH   -> AC ON @ 20°C      │
  └────────┬───────────────┘  └────────────┬────────────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 5. FastAPI Service Layer & REST/SSE Server              │
  │    Endpoints: /predict, /state, /health, /config, /events│
  └────────────────────────┬────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
  ┌────────────────────────┐  ┌─────────────────────────────┐
  │ 6. Occupancy Console   │  │ 7. Split AC Unit Simulator  │
  │    (dashboard/index.html) │  │    (dashboard/ac_sim.html) │
  └────────────────────────┘  └─────────────────────────────┘
```

---

## 🔑 Key Features & Technical Deliverables

### 1. YOLO Inference Service
- Powered by `ultralytics` YOLO (`yolo26s.pt`).
- Target inference settings: `conf=0.33`, `iou=0.95`, `imgsz=640`, COCO Class 0 ("person").
- Optimized PyTorch CPU threading (`torch.set_num_threads(4)`) for smooth, real-time FPS without CPU contention.
- Operates 100% locally on the edge container with zero cloud dependencies.

### 2. Occupancy Stabilization Pipeline (Anti-Flicker & Walkthrough Protection)
Raw detection counts are noisy due to occlusion, motion, or momentary detection drops. This pipeline ensures smooth state transitions:
- **Rolling Median Smoothing**: Maintained over a rolling time window (`4.0s`) to filter temporary spikes or drops.
- **Dual Hysteresis Thresholds**:
  - `LOW` → `MEDIUM` when smoothed count $\ge 3$
  - `MEDIUM` → `LOW` when smoothed count $\le 2$
  - `MEDIUM` → `HIGH` when smoothed count $\ge 10$
  - `HIGH` → `MEDIUM` when smoothed count $\le 9$
- **Walkthrough Immediate Abort**: If `raw_count` drops below the threshold (e.g., a 3rd person enters for 2s and leaves), the candidate timer aborts immediately so the state never flips after they leave.
- **Dwell / Debounce Timer**: A candidate state change must be sustained for a configurable duration (default: `5.0s`) before it is committed.
- **Stream Idle Timeout**: If no video feed is actively streaming for $>5.0\text{s}$, occupancy automatically decays back to `LOW` (AC OFF).

### 3. Janitor Cleaning Detection
- Heuristic activity classifier running on top of the video stream.
- **Rule**: `person_count == 1` AND sustained high frame motion magnitude / centroid displacement over a rolling window $\rightarrow$ Classifies activity as `"CLEANING"` (distinct from a static/seated single student `"NORMAL"`).

### 4. Software AC Simulation Engine
- Driven purely by the *stable* occupancy signal:
  - `LOW`: AC OFF
  - `MEDIUM`: AC ON @ 24°C
  - `HIGH`: AC ON @ 20°C
- Continuously tracks: current AC state (`ON`/`OFF`), target temperature, timestamp of last state change, and cumulative total AC running time (`HH:MM:SS`).
- Thresholds and temperature settings are configurable at runtime via `POST /config`.

### 5. Dual Web Dashboards
- **Occupancy Console (`http://localhost:5000/`)**: Displays live webcam/video stream, telemetry stats, janitor activity status, interactive strip chart recorder, transition change log, and threshold configuration controls.
- **AC Simulation UI (`http://localhost:5000/ac-simulation`)**: Animated SVG split AC unit with glowing power LED, digital temperature display, spinning fans, animated airflow streams, and ambient room tinting.

---

## 📁 Repository Structure

```text
smart-classroom-edge-ai/
├── .gitignore
├── Dockerfile
├── README.md
├── docker-compose.yml
├── requirements.txt
├── test_pipeline.py
├── ac-simulation/
│   └── index.html
├── backend/
│   ├── __init__.py
│   ├── ac_simulation.py
│   ├── cleaning_detector.py
│   ├── config.py
│   ├── inference.py
│   ├── main.py
│   └── stabilization.py
├── dashboard/
│   ├── ac_simulation.html
│   └── index.html
├── models/
│   ├── README.md
│   └── yolo26s.pt
└── videos/
    └── .gitkeep
```

---

## 🚀 Quick Start (Docker)

### Prerequisite: Place Model Weights
Drop your trained model weights (`yolo26s.pt`) into the `models/` directory:
```text
smart-classroom-edge-ai/
  └── models/
      └── yolo26s.pt
```
*(When running with Docker, `./models` is mounted as a volume into `/app/models`, so you can update or swap `.pt` model weights at runtime without rebuilding the Docker image).*

### 1. Build and Run with Docker Compose (Recommended)
```bash
docker compose up --build -d
```
*(The image uses CPU-only PyTorch wheels to keep the container lightweight at **~1.2 GB**).*

### 2. Run with Docker CLI
```bash
# Build Docker image
docker build -t smart-classroom-edge .

# Run container with mounted models directory
docker run -d -p 5000:5000 -v $(pwd)/models:/app/models --name smart-classroom-edge smart-classroom-edge
```

### Accessing the Web Interfaces:
- **Occupancy Dashboard**: [http://localhost:5000](http://localhost:5000)
- **AC Simulation Unit**: [http://localhost:5000/ac-simulation](http://localhost:5000/ac-simulation)

---

## 🛠️ Local Python Setup (Without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch FastAPI server
uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload
```

---

## 🧪 Running Unit Tests

Run the complete pipeline verification suite (Inference, Stabilization, Cleaning Detector, AC Simulator):

```bash
python test_pipeline.py
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check, model status, & inference params |
| `GET` | `/state` | Full live state snapshot (occupancy, AC state, runtime, activity) |
| `POST` | `/predict` | Body: `{ "image": "<base64 JPEG>" }`. Runs inference & pipeline update |
| `GET` | `/events` | Server-Sent Events (SSE) live telemetry stream |
| `GET` | `/config` | Get current hysteresis, stabilization, & AC temperature rules |
| `POST` | `/config` | Update hysteresis thresholds, dwell timer, & AC rules at runtime |

---

## ✅ Acceptance Checklist Verification

- [x] **Single-Command Docker Execution**: Runs complete app (inference + stabilization + AC logic + FastAPI + dashboards).
- [x] **Occupancy Stabilization**: Anti-flicker verified with rolling median window (`4.0s`), hysteresis dual thresholds (3/2 & 10/9), walkthrough abort, and dwell debounce timer (`5.0s`).
- [x] **AC Simulation Rules & Accrual**: Correct LOW/MEDIUM/HIGH mapping (OFF / 24°C / 20°C) with continuous wall-clock runtime accrual.
- [x] **Janitor Cleaning Activity**: Flagged distinctly from normal single-occupant occupancy via motion differencing & centroid displacement.
- [x] **Dashboard UI Alignment**: Preserves target HTML layout & aesthetic design specs while wired to live backend streams.
- [x] **Self-Contained Edge Execution**: No external cloud API calls required.
