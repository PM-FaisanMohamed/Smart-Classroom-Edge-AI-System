# Models Directory

Place your trained YOLO model file (`yolo26s.pt`) in this folder.

## File Requirements:
- Filename: `yolo26s.pt` (or any `.pt` file such as `best.pt`).
- Model type: Ultralytics YOLO trained for person detection (Class 0: person).
- Target inference settings:
  - `conf = 0.33`
  - `iou = 0.95`
  - `imgsz = 640`

When running with Docker, this folder is mounted into `/app/models` inside the container so you can swap or update weights without rebuilding the Docker container.
