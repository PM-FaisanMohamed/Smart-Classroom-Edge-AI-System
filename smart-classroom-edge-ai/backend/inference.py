import os
import time
from typing import Dict, List, Any, Tuple, Optional

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

class YOLOInferenceEngine:
    """
    YOLO Inference Service using Ultralytics YOLO.
    Configured for classroom person detection with conf=0.33, iou=0.95, imgsz=640.
    """
    def __init__(self, model_dir: str = "models", model_name: str = "yolo26s.pt"):
        self.model_dir = os.path.abspath(model_dir)
        self.requested_model_name = model_name
        self.model = None
        self.model_loaded = False
        self.loaded_model_path = ""
        self.conf = 0.33
        self.iou = 0.95
        self.imgsz = 640
        self.person_class_id = 0  # COCO class 0 is 'person'

        self._load_model()

    def _find_model_file(self) -> Optional[str]:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        target_path = os.path.join(self.model_dir, self.requested_model_name)
        if os.path.exists(target_path):
            return target_path

        for file in os.listdir(self.model_dir):
            if file.endswith(".pt") or file.endswith(".onnx") or file.endswith(".engine"):
                return os.path.join(self.model_dir, file)

        return self.requested_model_name

    def _load_model(self):
        try:
            import torch
            # Optimize CPU threads to prevent CPU thrashing and lag
            if not torch.cuda.is_available():
                num_threads = min(4, max(1, (os.cpu_count() or 4) // 2))
                torch.set_num_threads(num_threads)
                print(f"[INFO] Configured PyTorch CPU threads to {num_threads} for optimal FPS.")
        except Exception:
            pass

        try:
            from ultralytics import YOLO
            model_path = self._find_model_file()
            print(f"[INFO] Initializing YOLO model from: {model_path}")
            self.model = YOLO(model_path)
            self.model_loaded = True
            self.loaded_model_path = model_path
            print(f"[SUCCESS] Loaded YOLO model successfully: {model_path}")
        except Exception as e:
            print(f"[WARNING] Could not load YOLO model ({e}). Operating in Standby/Fallback Mode.")
            self.model = None
            self.model_loaded = False

    def set_parameters(self, conf: float = 0.33, iou: float = 0.95, imgsz: int = 640):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, image_np: Any) -> Dict[str, Any]:
        if image_np is None:
            return {"person_count": 0, "detections": [], "confidence_avg": 0.0}

        if not self.model_loaded or self.model is None or not HAS_CV2:
            return self._mock_detect(image_np)

        try:
            # Pre-resize high resolution frames to max width 960 for fast CPU pre-processing
            h, w = image_np.shape[:2]
            if w > 960:
                scale = 960.0 / w
                new_w, new_h = 960, int(h * scale)
                image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            results = self.model.predict(
                source=image_np,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=[self.person_class_id],
                verbose=False
            )

            person_count = 0
            detections = []
            confidences = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id == self.person_class_id:
                        person_count += 1
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()
                        confidences.append(conf)
                        
                        cx = (xyxy[0] + xyxy[2]) / 2.0
                        cy = (xyxy[1] + xyxy[3]) / 2.0

                        detections.append({
                            "bbox": [round(c, 1) for c in xyxy],
                            "centroid": [round(cx, 1), round(cy, 1)],
                            "confidence": round(conf, 3),
                            "class": "person"
                        })

            avg_conf = float(np.mean(confidences)) if confidences else 1.0

            w, h = 640, 400
            if hasattr(image_np, 'shape') and len(image_np.shape) >= 2:
                h, w = image_np.shape[:2]

            return {
                "person_count": person_count,
                "detections": detections,
                "confidence_avg": round(avg_conf, 3),
                "frame_size": [w, h]
            }

        except Exception as e:
            print(f"[ERROR] Error during YOLO inference: {e}")
            return self._mock_detect(image_np)

    def _mock_detect(self, image_np: Any) -> Dict[str, Any]:
        w, h = 640, 400
        mock_count = 0
        if HAS_CV2 and hasattr(image_np, 'shape') and image_np.size > 0:
            h, w = image_np.shape[:2]
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
            variance = float(np.var(gray))
            mock_count = int((variance / 500.0)) % 5

        return {
            "person_count": mock_count,
            "detections": [],
            "confidence_avg": 0.85,
            "frame_size": [w, h],
            "mock": True
        }
