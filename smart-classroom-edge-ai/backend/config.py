import os
from typing import Dict, Optional, Any

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    class BaseModel:
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)
        def dict(self):
            res = {}
            for k, v in self.__dict__.items():
                if isinstance(v, BaseModel):
                    res[k] = v.dict()
                elif isinstance(v, dict):
                    res[k] = {dk: (dv.dict() if isinstance(dv, BaseModel) else dv) for dk, dv in v.items()}
                else:
                    res[k] = v
            return res

class HysteresisConfig(BaseModel):
    low_to_medium: int = 3      # smoothed count >= 3 -> candidate MEDIUM
    medium_to_low: int = 2      # smoothed count <= 2 -> candidate LOW
    medium_to_high: int = 10    # smoothed count >= 10 -> candidate HIGH
    high_to_medium: int = 9     # smoothed count <= 9 -> candidate MEDIUM

class StabilizationConfig(BaseModel):
    window_seconds: float = 4.0   # Rolling window for median smoothing (3-5s)
    dwell_seconds: float = 5.0    # Minimum sustained duration before state commit (5s)

class CleaningConfig(BaseModel):
    motion_threshold: float = 0.03       # Frame differencing active motion ratio threshold
    displacement_threshold: float = 50.0  # Centroid pixel displacement threshold
    sustained_seconds: float = 3.0       # Duration requirement for cleaning classification

class ACRule(BaseModel):
    ac_on: bool
    temperature: Optional[float] = None

class SystemConfig(BaseModel):
    confidence_threshold: float = 0.33
    iou_threshold: float = 0.95
    imgsz: int = 640
    hysteresis: HysteresisConfig = HysteresisConfig()
    stabilization: StabilizationConfig = StabilizationConfig()
    cleaning: CleaningConfig = CleaningConfig()
    ac_rules: Dict[str, ACRule] = {
        "LOW": ACRule(ac_on=False, temperature=None),
        "MEDIUM": ACRule(ac_on=True, temperature=24.0),
        "HIGH": ACRule(ac_on=True, temperature=20.0)
    }

# Global config instance
current_config = SystemConfig()
