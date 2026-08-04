import time
import datetime
from typing import Dict, Any, Optional
from backend.config import ACRule

def get_local_now_str() -> str:
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

class ACSimulator:
    """
    Software Air Conditioner (AC) Actuator Simulator.
    Drives AC state purely from the stable occupancy signal:
    - LOW -> AC OFF
    - MEDIUM -> AC ON @ 24°C
    - HIGH -> AC ON @ 20°C
    Accurately accrues cumulative AC running time.
    """
    def __init__(self, ac_rules: Dict[str, ACRule]):
        self.ac_rules = ac_rules
        self.ac_on: bool = False
        self.temperature: Optional[float] = None
        self.last_change_timestamp: str = get_local_now_str()
        self.total_ac_seconds: float = 0.0
        self.last_time_check: float = time.time()

    def update_rules(self, new_rules: Dict[str, ACRule]):
        self.ac_rules = new_rules

    def accrue_runtime(self, now: float = None):
        """Accrue elapsed wall-clock seconds to total AC running time if AC is currently ON."""
        if now is None:
            now = time.time()
        elapsed = now - self.last_time_check
        self.last_time_check = now
        if self.ac_on:
            self.total_ac_seconds += max(0.0, elapsed)

    def update_occupancy_state(self, stable_occupancy: str, timestamp_str: str = None) -> Dict[str, Any]:
        """
        Update AC state according to current stable occupancy signal.
        """
        now = time.time()
        self.accrue_runtime(now)

        rule = self.ac_rules.get(stable_occupancy, ACRule(ac_on=False, temperature=None))
        target_ac_on = rule.ac_on
        target_temp = rule.temperature if target_ac_on else None

        changed = False
        if target_ac_on != self.ac_on or target_temp != self.temperature:
            self.ac_on = target_ac_on
            self.temperature = target_temp
            self.last_change_timestamp = timestamp_str or get_local_now_str()
            changed = True
            print(f"[AC SIMULATOR] State Changed -> AC {'ON' if self.ac_on else 'OFF'}, Temp: {self.temperature}°C at {self.last_change_timestamp}")

        secs = int(self.total_ac_seconds)
        formatted_runtime = str(datetime.timedelta(seconds=secs))

        return {
            "ac_on": self.ac_on,
            "temperature": self.temperature,
            "last_change_timestamp": self.last_change_timestamp,
            "total_ac_seconds": round(self.total_ac_seconds, 1),
            "total_ac_formatted": formatted_runtime,
            "state_changed": changed
        }

    def get_state(self) -> Dict[str, Any]:
        self.accrue_runtime()
        secs = int(self.total_ac_seconds)
        return {
            "ac_on": self.ac_on,
            "temperature": self.temperature,
            "last_change_timestamp": self.last_change_timestamp,
            "total_ac_seconds": round(self.total_ac_seconds, 1),
            "total_ac_formatted": str(datetime.timedelta(seconds=secs))
        }
