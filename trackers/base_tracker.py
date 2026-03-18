# trackers/base_tracker.py
import numpy as np
from core.tracker_interface import TrackerInterface
from config import TrackerConfig


class BaseBoxmotTracker(TrackerInterface):
    _tracker_type: str = ""
    _needs_reid:   bool = False

    def __init__(self, cfg: TrackerConfig):
        from boxmot import create_tracker
        self._cfg     = cfg
        self._tracker = create_tracker(
            tracker_type   = self._tracker_type,
            tracker_config = None,
            reid_weights   = cfg.reid_model if self._needs_reid else None,
            device         = cfg.device,
            half           = cfg.half
        )

    @property
    def name(self) -> str:
        return self._tracker_type

    def update(self,
               detections: np.ndarray,
               frame:      np.ndarray) -> np.ndarray:
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 6), dtype=np.float32)
        result = self._tracker.update(detections.copy(), frame)
        return result if result is not None else np.empty((0, 5))

    def reset(self) -> None:
        self.__init__(self._cfg)
