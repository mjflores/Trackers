from abc import ABC, abstractmethod
import numpy as np


class TrackerInterface(ABC):
    """Contrato para cualquier tracker."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def update(self,
               detections: np.ndarray,
               frame: np.ndarray) -> np.ndarray:
        """
        Args:
            detections: (N,6) [x1,y1,x2,y2,conf,cls]
            frame:      BGR image
        Returns:
            np.ndarray (M,5): [x1,y1,x2,y2,track_id]
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reinicia el estado interno del tracker."""
        ...
