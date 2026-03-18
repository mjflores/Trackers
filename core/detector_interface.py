from abc import ABC, abstractmethod
import numpy as np


class DetectorInterface(ABC):
    """Contrato para cualquier detector."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR image
        Returns:
            np.ndarray shape (N, 6): [x1, y1, x2, y2, conf, cls]
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Pre-carga el modelo para evitar lag en el primer frame."""
        ...
