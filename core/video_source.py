import cv2
import threading
import numpy as np
from typing import Optional, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoSource:
    """
    Singleton que captura frames una única vez y los distribuye.
    Patrón: Singleton + Thread-safe
    """
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, source, max_frames: int = 300):
        if self._initialized:
            return

        self.source     = source
        self.max_frames = max_frames
        self._cap       = None
        self._frame_id  = 0
        self._lock      = threading.Lock()
        self._initialized = True

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error(f"No se pudo abrir la fuente: {self.source}")
            return False
        logger.info(f"Fuente abierta: {self.source}")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Retorna (success, frame, frame_id). Thread-safe."""
        with self._lock:
            if self._frame_id >= self.max_frames:
                return False, None, self._frame_id

            ret, frame = self._cap.read()
            if ret:
                self._frame_id += 1

            return ret, frame, self._frame_id

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def release(self):
        if self._cap:
            self._cap.release()
        VideoSource._instance = None
        logger.info("VideoSource liberado")
