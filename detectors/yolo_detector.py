# detectors/yolo_detector.py
import numpy as np
import torch
from ultralytics import YOLO
from core.detector_interface import DetectorInterface
from config import DetectorConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class YOLODetector(DetectorInterface):

    def __init__(self, cfg: DetectorConfig):
        self._cfg    = cfg
        self._model  = YOLO(cfg.model_path)
        self._model.to(cfg.device)
        logger.info(
            f"YOLODetector | device={cfg.device} "
            f"imgsz={cfg.input_size} half={cfg.half}"
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        results = self._model(
            frame,
            verbose=False,
            imgsz=self._cfg.input_size,
            half=self._cfg.half,
            classes=self._cfg.classes
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls  = results.boxes.cls.cpu().numpy().reshape(-1, 1)

        return np.hstack([xyxy, conf, cls]).astype(np.float32)

    def warmup(self) -> None:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(3):
            self._model(
                dummy,
                verbose=False,
                imgsz=self._cfg.input_size
            )
        logger.info("YOLODetector warmup completado")
