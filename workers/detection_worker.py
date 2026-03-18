import threading
import numpy as np
from core.video_source     import VideoSource
from core.detector_interface import DetectorInterface
from core.frame_bus        import FrameBus
from utils.logger          import get_logger

logger = get_logger(__name__)


class DetectionWorker(threading.Thread):
    """
    Producer: lee frames de VideoSource, ejecuta YOLO
    y publica (frame, detections, frame_id) en el FrameBus.
    """

    def __init__(self,
                 source:   VideoSource,
                 detector: DetectorInterface,
                 bus:      FrameBus):
        super().__init__(name="DetectionWorker", daemon=True)
        self._source   = source
        self._detector = detector
        self._bus      = bus
        self._stop_evt = threading.Event()

    def run(self):
        logger.info("DetectionWorker iniciado")

        while not self._stop_evt.is_set():
            ok, frame, fid = self._source.read()

            if not ok:
                logger.info(f"Fin del video en frame {fid}")
                break

            dets = self._detector.detect(frame)
            self._bus.publish((frame, dets, fid))

        self._bus.stop_all()
        logger.info("DetectionWorker finalizado")

    def stop(self):
        self._stop_evt.set()
