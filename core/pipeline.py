import queue
import threading
import cv2
from typing import List
from config               import SystemConfig
from core.video_source    import VideoSource
from core.frame_bus       import FrameBus
from core.tracker_interface import TrackerInterface
from detectors.yolo_detector import YOLODetector
from workers.detection_worker import DetectionWorker
from workers.tracker_worker   import TrackerWorker
from utils.visualizer         import Visualizer
from utils.fps_counter        import FPSCounter
from utils.logger             import get_logger

logger = get_logger(__name__)


class TrackingPipeline:
    """
    Orchestrator: conecta todas las piezas.
    VideoSource → DetectionWorker → FrameBus → TrackerWorkers → Display
    """

    def __init__(self, cfg: SystemConfig, trackers: List[TrackerInterface]):
        self._cfg      = cfg
        self._trackers = trackers
        self._source   = VideoSource(cfg.video.source, cfg.video.max_frames)
        self._detector = YOLODetector(cfg.detector)
        self._bus      = FrameBus(maxsize=4)
        self._result_q = queue.Queue()
        self._fps_main = FPSCounter(cfg.fps_window)

        # Un worker por tracker
        self._workers: List[TrackerWorker] = [
            TrackerWorker(
                tracker   = t,
                frame_q   = self._bus.subscribe(t.name),
                result_q  = self._result_q,
                fps_window = cfg.fps_window
            )
            for t in trackers
        ]
        self._det_worker = DetectionWorker(
            self._source, self._detector, self._bus
        )

    def run(self):
        if not self._source.open():
            return

        self._detector.warmup()
        self._fps_main.start()

        # Arrancar workers
        for w in self._workers:
            w.start()
        self._det_worker.start()

        active     = len(self._workers)
        frames_vis = {}

        while active > 0:
            item = self._result_q.get()

            if item is None:
                active -= 1
                continue

            name, frame, tracks, fid = item
            worker = next(w for w in self._workers if w.name == f"TrackerWorker-{name}")

            frames_vis[name] = Visualizer.draw_tracks(
                frame, tracks, name, fps=worker.fps
            )
            self._fps_main.tick()

            # Mostrar cuando tengamos resultado de todos
            if len(frames_vis) == len(self._workers):
                grid = Visualizer.build_grid(
                    frames_vis, scale=self._cfg.video.display_scale
                )
                if grid is not None:
                    cv2.imshow("Tracker Comparison", grid)

                if cv2.waitKey(1) & 0xFF == 27:
                    self._det_worker.stop()
                    break

        cv2.destroyAllWindows()
        self._source.release()
        self._fps_main.summary()

        # FPS por tracker
        logger.info("\n── FPS por tracker ──")
        for w in self._workers:
            w.fps.summary(prefix=w.name)
