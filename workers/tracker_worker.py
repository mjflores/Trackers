import threading
import queue
import numpy as np
from core.tracker_interface import TrackerInterface
from core.frame_bus         import FrameBus
from utils.fps_counter      import FPSCounter
from utils.logger           import get_logger

logger = get_logger(__name__)


class TrackerWorker(threading.Thread):
    """
    Consumer: recibe frames del FrameBus y corre su tracker.
    Cada instancia es independiente → paralelismo real.
    """

    def __init__(self,
                 tracker:  TrackerInterface,
                 frame_q:  queue.Queue,
                 result_q: queue.Queue,
                 fps_window: int = 30):
        super().__init__(name=f"TrackerWorker-{tracker.name}", daemon=True)
        self._tracker   = tracker
        self._frame_q   = frame_q
        self._result_q  = result_q
        self._fps       = FPSCounter(window_size=fps_window)

    @property
    def fps(self) -> "FPSCounter":
        return self._fps

    def run(self):
        self._fps.start()
        logger.info(f"{self.name} iniciado")

        while True:
            item = self._frame_q.get()

            if item is None:       # señal de fin
                self._result_q.put(None)
                break

            frame, dets, fid = item

            try:
                tracks = self._tracker.update(dets, frame)
            except Exception as e:
                logger.warning(f"{self.name} error en frame {fid}: {e}")
                tracks = np.empty((0, 5))

            self._fps.tick()
            self._result_q.put((self._tracker.name, frame, tracks, fid))

        logger.info(f"{self.name} finalizado | "
                    f"FPS global: {self._fps.fps_global:.2f}")
