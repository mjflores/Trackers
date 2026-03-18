# utils/fps_counter.py
import time
import cv2
import numpy as np
from collections import deque
from typing import Optional


class FPSCounter:

    def __init__(self, window_size: int = 30):
        self._timestamps   = deque(maxlen=window_size)
        self._start_time:  Optional[float] = None
        self._total_frames = 0

    def start(self) -> "FPSCounter":
        self._start_time = time.perf_counter()
        return self

    def tick(self) -> None:
        now = time.perf_counter()
        self._timestamps.append(now)
        self._total_frames += 1
        if self._start_time is None:
            self._start_time = now

    @property
    def fps_local(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        delta = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / delta if delta > 0 else 0.0

    @property
    def fps_global(self) -> float:
        if not self._start_time or not self._total_frames:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        return self._total_frames / elapsed if elapsed > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start_time if self._start_time else 0.0

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def summary(self, prefix: str = "") -> None:
        label = f"[{prefix}] " if prefix else ""
        print(
            f"{label}Frames: {self.total_frames} | "
            f"Tiempo: {self.elapsed:.2f}s | "
            f"FPS global: {self.fps_global:.2f}"
        )

    def draw(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()
        h     = frame.shape[0]
        for text, pos in [
            (f"FPS: {self.fps_local:.1f}",  (10, h - 40)),
            (f"AVG: {self.fps_global:.1f}", (10, h - 15)),
        ]:
            cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, text, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame
