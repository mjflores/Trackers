import cv2
import numpy as np
from typing import Dict, Optional
from utils.fps_counter import FPSCounter


class Visualizer:
    """Factory de frames visualizados + grid compositor."""

    @staticmethod
    def draw_tracks(frame:     np.ndarray,
                    tracks:    np.ndarray,
                    name:      str,
                    fps:       Optional[FPSCounter] = None) -> np.ndarray:
        out = frame.copy()

        # Nombre del tracker
        cv2.putText(out, name, (11,31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.putText(out, name, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                x1,y1,x2,y2,tid = map(int, t[:5])
                color = ((tid*37)%255, (tid*17)%255, (tid*29)%255)
                cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
                cv2.putText(out, f"ID:{tid}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if fps is not None:
            out = fps.draw(out)

        return out

    @staticmethod
    def build_grid(frames: Dict[str, np.ndarray],
                   scale:  float = 1.0) -> Optional[np.ndarray]:
        if not frames:
            return None

        names = list(frames.keys())
        h, w  = frames[names[0]].shape[:2]
        cols  = 2 if len(names) > 1 else 1
        rows  = (len(names) + cols - 1) // cols

        grid = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
        for i, name in enumerate(names):
            r, c = divmod(i, cols)
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = frames[name]

        if scale != 1.0:
            grid = cv2.resize(
                grid,
                (int(w*cols*scale), int(h*rows*scale))
            )
        return grid
