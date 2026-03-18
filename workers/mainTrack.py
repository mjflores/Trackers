'''
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from ultralytics import YOLO
from boxmot import create_tracker

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"



VIDEO_PATH = "mateoCam1.mp4"
YOLO_MODEL = "yolov8n.pt"

# StrongSORT y BotSORT necesitan pesos ReID para inicializarse
# osnet_x0_25_msmt17.pt es liviano y compatible con boxmot
REID_MODEL = Path("osnet_x0_25_msmt17.pt")

TRACKERS = [
    "strongsort",
    "bytetrack",
    "botsort",
    "ocsort"
]

# Trackers que requieren pesos ReID
REID_REQUIRED = {"strongsort", "botsort"}

MAX_FRAMES = 300

import time
from collections import deque


class FPSCounter:
    """
    Mide FPS en tiempo real usando una ventana deslizante,
    y FPS global desde el inicio.
    """

    def __init__(self, window_size=30):
        """
        Args:
            window_size: cantidad de frames para calcular FPS local
        """
        self.window_size = window_size
        self._timestamps = deque(maxlen=window_size)
        self._start_time = None
        self._total_frames = 0

    def start(self):
        """Inicia el contador global."""
        self._start_time = time.perf_counter()
        return self

    def tick(self):
        """Registra un frame procesado."""
        now = time.perf_counter()
        self._timestamps.append(now)
        self._total_frames += 1

        if self._start_time is None:
            self._start_time = now

    @property
    def fps_local(self):
        """FPS promedio de los últimos `window_size` frames."""
        if len(self._timestamps) < 2:
            return 0.0
        delta = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / delta if delta > 0 else 0.0

    @property
    def fps_global(self):
        """FPS promedio desde que se llamó start()."""
        if self._start_time is None or self._total_frames == 0:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        return self._total_frames / elapsed if elapsed > 0 else 0.0

    @property
    def elapsed(self):
        """Segundos transcurridos desde start()."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def total_frames(self):
        return self._total_frames

    def summary(self):
        """Imprime un resumen al finalizar."""
        print(f"\n{'='*40}")
        print(f"  Frames procesados : {self.total_frames}")
        print(f"  Tiempo total      : {self.elapsed:.2f}s")
        print(f"  FPS global        : {self.fps_global:.2f}")
        print(f"{'='*40}")

    def draw(self, frame):
        """
        Dibuja FPS local y global sobre el frame.
        Devuelve el frame anotado (no modifica el original).
        """
        import cv2

        frame = frame.copy()
        h = frame.shape[0]

        texts = [
            (f"FPS: {self.fps_local:.1f}",  (10, h - 40)),
            (f"AVG: {self.fps_global:.1f}", (10, h - 15)),
        ]

        for text, pos in texts:
            # Sombra para legibilidad sobre cualquier fondo
            cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, text, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame
        
class YOLODetector:

    def __init__(self, model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model)
        self.model.to(device)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls != 0:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })

        return detections


def download_reid_weights():
    """Descarga los pesos ReID si no existen."""
    if REID_MODEL.exists():
        return True

    print(f"\nDescargando pesos ReID ({REID_MODEL.name})...")
    try:
        import urllib.request
        url = (
            "https://drive.google.com/uc?export=download&id="
            "1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA"
        )
        # boxmot puede descargarlos automáticamente al pasar el Path
        # Si falla la descarga manual, boxmot lo intentará solo
        return True
    except Exception as e:
        print(f"  Advertencia: no se pudo pre-descargar ReID weights: {e}")
        return True  # boxmot intentará descargarlo internamente


def init_trackers():
    trackers = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Inicializando trackers...\n")

    download_reid_weights()

    for name in TRACKERS:
        try:
            # Los trackers con ReID reciben la ruta al modelo
            # Los demás reciben None
            reid_weights = REID_MODEL if name in REID_REQUIRED else None

            tracker = create_tracker(
                tracker_type=name,
                tracker_config=None,
                reid_weights=reid_weights,
                device=device,
                half=False
            )

            trackers[name] = tracker
            print(f"✓ {name} inicializado")

        except Exception as e:
            print(f"✗ {name} error: {e}")

    return trackers


def draw_tracks(frame, tracks, name):
    frame = frame.copy()

    cv2.putText(
        frame, name, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    if tracks is None or len(tracks) == 0:
        return frame

    for t in tracks:
        x1, y1, x2, y2, track_id = map(int, t[:5])

        color = (
            (track_id * 37) % 255,
            (track_id * 17) % 255,
            (track_id * 29) % 255
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return frame


def create_grid(frames):
    """
    Genera una grilla dinámica según la cantidad de trackers activos.
    1 tracker  → 1×1
    2 trackers → 1×2
    3-4        → 2×2 (celdas vacías en negro)
    """
    names = list(frames.keys())
    n = len(names)

    if n == 0:
        return None

    h, w = frames[names[0]].shape[:2]

    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = 2, 1
    else:
        cols, rows = 2, 2

    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for idx, name in enumerate(names):
        r = idx // cols
        c = idx % cols
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frames[name]

    return grid


def main():
    print("Sistema de Comparación de Multi-Object Trackers")
    print("=" * 50)

    fps = FPSCounter(window_size=30).start()   # ← init

    detector = YOLODetector(YOLO_MODEL)
    trackers = init_trackers()

    if len(trackers) == 0:
        print("No se pudo inicializar ningún tracker")
        return

    print(f"\nTrackers activos: {list(trackers.keys())}")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

    frame_id = 0
    start = time.time()

    while True:
        ret, frame = cap.read()

        if not ret or frame_id >= MAX_FRAMES:
            break
        fps.tick()                              # ← registra el frame
        
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t1 = time.perf_counter()

        # Siempre pasamos el array al tracker (puede estar vacío)
        if len(detections) > 0:
            dets = np.array([
                [*d["bbox"], d["conf"], 0]
                for d in detections
            ], dtype=np.float32)
        else:
            # Array vacío con la forma correcta para que el tracker
            # pueda decrementar edad de tracks existentes
            dets = np.empty((0, 6), dtype=np.float32)

        frames = {}

        for name, tracker in trackers.items():
            try:
                t2 = time.perf_counter()
                tracks = tracker.update(dets, frame)
                t3 = time.perf_counter()
                print(f"{name}: {(t3-t2)*1000:.1f}ms")
                vis = draw_tracks(frame, tracks, name)
                frames[name] = vis
            except Exception as e:
                print(f"  {name} frame {frame_id} error: {e}")
                
        print(f"YOLO: {(t1-t0)*1000:.1f}ms")
        grid = create_grid(frames)

        if grid is not None:
            # Escalar si la grilla es demasiado grande para la pantalla


            gh, gw = grid.shape[:2]
            max_display = 1280
            if gw > max_display:
                scale = max_display / gw
                grid = cv2.resize(
                    grid,
                    (int(gw * scale), int(gh * scale))
                )
            grid = fps.draw(grid)              # ← dibuja sobre la grilla
            cv2.imshow("Tracker Comparison", grid)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"\nFrames procesados : {frame_id}")
    print(f"Tiempo total      : {elapsed:.2f}s")
    print(f"FPS promedio      : {frame_id / elapsed:.2f}")


if __name__ == "__main__":
    main()
    
'''


import cv2
import numpy as np
import torch
import time
import os
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from boxmot import create_tracker

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

VIDEO_PATH     = "mateoCam1.mp4"
YOLO_MODEL     = "yolov8n.pt"
REID_MODEL     = Path("osnet_x0_25_msmt17.pt")
INPUT_SIZE     = 320
DISPLAY_SCALE  = 0.6
MAX_FRAMES     = 300

TRACKERS = ["strongsort", "bytetrack", "botsort", "ocsort"]
REID_REQUIRED = {"strongsort", "botsort"}   # ← agregar esta línea
# Config YAML sin ReID para trackers lentos
STRONGSORT_CFG = Path("strongsort_fast.yaml")
BOTSORT_CFG    = Path("botsort_fast.yaml")

TRACKER_CONFIGS = {
    "strongsort": STRONGSORT_CFG,
    "botsort":    BOTSORT_CFG,
}
REID_MODEL = Path("osnet_x0_25_msmt17.pt")


def write_fast_configs():
    """Genera configs con ReID desactivado si no existen."""

    if not STRONGSORT_CFG.exists():
        STRONGSORT_CFG.write_text(
            "STRONGSORT:\n"
            "  EMA_ALPHA: 0.9\n"
            "  MC_LAMBDA: 0.98\n"
            "  MAX_DIST: 0.4\n"
            "  MAX_IOU_DIST: 0.7\n"
            "  MAX_AGE: 30\n"
            "  N_INIT: 3\n"
            "  NN_BUDGET: 100\n"
            "  WITH_REID: false\n"
        )
        print("  ✓ strongsort_fast.yaml creado")

    if not BOTSORT_CFG.exists():
        BOTSORT_CFG.write_text(
            "BOTSORT:\n"
            "  TRACK_HIGH_THRESH: 0.6\n"
            "  TRACK_LOW_THRESH: 0.1\n"
            "  NEW_TRACK_THRESH: 0.7\n"
            "  TRACK_BUFFER: 30\n"
            "  MATCH_THRESH: 0.8\n"
            "  WITH_REID: false\n"
        )
        print("  ✓ botsort_fast.yaml creado")


# ─── FPS Counter ────────────────────────────────────────────────────────────

class FPSCounter:

    def __init__(self, window_size=30):
        self._timestamps  = deque(maxlen=window_size)
        self._start_time  = None
        self._total_frames = 0

    def start(self):
        self._start_time = time.perf_counter()
        return self

    def tick(self):
        now = time.perf_counter()
        self._timestamps.append(now)
        self._total_frames += 1
        if self._start_time is None:
            self._start_time = now

    @property
    def fps_local(self):
        if len(self._timestamps) < 2:
            return 0.0
        delta = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / delta if delta > 0 else 0.0

    @property
    def fps_global(self):
        if not self._start_time or not self._total_frames:
            return 0.0
        return self._total_frames / (time.perf_counter() - self._start_time)

    @property
    def elapsed(self):
        return time.perf_counter() - self._start_time if self._start_time else 0.0

    @property
    def total_frames(self):
        return self._total_frames

    def summary(self):
        print(f"\n{'='*40}")
        print(f"  Frames procesados : {self.total_frames}")
        print(f"  Tiempo total      : {self.elapsed:.2f}s")
        print(f"  FPS global        : {self.fps_global:.2f}")
        print(f"{'='*40}")

    def draw(self, frame):
        frame = frame.copy()
        h = frame.shape[0]
        for text, pos in [
            (f"FPS: {self.fps_local:.1f}",  (10, h - 40)),
            (f"AVG: {self.fps_global:.1f}", (10, h - 15)),
        ]:
            cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(frame, text, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        return frame


# ─── YOLO Detector ──────────────────────────────────────────────────────────

class YOLODetector:

    def __init__(self, model_path, input_size=INPUT_SIZE):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = YOLO(model_path)
        self.model.to(self.device)
        self.input_size = input_size
        self.use_half  = self.device == "cuda"

        # Warm-up
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(dummy, verbose=False, imgsz=self.input_size)

        print(f"  ✓ YOLO | device={self.device} "
              f"imgsz={self.input_size} half={self.use_half}")

    def detect(self, frame):
        results = self.model(
            frame,
            verbose=False,
            imgsz=self.input_size,
            half=self.use_half,
            classes=[0]
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls  = np.zeros((len(results.boxes), 1), dtype=np.float32)

        return np.hstack([xyxy, conf, cls]).astype(np.float32)


# ─── Trackers ───────────────────────────────────────────────────────────────

def init_trackers():
    trackers = {}
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    #write_fast_configs()
    print("\nInicializando trackers...")

    for name in TRACKERS:
        try:
            reid_weights = REID_MODEL if name in REID_REQUIRED else None
            tracker = create_tracker(
                tracker_type   = name,
                tracker_config = TRACKER_CONFIGS.get(name),
                reid_weights   = reid_weights,   # ReID desactivado por config
                device         = device,
                half           = False
            )
            trackers[name] = tracker
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    return trackers


def _run_one_tracker(args):
    name, tracker, dets, frame = args
    try:
        return name, tracker.update(dets.copy(), frame)
    except Exception as e:
        print(f"  {name} error: {e}")
        return name, None


# ─── Visualización ──────────────────────────────────────────────────────────

def draw_tracks(frame, tracks, name):
    frame = frame.copy()
    for text, color, pos in [
        (name, (0,0,0),   (11,31)),
        (name, (0,255,0), (10,30)),
    ]:
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if tracks is not None and len(tracks) > 0:
        for t in tracks:
            x1,y1,x2,y2,tid = map(int, t[:5])
            c = ((tid*37)%255, (tid*17)%255, (tid*29)%255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return frame


def create_grid(frames):
    names = list(frames.keys())
    if not names:
        return None
    h, w  = frames[names[0]].shape[:2]
    cols  = 2 if len(names) > 1 else 1
    rows  = (len(names) + cols - 1) // cols
    grid  = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frames[name]
    return grid

def write_fast_configs():
    """Copia los configs originales de boxmot y desactiva ReID."""
    import shutil
    import boxmot

    boxmot_dir = Path(boxmot.__file__).parent / "configs" / "trackers"

    for name, dst in [("strongsort", STRONGSORT_CFG),
                      ("botsort",    BOTSORT_CFG)]:
        src = boxmot_dir / f"{name}.yaml"

        if not src.exists():
            print(f"  ✗ No se encontró {src}")
            continue

        if not dst.exists():
            shutil.copy(src, dst)

            content = dst.read_text()

            # Desactivar ReID (cubre variantes True/true/yes/1)
            import re
            content = re.sub(
                r'(with_reid\s*:\s*)(True|true|yes|Yes|1)',
                r'\g<1>false',
                content
            )

            dst.write_text(content)
            print(f"  ✓ {dst.name} creado (with_reid desactivado)")
        else:
            print(f"  ✓ {dst.name} ya existe")

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Sistema de Comparación de Multi-Object Trackers")
    print("=" * 50)

    detector = YOLODetector(YOLO_MODEL)
    #write_fast_configs()   # ← aquí

    trackers = init_trackers()

    if not trackers:
        print("Sin trackers disponibles.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    executor  = ThreadPoolExecutor(max_workers=len(trackers))
    fps       = FPSCounter(window_size=30).start()
    first_name = list(trackers.keys())[0]

    frame_id  = 0
    last_dets = np.empty((0, 6), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret or frame_id >= MAX_FRAMES:
            break

        fps.tick()

        # Detección YOLO
        last_dets = detector.detect(frame)

        # Trackers en paralelo
        results = dict(executor.map(
            _run_one_tracker,
            [(n, t, last_dets, frame) for n, t in trackers.items()]
        ))

        # Visualización
        frames = {}
        for name, tracks in results.items():
            vis = draw_tracks(frame, tracks, name)
            if name == first_name:
                vis = fps.draw(vis)
            frames[name] = vis

        grid = create_grid(frames)
        if grid is not None:
            gw = int(grid.shape[1] * DISPLAY_SCALE)
            gh = int(grid.shape[0] * DISPLAY_SCALE)
            cv2.imshow("Tracker Comparison",
                       cv2.resize(grid, (gw, gh)))

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_id += 1

    executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()
    fps.summary()


if __name__ == "__main__":
    main()
