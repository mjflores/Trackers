from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch


@dataclass
class DetectorConfig:
    model_path: str  = "/home/mjflores/Descargas/Trackers/assets/models_yolo/yolov8n.pt"
    input_size: int  = 320
    confidence: float = 0.3
    classes:    List  = field(default_factory=lambda: [0])
    device:     str   = "cuda" if torch.cuda.is_available() else "cpu"
    half:       bool  = False  # True solo con GPU


@dataclass
class TrackerConfig:
    reid_model: Path  = Path("/home/mjflores/Descargas/Trackers/assets/reid_weights/osnet_x0_25_msmt17.pt")
    device:     str   = "cuda" if torch.cuda.is_available() else "cpu"
    half:       bool  = False


@dataclass
class VideoConfig:
    source:        str = "/home/mjflores/Descargas/Trackers/assets/videos/mateoCam1.mp4"  # 0 para webcam
    max_frames:    int = 300
    display_scale: float = 0.6


@dataclass
class SystemConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker:  TrackerConfig  = field(default_factory=TrackerConfig)
    video:    VideoConfig    = field(default_factory=VideoConfig)
    trackers_enabled: List[str] = field(
        #default_factory=lambda: ["strongsort", "bytetrack", "botsort", "ocsort"]
        #default_factory=lambda: ["ocsort"]
        default_factory=lambda: ["bytetrack", "ocsort"]
    )
    fps_window: int = 30
