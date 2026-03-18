# Object Tracking System

## Instalación
```bash
python3.10 -m venv venvTrack310
source venvTrack310/bin/activate
pip install -r requirements.txt
```

## Assets requeridos (descargar manualmente)

Crear la estructura:
```
assets/
├── models/
│   └── yolov8n.pt            # https://github.com/ultralytics/assets/releases
├── reid_weights/
│   └── osnet_x0_25_msmt17.pt # descargado automáticamente por boxmot
└── videos/
    └── tu_video.mp4
```

## Uso
```bash
python3 main.py
```
