'''
                                   Recomendación para vigilancia

Usa OcSORT como tracker principal.

    ByteTrack  → rápido pero pierde IDs en oclusiones (no apto para vigilancia)
    OcSORT     → mismo FPS que ByteTrack + maneja oclusiones correctamente
    StrongSORT → mejor precisión pero inutilizable sin GPU
    BotSORT    → igual de lento, necesita GPU
'''

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

from config import SystemConfig
from core.pipeline import TrackingPipeline
from trackers.strongsort_tracker import StrongSortTracker
from trackers.bytetrack_tracker  import ByteTrackTracker
from trackers.botsort_tracker    import BotSortTracker
from trackers.ocsort_tracker     import OcSortTracker
from utils.logger                import get_logger

logger = get_logger(__name__)

TRACKER_REGISTRY = {
    "strongsort": StrongSortTracker,
    "bytetrack":  ByteTrackTracker,
    "botsort":    BotSortTracker,
    "ocsort":     OcSortTracker,
}


def main():
    cfg = SystemConfig()

    trackers = []
    for name in cfg.trackers_enabled:
        cls = TRACKER_REGISTRY.get(name)
        if cls is None:
            logger.warning(f"Tracker desconocido: {name}")
            continue
        try:
            trackers.append(cls(cfg.tracker))
            logger.info(f"✓ {name} inicializado")
        except Exception as e:
            logger.error(f"✗ {name}: {e}")

    if not trackers:
        logger.error("Sin trackers disponibles.")
        return

    pipeline = TrackingPipeline(cfg, trackers)
    pipeline.run()


if __name__ == "__main__":
    main()
