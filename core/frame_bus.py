import threading
import queue
import numpy as np
from typing import Dict
from utils.logger import get_logger

logger = get_logger(__name__)

_SENTINEL = None  # señal de fin


class FrameBus:
    """
    Distribuye cada frame a todos los trackers suscritos.
    Patrón: Observer (publisher / subscriber)
    Cada suscriptor tiene su propia queue → no se bloquean entre sí.
    """

    def __init__(self, maxsize: int = 4):
        self._queues:  Dict[str, queue.Queue] = {}
        self._maxsize  = maxsize
        self._lock     = threading.Lock()

    def subscribe(self, name: str) -> queue.Queue:
        """Registra un tracker y devuelve su queue."""
        with self._lock:
            q = queue.Queue(maxsize=self._maxsize)
            self._queues[name] = q
            logger.debug(f"Suscriptor registrado: {name}")
            return q

    def publish(self, item) -> None:
        """
        Envía un item a todas las queues.
        Si una queue está llena, descarta el frame más antiguo (no bloquea).
        """
        with self._lock:
            queues = list(self._queues.values())

        for q in queues:
            if q.full():
                try:
                    q.get_nowait()   # descarta frame viejo
                except queue.Empty:
                    pass
            q.put_nowait(item)

    def stop_all(self) -> None:
        """Envía señal de fin a todos los suscriptores."""
        self.publish(_SENTINEL)
        logger.info("FrameBus: señal de parada enviada")
