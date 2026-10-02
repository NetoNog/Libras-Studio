"""
Pipeline de Captura de Vídeo Assíncrono com Threading Desacoplado.
Mantém a taxa de quadros (FPS) da câmera travada na frequência máxima,
eliminando atrasos de buffer interno e garantindo latência zero para a IA.
"""

import sys
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class ThreadedCamera:
    """
    Captura de vídeo em thread dedicada em segundo plano com buffer de 1 frame.
    Evita gargalos de sincronismo entre captura e inferência de visão computacional.
    """

    def __init__(self,
                 camera_index: int = 0,
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 30,
                 use_dshow: bool = True):
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.use_dshow = use_dshow

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False
        self.running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        self._open_capture()

    def _open_capture(self):
        """Inicializa o dispositivo de captura de vídeo com backend otimizado."""
        if self.use_dshow and sys.platform.startswith("win"):
            print(f"[*] Conectando à câmera {self.camera_index} via DirectShow (Windows)...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("[AVISO] DirectShow não abriu. Tentando backend padrão...")
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera no índice {self.camera_index}.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Lê o primeiro frame síncrono para garantir prontidão
        self.ret, self.frame = self.cap.read()
        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[ThreadedCamera] Câmera pronta: {self.actual_w}x{self.actual_h} @ {self.target_fps} FPS.")

    def start(self) -> "ThreadedCamera":
        """Inicia a thread de captura contínua."""
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, name="WebcamCaptureThread", daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        """Loop contínuo de captura na thread secundária."""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retorna instantaneamente o frame mais fresco sem bloqueio."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def set_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Altera a resolução da webcam dinamicamente em tempo de execução."""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.target_width = width
                self.target_height = height
                self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[ThreadedCamera] Resolução atualizada para: {self.actual_w}x{self.actual_h}")
                return self.actual_w, self.actual_h
            return self.target_width, self.target_height

    def release(self):

        """Para a thread e libera o recurso da câmera."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[ThreadedCamera] Câmera liberada com sucesso.")
