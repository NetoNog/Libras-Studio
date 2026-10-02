"""
Módulo de Rastreamento de Pontos-Chave Faciais para Ponto de Articulação em Libras.
Extrai as coordenadas normalizadas do queixo, boca, nariz e testa para desambiguação
espacial de sinais lexicais (ex.: 'ÁGUA' no queixo, 'DESCULPA' na lateral da mandíbula).
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class FaceKeypoints:
    """Pontos anatômicos da face relevantes para a fonologia de Libras."""
    chin: np.ndarray       # Landmark 152 (Queixo) [x, y, z]
    mouth: np.ndarray      # Landmark 13 (Boca) [x, y, z]
    nose: np.ndarray       # Landmark 1 (Nariz) [x, y, z]
    forehead: np.ndarray   # Landmark 10 (Testa) [x, y, z]
    face_size: float       # Distância entre testa e queixo


class FaceTracker:
    """
    Rastreador leve de referências faciais para determinar o Ponto de Articulação.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    MODEL_PATH = "face_landmarker.task"

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Inicializa o FaceLandmarker do MediaPipe Tasks."""
        self._ensure_model_file()
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python

            base_options = mp_python.BaseOptions(model_asset_path=self.MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[FaceTracker] Não foi possível inicializar FaceLandmarker: {e}")

    def _ensure_model_file(self):
        """Baixa o modelo se não existir localmente."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"[FaceTracker] Baixando modelo facial para '{self.MODEL_PATH}'...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("[FaceTracker] Modelo facial baixado.")

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceKeypoints]:
        """
        Processa o frame e retorna os pontos-chave faciais (ou None se não houver rosto).
        """
        if self.detector is None:
            return None

        try:
            import mediapipe as mp
            frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_img)


            if not result.face_landmarks or len(result.face_landmarks) == 0:
                return None

            lms = result.face_landmarks[0]

            chin = np.array([lms[152].x, lms[152].y, lms[152].z], dtype=np.float32)
            mouth = np.array([lms[13].x, lms[13].y, lms[13].z], dtype=np.float32)
            nose = np.array([lms[1].x, lms[1].y, lms[1].z], dtype=np.float32)
            forehead = np.array([lms[10].x, lms[10].y, lms[10].z], dtype=np.float32)

            face_size = float(np.linalg.norm(forehead[:2] - chin[:2]))
            if face_size < 1e-4:
                face_size = 0.25

            return FaceKeypoints(
                chin=chin,
                mouth=mouth,
                nose=nose,
                forehead=forehead,
                face_size=face_size
            )
        except Exception:
            return None

    def draw_landmarks(self, frame: np.ndarray, face: FaceKeypoints, width: int, height: int):
        """Desenha pontos discretos sutis para indicar o ponto de articulação rastreado."""
        for pt, color in [
            (face.chin, (0, 220, 255)),      # Queixo (Ciano)
            (face.mouth, (60, 230, 120)),    # Boca (Verde)
            (face.forehead, (245, 180, 40))  # Testa (Dourado)
        ]:
            px = int(pt[0] * width)
            py = int(pt[1] * height)
            cv2.circle(frame, (px, py), 4, (10, 10, 15), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 3, color, -1, cv2.LINE_AA)
