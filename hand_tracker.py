"""
Módulo de Rastreamento de Mão e Extração de Landmarks com MediaPipe.
Suporta de forma transparente tanto a API legada (mp.solutions.hands)
quanto a moderna API Tasks (mediapipe.tasks.vision.HandLandmarker) para
garantir compatibilidade universal em Python 3.10, 3.11, 3.12 e 3.13+.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from landmark_filter import OneEuroFilter

# Conexões anatômicas dos 21 pontos-chave da mão
HAND_CONNECTIONS = [
    # Polegar
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Indicador
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Médio
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Anelar
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Mínimo
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

FINGERTIP_IDS = [4, 8, 12, 16, 20]


@dataclass
class HandDetection:
    """Estrutura uniforme de dados de detecção da mão."""
    landmarks_norm: np.ndarray        # Shape (21, 3): x, y, z normalizados [0.0, 1.0]
    landmarks_pixel: np.ndarray       # Shape (21, 2): x, y em coordenadas de pixel do frame
    handedness: str                   # 'Left' ou 'Right'
    confidence: float                 # Confiança de detecção [0.0, 1.0]


class HandTracker:
    """
    Rastreador de landmarks da mão utilizando MediaPipe.
    Detecta automaticamente o backend disponível e normaliza a saída.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

    def __init__(self,
                 max_hands: int = 1,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6,
                 model_path: str = "hand_landmarker.task"):
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = model_path
        self.backend = None
        self.filters = [OneEuroFilter(min_cutoff=1.0, beta=0.08) for _ in range(max(2, max_hands))]

        self._init_detector()

    def _init_detector(self):
        """Inicializa o MediaPipe tentando a API Solutions ou Tasks."""
        import mediapipe as mp

        # 1. Verifica se a API clássica mp.solutions está disponível
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands
                self.detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.backend = "solutions"
                return
            except Exception as e:
                print(f"[HandTracker] Falha ao inicializar mp.solutions: {e}. Alternando para Tasks API.")

        # 2. Alternativa moderna: MediaPipe Tasks API (Python 3.12/3.13+)
        self._ensure_model_file()
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python

            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.backend = "tasks"
        except Exception as e:
            raise RuntimeError(f"Não foi possível inicializar o MediaPipe HandLandmarker: {e}")

    def _ensure_model_file(self):
        """Garante a existência do arquivo de modelo local da Tasks API."""
        if not os.path.exists(self.model_path):
            print(f"[HandTracker] Baixando modelo MediaPipe HandLandmarker para '{self.model_path}'...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            print("[HandTracker] Modelo baixado com sucesso.")

    def process(self, frame_bgr: np.ndarray) -> List[HandDetection]:
        """
        Processa um frame BGR e retorna a lista de mãos detectadas (vazia se nenhuma for encontrada).
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            return self._process_solutions(frame_rgb, w, h)
        elif self.backend == "tasks":
            return self._process_tasks(frame_rgb, w, h)
        return []

    def _process_solutions(self, frame_rgb: np.ndarray, w: int, h: int) -> List[HandDetection]:
        results = self.detector.process(frame_rgb)
        if not results.multi_hand_landmarks:
            for f in self.filters:
                f.reset()
            return []

        detections = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:self.max_hands]):
            handedness = "Right"
            confidence = 0.90

            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label
                confidence = results.multi_handedness[idx].classification[0].score

            norm_coords = np.zeros((21, 3), dtype=np.float32)
            pixel_coords = np.zeros((21, 2), dtype=np.int32)

            for i, lm in enumerate(hand_landmarks.landmark):
                norm_coords[i] = [lm.x, lm.y, lm.z]

            # Estabilização temporal One-Euro para cada mão
            filter_obj = self.filters[idx] if idx < len(self.filters) else self.filters[0]
            norm_coords = filter_obj.filter(norm_coords)

            pixel_coords[:, 0] = np.clip(norm_coords[:, 0] * w, 0, w - 1).astype(np.int32)
            pixel_coords[:, 1] = np.clip(norm_coords[:, 1] * h, 0, h - 1).astype(np.int32)

            detections.append(HandDetection(
                landmarks_norm=norm_coords,
                landmarks_pixel=pixel_coords,
                handedness=handedness,
                confidence=float(confidence)
            ))

        return detections

    def _process_tasks(self, frame_rgb: np.ndarray, w: int, h: int) -> List[HandDetection]:
        import mediapipe as mp
        frame_contiguous = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_contiguous)
        result = self.detector.detect(mp_img)


        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            for f in self.filters:
                f.reset()
            return []

        detections = []
        for idx, hand_lms in enumerate(result.hand_landmarks[:self.max_hands]):
            handedness = "Right"
            confidence = 0.90

            if result.handedness and idx < len(result.handedness) and len(result.handedness[idx]) > 0:
                handedness = result.handedness[idx][0].category_name
                confidence = result.handedness[idx][0].score

            norm_coords = np.zeros((21, 3), dtype=np.float32)
            pixel_coords = np.zeros((21, 2), dtype=np.int32)

            for i, lm in enumerate(hand_lms):
                norm_coords[i] = [lm.x, lm.y, lm.z]

            # Estabilização temporal One-Euro para cada mão
            filter_obj = self.filters[idx] if idx < len(self.filters) else self.filters[0]
            norm_coords = filter_obj.filter(norm_coords)

            pixel_coords[:, 0] = np.clip(norm_coords[:, 0] * w, 0, w - 1).astype(np.int32)
            pixel_coords[:, 1] = np.clip(norm_coords[:, 1] * h, 0, h - 1).astype(np.int32)

            detections.append(HandDetection(
                landmarks_norm=norm_coords,
                landmarks_pixel=pixel_coords,
                handedness=handedness,
                confidence=float(confidence)
            ))

        return detections

    def draw_hands(self, frame: np.ndarray, detections: List[HandDetection]) -> np.ndarray:
        """Renderiza os esqueletos de todas as mãos detectadas com cores distintas."""
        colors = [
            ((255, 190, 40), (0, 240, 255)),  # Mão 1: Dourado / Ciano
            ((255, 80, 180), (120, 255, 160))  # Mão 2: Magenta / Verde Menta
        ]
        for idx, det in enumerate(detections):
            bone_c, joint_c = colors[idx % len(colors)]
            self.draw_hand(frame, det, bone_color=bone_c, joint_color=joint_c)
        return frame

    def draw_hand(self,
                  frame: np.ndarray,
                  detection: HandDetection,
                  bone_color: Tuple[int, int, int] = (255, 190, 40),
                  joint_color: Tuple[int, int, int] = (0, 240, 255),
                  tip_color: Tuple[int, int, int] = (60, 230, 100)) -> np.ndarray:
        """
        Renderiza overlay visual do esqueleto da mão com estética futurista e alta visibilidade.
        """
        pts = detection.landmarks_pixel

        # 1. Desenha os ossos (linhas de conexão) com efeito duplo (borda e núcleo)
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (pts[start_idx, 0], pts[start_idx, 1])
            pt2 = (pts[end_idx, 0], pts[end_idx, 1])
            # Linha de fundo / sombra para contraste
            cv2.line(frame, pt1, pt2, (10, 10, 15), 5, cv2.LINE_AA)
            # Linha principal com cor vibrante
            cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        # 2. Desenha as articulações intermediárias
        for i in range(21):
            x, y = pts[i, 0], pts[i, 1]
            if i in FINGERTIP_IDS:
                # Pontas dos dedos: anel brilhante com núcleo destacado
                cv2.circle(frame, (x, y), 8, (10, 10, 15), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 6, tip_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Articulações comuns
                cv2.circle(frame, (x, y), 5, (10, 10, 15), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 3, joint_color, -1, cv2.LINE_AA)

        return frame

    def release(self):
        """Libera recursos se aplicável."""
        if hasattr(self, "detector") and hasattr(self.detector, "close"):
            self.detector.close()
