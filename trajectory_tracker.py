"""
Rastreador de Trajetória e Reconhecedor de Sinais Dinâmicos de Libras (Letras 'J' e 'Z').
Monitora a cinemática da ponta dos dedos ao longo do tempo (deslocamentos normalizados),
permitindo identificar movimentos específicos no ar sem necessidade de redes neurais pesadas.
"""

from collections import deque
import time
from typing import List, Optional, Tuple
import numpy as np

from hand_tracker import HandDetection


class TrajectoryTracker:
    """
    Rastreia o movimento das pontas dos dedos e reconhece gestos com trajetória espacial.
    """

    def __init__(self, max_points: int = 25, cool_down_sec: float = 1.2):
        self.max_points = max_points
        self.cool_down = cool_down_sec
        self.last_detection_time = 0.0

        # Histórico de coordenadas relativas (ponta - pulso) com timestamps
        # Cada item: (x, y, timestamp)
        self.pinky_history = deque(maxlen=max_points)
        self.index_history = deque(maxlen=max_points)

        # Histórico de coordenadas em pixels para desenho da trilha visual no HUD
        self.trail_pixels = deque(maxlen=max_points)

    def reset(self):
        """Limpa as trajetórias acumuladas."""
        self.pinky_history.clear()
        self.index_history.clear()
        self.trail_pixels.clear()

    def update(self,
               detection: Optional[HandDetection],
               finger_states: Optional[List[bool]] = None) -> Optional[Tuple[str, float]]:
        """
        Atualiza o histórico temporal e avalia se uma letra dinâmica ('J' ou 'Z') foi concluída.
        Retorna (letra, confiança) ou None.
        """
        now = time.time()

        if detection is None or finger_states is None:
            # Sem mão no frame: esvazia suavemente
            if len(self.trail_pixels) > 0:
                self.trail_pixels.popleft()
            return None

        # Respeita intervalo de cooldown após reconhecer um sinal dinâmico
        if (now - self.last_detection_time) < self.cool_down:
            return None

        lm = detection.landmarks_norm
        pts_px = detection.landmarks_pixel
        t, i, m, r, p = finger_states

        # Posição do pulso (Landmark 0) para subtrair deslocamento do corpo
        wrist_norm = lm[0, :2]

        # -------------------------------------------------------------
        # 1. RASTREAMENTO DA LETRA 'J' (Configuração 'I' - Mínimo em gancho)
        # -------------------------------------------------------------
        if p and not i and not m and not r:
            # Dedo mínimo levantado
            pinky_rel = lm[20, :2] - wrist_norm
            self.pinky_history.append((float(pinky_rel[0]), float(pinky_rel[1]), now))
            self.trail_pixels.append((int(pts_px[20, 0]), int(pts_px[20, 1])))

            # Avalia trajetória do 'J'
            detected_j = self._evaluate_j_hook()
            if detected_j:
                self.last_detection_time = now
                self.reset()
                return "J", 0.93

        else:
            if len(self.pinky_history) > 0:
                self.pinky_history.popleft()

        # -------------------------------------------------------------
        # 2. RASTREAMENTO DA LETRA 'Z' (Configuração 'D'/Apontar - Zigue-zague)
        # -------------------------------------------------------------
        if i and not m and not r and not p:
            # Indicador apontado
            index_rel = lm[8, :2] - wrist_norm
            self.index_history.append((float(index_rel[0]), float(index_rel[1]), now))
            self.trail_pixels.append((int(pts_px[8, 0]), int(pts_px[8, 1])))

            # Avalia trajetória do 'Z'
            detected_z = self._evaluate_z_stroke()
            if detected_z:
                self.last_detection_time = now
                self.reset()
                return "Z", 0.92

        else:
            if len(self.index_history) > 0:
                self.index_history.popleft()

        return None

    def _evaluate_j_hook(self) -> bool:
        """
        Reconhece o movimento do gancho do 'J':
        1. Desce em Y
        2. Curva na base e sobe em Y com deflexão lateral
        """
        if len(self.pinky_history) < 14:
            return False

        pts = list(self.pinky_history)
        dt = pts[-1][2] - pts[0][2]
        if not (0.25 <= dt <= 1.4):
            return False

        # Divide o movimento em primeira metade (descida) e segunda metade (curva/subida)
        n = len(pts)
        p_start = pts[0]
        p_mid = pts[n // 2]
        p_end = pts[-1]

        # Em imagem, Y aumenta para baixo
        delta_y_descida = p_mid[1] - p_start[1]   # Deve ser positivo (desceu)
        delta_y_subida = p_end[1] - p_mid[1]     # Deve ser negativo (subiu)
        delta_x_curva = abs(p_end[0] - p_mid[0]) # Curva lateral

        if delta_y_descida > 0.05 and delta_y_subida < -0.02 and delta_x_curva > 0.03:
            return True

        return False

    def _evaluate_z_stroke(self) -> bool:
        """
        Reconhece o desenho da letra 'Z' no ar:
        1. Traço horizontal para a direita
        2. Diagonal descendo para a esquerda
        3. Traço horizontal para a direita
        """
        if len(self.index_history) < 16:
            return False

        pts = list(self.index_history)
        dt = pts[-1][2] - pts[0][2]
        if not (0.35 <= dt <= 1.8):
            return False

        n = len(pts)
        idx1 = n // 3
        idx2 = (2 * n) // 3

        p0 = pts[0]
        p1 = pts[idx1]
        p2 = pts[idx2]
        p3 = pts[-1]

        # Traço 1: Movimento horizontal significativo
        dx1 = abs(p1[0] - p0[0])
        # Traço 2: Diagonal com componente vertical
        dy2 = p2[1] - p1[1]
        # Traço 3: Movimento horizontal final
        dx3 = abs(p3[0] - p2[0])

        if dx1 > 0.04 and dy2 > 0.04 and dx3 > 0.04:
            return True

        return False

    def get_trail_pixels(self) -> List[Tuple[int, int]]:
        """Retorna as coordenadas de pixel dos pontos recentes para desenho da trilha."""
        return list(self.trail_pixels)
