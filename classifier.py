"""
Motor de Classificação Heurística Geométrica Vetorial de Alta Precisão para Libras.
Utiliza projeção canônica da mão (invariância total à rotação e escala), cálculo de
ângulos de flexão biológica das falanges e regras de desambiguação topológica fina.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from hand_tracker import HandDetection
from profile_manager import SignProfileManager, extract_hand_features


@dataclass
class ClassificationResult:
    """Resultado da inferência geométrica e vetorial de um frame."""
    letter: Optional[str]                  # Letra reconhecida ('A', 'B', 'L', etc.) ou None
    confidence: float                      # Confiança geométrica calculada [0.0, 1.0]
    finger_states: List[bool]              # [Polegar, Indicador, Médio, Anelar, Mínimo]
    finger_names: Tuple[str, ...]          # ('Polegar', 'Indicador', 'Médio', 'Anelar', 'Mínimo')
    details: Dict[str, float]              # Métricas geométricas e cinéticas auxiliares
    is_transition: bool = False            # Indicador de fase cinética transitória (coarticulação rápida)
    kinetic_velocity: float = 0.0          # Velocidade cinemática das pontas dos dedos
    vector_score: float = 0.0              # Pontuação de similaridade com perfil vetorial
    best_vector_sign: Optional[str] = None # Melhor sinal correspondente no espaço vetorial


class LibrasClassifier:
    """
    Classificador geométrico determinístico com base canônica invariante à pose
    e modelo híbrido de assinaturas vetoriais com filtro cinemático de coarticulação.
    """

    FINGER_NAMES = ("Polegar", "Indicador", "Médio", "Anelar", "Mínimo")

    def __init__(self):
        # Índices dos marcos anatômicos do MediaPipe
        self.WRIST = 0
        self.THUMB = {"cmc": 1, "mcp": 2, "ip": 3, "tip": 4}
        self.INDEX = {"mcp": 5, "pip": 6, "dip": 7, "tip": 8}
        self.MIDDLE = {"mcp": 9, "pip": 10, "dip": 11, "tip": 12}
        self.RING = {"mcp": 13, "pip": 14, "dip": 15, "tip": 16}
        self.PINKY = {"mcp": 17, "pip": 18, "dip": 19, "tip": 20}

        # Limiares adaptativos
        self.spread_threshold: float = 0.22
        self.extension_threshold: float = 1.08

        # Motor de assinaturas de perfil e calibração biométrica
        self.profile_manager = SignProfileManager()

        # Histórico temporal para detecção de velocidade cinética e transição
        self._prev_landmarks: Optional[np.ndarray] = None
        self._prev_time: float = 0.0

    def set_calibration_profile(self, profile):
        """Aplica os limiares calibrados da mão do usuário."""
        if profile and hasattr(profile, "is_calibrated") and profile.is_calibrated:
            self.spread_threshold = profile.finger_spread_threshold
            self.extension_threshold = profile.finger_extension_threshold
            print(f"[Classificador] Limiares calibrados: Spread={self.spread_threshold:.2f}, Ext={self.extension_threshold:.2f}")

    def _dist(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Distância euclidiana 2D."""
        return float(np.linalg.norm(p1[:2] - p2[:2]))

    def _dist3d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Distância euclidiana 3D."""
        return float(np.linalg.norm(p1 - p2))

    def _angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Ângulo em graus entre dois vetores."""
        dot = np.dot(v1, v2)
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-7
        cos_theta = np.clip(dot / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _compute_canonical_frame(self, lm: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Projeta os landmarks 3D no referencial canônico da mão:
        - Origem no Pulso (0)
        - Eixo +Y aponta ao longo da mão (Pulso -> Junta do Médio 9)
        - Eixo +X aponta ao longo das juntas MCP (5 -> 17)
        - Eixo +Z aponta na normal da palma
        Garante invariância absoluta a rotações e inclinações da mão.
        """
        p0 = lm[self.WRIST]
        p5 = lm[self.INDEX["mcp"]]
        p9 = lm[self.MIDDLE["mcp"]]
        p17 = lm[self.PINKY["mcp"]]

        palm_size = float(np.linalg.norm(p9[:2] - p0[:2]))
        if palm_size < 1e-4:
            palm_size = 0.2

        # 1. Eixo Y local (Longitudinal / Direção para cima da mão)
        u_y = p9 - p0
        u_y = u_y / (np.linalg.norm(u_y) + 1e-6)

        # 2. Eixo X local (Transversal / Ao longo das juntas)
        v_knuckles = p17 - p5
        u_x = v_knuckles - np.dot(v_knuckles, u_y) * u_y
        u_x = u_x / (np.linalg.norm(u_x) + 1e-6)

        # 3. Eixo Z local (Normal da palma)
        u_z = np.cross(u_x, u_y)
        u_z = u_z / (np.linalg.norm(u_z) + 1e-6)

        # Matriz de rotação canônica 3x3
        R = np.vstack([u_x, u_y, u_z])

        # Landmarks canônicos normalizados pela escala da palma
        lm_canon = ((lm - p0) / palm_size) @ R.T

        thumb_is_on_left = lm[self.INDEX["mcp"], 0] < lm[self.PINKY["mcp"], 0]
        return lm_canon, palm_size, thumb_is_on_left

    def classify(self,
                 detection: HandDetection,
                 timestamp: Optional[float] = None) -> ClassificationResult:
        """
        Inferência geométrica e vetorial de alta precisão baseada em coordenadas canônicas,
        filtro cinético de coarticulação e modelo híbrido de assinaturas biológicas.
        """
        lm = detection.landmarks_norm  # Shape (21, 3)

        # 1. Projeção Canônica
        c, palm_size, thumb_is_on_left = self._compute_canonical_frame(lm)

        # 2. Avaliação Cinética de Deslocamento das Pontas dos Dedos (Filtro de Coarticulação)
        now = time.time() if timestamp is None else timestamp
        dt = (now - self._prev_time) if self._prev_time > 0 else 0.033
        if dt <= 0:
            dt = 0.033

        kinetic_vel = 0.0
        is_transition = False
        if self._prev_landmarks is not None and palm_size > 1e-4:
            tips = [4, 8, 12, 16, 20]
            disp = float(np.mean([
                np.linalg.norm(lm[tip, :2] - self._prev_landmarks[tip, :2])
                for tip in tips
            ]))
            kinetic_vel = float(disp / (palm_size * dt))
            # Se as pontas dos dedos estiverem em deslocamento rápido, é transição cinética
            if kinetic_vel > 1.25:
                is_transition = True

        self._prev_landmarks = lm.copy()
        self._prev_time = now

        # 3. Extração do Vetor Denso Invariante e Similaridade com Assinaturas
        features = extract_hand_features(c, lm)
        best_vec_sign, vec_score = self.profile_manager.get_best_match(features)

        # 4. Avaliação de Extensão das 4 Falanges (Indicador, Médio, Anelar, Mínimo)
        # No referencial canônico: Y é a extensão ao longo do eixo longitudinal da mão
        def is_finger_extended(f_dict):
            mcp_idx, pip_idx, tip_idx = f_dict["mcp"], f_dict["pip"], f_dict["tip"]
            y_diff = c[tip_idx, 1] - c[pip_idx, 1]
            dist_ratio = self._dist(lm[self.WRIST], lm[tip_idx]) / max(1e-4, self._dist(lm[self.WRIST], lm[pip_idx]))
            return bool(y_diff > 0.16 and dist_ratio > (self.extension_threshold - 0.06))

        index_ext = is_finger_extended(self.INDEX)
        middle_ext = is_finger_extended(self.MIDDLE)
        ring_ext = is_finger_extended(self.RING)
        pinky_ext = is_finger_extended(self.PINKY)

        # 5. Avaliação de Extensão do Polegar
        thumb_lateral_dist = abs(c[self.THUMB["tip"], 0] - c[self.THUMB["mcp"], 0])
        thumb_dist_pinky = self._dist(lm[self.THUMB["tip"]], lm[self.PINKY["mcp"]]) / palm_size
        thumb_tip_y = c[self.THUMB["tip"], 1]

        # Polegar estendido para fora
        thumb_ext = bool(thumb_dist_pinky > 0.46 and (thumb_lateral_dist > 0.22 or thumb_tip_y > 0.85))

        finger_states = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]

        # 6. Métricas Relativas Canônicas
        d_thumb_index = self._dist(lm[self.THUMB["tip"]], lm[self.INDEX["tip"]]) / palm_size
        d_thumb_middle = self._dist(lm[self.THUMB["tip"]], lm[self.MIDDLE["tip"]]) / palm_size
        d_index_middle = self._dist(lm[self.INDEX["tip"]], lm[self.MIDDLE["tip"]]) / palm_size
        d_thumb_pinky = thumb_dist_pinky

        v_thumb = lm[self.THUMB["tip"], :2] - lm[self.THUMB["mcp"], :2]
        v_index = lm[self.INDEX["tip"], :2] - lm[self.INDEX["mcp"], :2]
        thumb_index_angle = self._angle(v_thumb, v_index)

        details = {
            "palm_size": palm_size,
            "d_thumb_index": d_thumb_index,
            "d_thumb_middle": d_thumb_middle,
            "d_index_middle": d_index_middle,
            "d_thumb_pinky": d_thumb_pinky,
            "thumb_index_angle": thumb_index_angle,
            "kinetic_velocity": kinetic_vel,
            "vector_score": vec_score,
            "is_transition": 1.0 if is_transition else 0.0
        }

        # 7. Avaliação Heurística Fina dos Sinais de Libras
        letter, conf = self._evaluate_signs(
            lm=lm,
            c=c,
            finger_states=finger_states,
            palm_size=palm_size,
            thumb_is_on_left=thumb_is_on_left,
            d_thumb_index=d_thumb_index,
            d_thumb_middle=d_thumb_middle,
            d_index_middle=d_index_middle,
            d_thumb_pinky=d_thumb_pinky,
            thumb_index_angle=thumb_index_angle
        )

        # 8. Fusão Híbrida Inteligente (Regras Anatômicas + Modelo de Assinaturas Vetoriais)
        if letter is not None and letter == best_vec_sign:
            # Alta concordância entre as regras canônicas e a assinatura vetorial
            conf = min(0.99, max(conf, vec_score) * 1.04)
        elif letter is None and vec_score >= 0.88:
            # Resgate de sinal pelo perfil vetorial quando as regras ficam indecisas
            letter = best_vec_sign
            conf = float(vec_score * 0.94)
        elif best_vec_sign and self.profile_manager.is_calibrated(best_vec_sign) and vec_score >= 0.88:
            # O usuário calibrou pessoalmente este sinal com sua própria mão (biometria personalizada)
            letter = best_vec_sign
            conf = max(conf, vec_score)

        # 9. Atenuação durante fase rápida de transição (elimina ruído de coarticulação)
        if is_transition:
            conf = conf * 0.40

        return ClassificationResult(
            letter=letter,
            confidence=conf,
            finger_states=finger_states,
            finger_names=self.FINGER_NAMES,
            details=details,
            is_transition=is_transition,
            kinetic_velocity=kinetic_vel,
            vector_score=vec_score,
            best_vector_sign=best_vec_sign
        )

    def _evaluate_signs(self,
                        lm: np.ndarray,
                        c: np.ndarray,
                        finger_states: List[bool],
                        palm_size: float,
                        thumb_is_on_left: bool,
                        d_thumb_index: float,
                        d_thumb_middle: float,
                        d_index_middle: float,
                        d_thumb_pinky: float,
                        thumb_index_angle: float) -> Tuple[Optional[str], float]:
        """Avalia regras anatômicas no espaço canônico e espacial para cada sinal."""
        t, i, m, r, p = finger_states

        # Vetor vertical da mão (pulso -> junta base do dedo médio)
        # Em coordenadas normalizadas de imagem: y=0 no topo e y=1 na base
        # Mão normal ereta para cima: lm[9, 1] < lm[0, 1] => v_hand_y < -0.10
        # Mão apontada para baixo (M / N / P): lm[9, 1] >= lm[0, 1] => v_hand_y >= -0.05
        v_hand_y = lm[self.MIDDLE["mcp"], 1] - lm[self.WRIST, 1]

        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # 1. COMANDOS ESPECIAIS & SINAIS DE 5 DEDOS
        # -------------------------------------------------------------
        # APAGAR (Polegar apontando para baixo com punho fechado - Thumbs Down)
        if c[self.THUMB["tip"], 1] < -0.15 and not i and not m and not r and not p:
            return "APAGAR", 0.95

        # ESPAÇO (Mão completamente aberta e espalmada de frente com dedos separados e erguidos)
        if v_hand_y < -0.05 and t and i and m and r and p and d_index_middle > 0.20 and d_thumb_pinky > 0.65:
            return "ESPAÇO", 0.95


        # -------------------------------------------------------------
        # 2. SINAIS COM ORIENTAÇÃO DA MÃO PARA BAIXO: 'M', 'N', 'P', 'Q'
        # -------------------------------------------------------------
        if v_hand_y >= -0.05:
            index_pointing_down = lm[self.INDEX["tip"], 1] > lm[self.INDEX["mcp"], 1] + 0.05
            middle_pointing_down = lm[self.MIDDLE["tip"], 1] > lm[self.MIDDLE["mcp"], 1] + 0.05

            if index_pointing_down and middle_pointing_down and not p:
                ring_pointing_down = lm[self.RING["tip"], 1] > lm[self.RING["mcp"], 1] + 0.10
                if ring_pointing_down:
                    return "M", 0.95
                else:
                    return "N", 0.95

            # LETRA 'Q': Indicador e polegar estendidos paralelamente apontando para baixo
            if t and i and not m and not r and not p and thumb_index_angle < 52.0:
                return "Q", 0.94

            # Letra 'P': Configuração do K voltada para baixo
            middle_forward = (c[self.MIDDLE["tip"], 1] < c[self.INDEX["tip"], 1] - 0.20 and
                              c[self.MIDDLE["tip"], 1] > 0.35)
            if i and not r and not p and middle_forward:
                return "P", 0.92

        # -------------------------------------------------------------
        # 3. SINAIS DE 4 DEDOS LEVANTADOS: 'B'
        # -------------------------------------------------------------
        # LETRA 'B' (4 dedos estendidos para cima unidos, polegar recolhido)
        if i and m and r and p:
            if d_index_middle < 0.35:
                return "B", 0.96

        # -------------------------------------------------------------
        # 4. SINAIS DE 3 DEDOS LEVANTADOS: 'W', 'F', 'T'
        # -------------------------------------------------------------
        # LETRA 'W' (Indicador, Médio e Anelar estendidos para cima)
        if i and m and r and not p:
            return "W", 0.95

        # LETRAS 'F' e 'T' (Indicador dobrado com 3 dedos erguidos)
        if not i and m and r and p:
            # LETRA 'T': Polegar inserido por DENTRO (entre indicador e médio)
            is_inside = (d_thumb_middle < 0.36) or (c[self.THUMB["tip"], 0] > c[self.INDEX["mcp"], 0] + 0.04)
            if is_inside:
                return "T", 0.94
            else:
                # LETRA 'F': Polegar apoiado por FORA (lateral externa do indicador)
                return "F", 0.94

        # -------------------------------------------------------------
        # 5. SINAIS DE 2 DEDOS LEVANTADOS: 'Y', 'R', 'V', 'U'
        # -------------------------------------------------------------
        # LETRA 'Y' (Polegar e Mínimo estendidos - Hang Loose)
        if t and not i and not m and not r and p and d_thumb_pinky > 0.65:
            return "Y", 0.96

        # LETRAS 'R', 'V', 'U' (Indicador e Médio estendidos)
        if i and m and not r and not p:
            # LETRA 'R' (Indicador e médio cruzados em hélice)
            crossed = (c[self.INDEX["tip"], 0] > c[self.MIDDLE["tip"], 0] + 0.02)
            if crossed:
                return "R", 0.95
            elif d_index_middle > self.spread_threshold:
                return "V", 0.95
            else:
                return "U", 0.94

        # -------------------------------------------------------------
        # 6. SINAIS DE 1 DEDO LEVANTADO: 'L', 'G', 'I', 'D', 'K', 'X'
        # -------------------------------------------------------------
        # LETRA 'L' ou 'G' (Polegar e Indicador)
        if t and i and not m and not r and not p:
            if thumb_index_angle >= 48.0:
                return "L", 0.96
            else:
                if v_hand_y < 0.15:
                    return "G", 0.93

        # LETRA 'I' (Apenas dedo mínimo levantado)
        if not i and not m and not r and p:
            if d_thumb_pinky < 0.70:
                return "I", 0.95

        # LETRAS 'D', 'K', 'X' (Indicador erguido ou gancho)
        if not r and not p:
            # LETRA 'X': Indicador em gancho (PIP erguida mas ponta dobrada para trás)
            index_hook = (c[self.INDEX["pip"], 1] > c[self.INDEX["mcp"], 1] + 0.12 and
                          c[self.INDEX["tip"], 1] < c[self.INDEX["pip"], 1] - 0.04 and
                          not m)
            if index_hook:
                return "X", 0.93

            # LETRA 'K': Médio projetado para frente com polegar apoiado
            middle_forward = (c[self.MIDDLE["tip"], 1] < c[self.INDEX["tip"], 1] - 0.20 and
                              c[self.MIDDLE["tip"], 1] > 0.35)
            if i and middle_forward and d_thumb_middle < 0.50:
                return "K", 0.92

            # LETRA 'D': Indicador para cima, outros dedos dobrados
            if i and not m:
                return "D", 0.94

        # -------------------------------------------------------------
        # 7. SINAIS CURVADOS: 'O', 'C'
        # -------------------------------------------------------------
        if not (i or m or r or p):
            # LETRA 'O' (Pontas dos dedos e polegar tocando em círculo fechado)
            if d_thumb_index < 0.22 and d_thumb_middle < 0.26:
                return "O", 0.93

            # LETRA 'C' (Mão formando arco semicircular aberto)
            if 0.25 < d_thumb_index < 0.65:
                if (0.80 < c[self.INDEX["tip"], 1] < 1.60 and
                    abs(c[self.THUMB["tip"], 1] - c[self.INDEX["tip"], 1]) > 0.20):
                    return "C", 0.91

        # -------------------------------------------------------------
        # 8. PUNHOS FECHADOS: 'A', 'S', 'E'
        # -------------------------------------------------------------
        if not i and not m and not r and not p:
            thumb_canon_y = c[self.THUMB["tip"], 1]
            thumb_canon_x = c[self.THUMB["tip"], 0]
            index_mcp_canon_x = c[self.INDEX["mcp"], 0]
            index_mcp_canon_y = c[self.INDEX["mcp"], 1]

            # LETRA 'A': Polegar ereto na lateral externa da mão (ao lado do indicador)
            if (thumb_canon_x <= index_mcp_canon_x + 0.04) and (thumb_canon_y >= index_mcp_canon_y - 0.20):
                return "A", 0.95

            # LETRA 'S': Polegar cruzado SOBRE a frente dos dedos
            if (thumb_canon_x > index_mcp_canon_x + 0.04) and (thumb_canon_y >= index_mcp_canon_y - 0.38):
                return "S", 0.94

            # LETRA 'E': Dedos dobrados para baixo repousando sobre o polegar recolhido
            return "E", 0.92

        return None, 0.0

