"""
Classificador de Sinais Lexicais e Bimanuais de Libras.
Interpreta palavras inteiras através de:
1. Rastreamento bimanual (interação entre 2 mãos: CASA, AJUDA, TRABALHO, FAMÍLIA).
2. Ponto de Articulação Facial (mão relativa ao queixo/boca/testa: ÁGUA, DESCULPA, BOM, SABER).
3. Sinais lexicais dinâmicos unimanuais (OI, SIM, NÃO, EU, VOCÊ).
"""

from collections import deque
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from classifier import LibrasClassifier
from face_tracker import FaceKeypoints
from hand_tracker import HandDetection


class LexicalClassifier:
    """
    Motor de reconhecimento de sinais lexicais inteiros e interação bimanual.
    """

    def __init__(self, static_classifier: LibrasClassifier):
        self.static_clf = static_classifier
        self.last_lexical_time = 0.0
        self.cool_down = 1.2  # Evita repetições consecutivas instantâneas

        # Histórico de oscilações para sinais dinâmicos (SIM e NÃO)
        self.oscillation_history = deque(maxlen=20)
        self.oi_history = deque(maxlen=25)

    def evaluate(self,
                 hands: List[HandDetection],
                 face: Optional[FaceKeypoints]) -> Optional[Tuple[str, float, str]]:
        """
        Avalia o frame e retorna (palavra, confiança, tipo_de_sinal) ou None.
        tipo_de_sinal pode ser: 'BIMANUAL', 'FACIAL', 'LEXICAL_DINAMICO'.
        """
        now = time.time()
        if (now - self.last_lexical_time) < self.cool_down:
            return None

        # -------------------------------------------------------------
        # 1. SINAIS BIMANUAIS (EXATAMENTE 2 MÃOS DETECTADAS)
        # -------------------------------------------------------------
        if len(hands) >= 2:
            res_bimanual = self._evaluate_bimanual(hands[0], hands[1])
            if res_bimanual:
                self.last_lexical_time = now
                return res_bimanual[0], res_bimanual[1], "BIMANUAL"

        # -------------------------------------------------------------
        # 2. SINAIS COM PONTO DE ARTICULAÇÃO FACIAL (1 MÃO + ROSTO)
        # -------------------------------------------------------------
        if len(hands) >= 1 and face is not None:
            res_facial = self._evaluate_facial_articulation(hands[0], face)
            if res_facial:
                self.last_lexical_time = now
                return res_facial[0], res_facial[1], "FACIAL"

        # -------------------------------------------------------------
        # 3. SINAIS LEXICAIS DINÂMICOS UNIMANUAIS (1 MÃO)
        # -------------------------------------------------------------
        if len(hands) >= 1:
            res_dyn = self._evaluate_unimanual_lexical(hands[0])
            if res_dyn:
                self.last_lexical_time = now
                return res_dyn[0], res_dyn[1], "LEXICAL_DINAMICO"

        return None

    def _evaluate_bimanual(self, h1: HandDetection, h2: HandDetection) -> Optional[Tuple[str, float]]:
        """Avalia interações geométricas e espaciais entre as duas mãos."""
        # Classifica as formas estáticas individuais de cada mão
        res1 = self.static_clf.classify(h1)
        res2 = self.static_clf.classify(h2)

        p1_wrist = h1.landmarks_norm[0, :2]
        p2_wrist = h2.landmarks_norm[0, :2]

        p1_index = h1.landmarks_norm[8, :2]
        p2_index = h2.landmarks_norm[8, :2]

        p1_thumb = h1.landmarks_norm[4, :2]
        p2_thumb = h2.landmarks_norm[4, :2]

        d_wrists = float(np.linalg.norm(p1_wrist - p2_wrist))
        d_index_tips = float(np.linalg.norm(p1_index - p2_index))
        d_thumbs = float(np.linalg.norm(p1_thumb - p2_thumb))

        # -------------------------------------------------------------
        # SINAL: 'CASA'
        # Ambas as mãos espalmadas (B), pontas dos dedos tocando no topo (telhado)
        # e pulsos afastados na base
        # -------------------------------------------------------------
        both_open = (res1.letter == "B" or (res1.finger_states[1] and res1.finger_states[2])) and \
                    (res2.letter == "B" or (res2.finger_states[1] and res2.finger_states[2]))
        if both_open and d_index_tips < 0.16 and d_wrists > 0.28:
            return "CASA", 0.95

        # -------------------------------------------------------------
        # SINAL: 'AJUDA'
        # Uma mão aberta plana horizontal (base) e a outra em punho com polegar ereto sobre ela
        # -------------------------------------------------------------
        one_flat = (res1.letter == "B" and abs(h1.landmarks_norm[8, 1] - h1.landmarks_norm[0, 1]) < 0.15) or \
                   (res2.letter == "B" and abs(h2.landmarks_norm[8, 1] - h2.landmarks_norm[0, 1]) < 0.15)

        one_fist = (res1.letter in ("A", "S") or res1.finger_states[0]) or \
                   (res2.letter in ("A", "S") or res2.finger_states[0])

        if one_flat and one_fist and d_index_tips < 0.22:
            return "AJUDA", 0.93

        # -------------------------------------------------------------
        # SINAL: 'TRABALHO'
        # Duas mãos em configuração 'L' apontando para baixo
        # -------------------------------------------------------------
        both_l = (res1.letter in ("L", "G")) and (res2.letter in ("L", "G"))
        both_pointing_down = (h1.landmarks_norm[8, 1] > h1.landmarks_norm[0, 1]) and \
                             (h2.landmarks_norm[8, 1] > h2.landmarks_norm[0, 1])
        if both_l and both_pointing_down and d_wrists < 0.40:
            return "TRABALHO", 0.94

        # -------------------------------------------------------------
        # SINAL: 'FAMÍLIA'
        # Duas mãos em configuração 'F' unidas
        # -------------------------------------------------------------
        both_f = (res1.letter == "F") and (res2.letter == "F")
        if both_f and (d_index_tips < 0.20 or d_thumbs < 0.20):
            return "FAMILIA", 0.93

        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # SINAL: 'APLAUSOS' (Palmas em Libras)
        # Ambas as mãos no alto acenando com dedos abertos
        # -------------------------------------------------------------
        both_high = (h1.landmarks_norm[0, 1] < 0.45) and (h2.landmarks_norm[0, 1] < 0.45)
        both_all_open = all(res1.finger_states) and all(res2.finger_states)
        if both_high and both_all_open and d_wrists > 0.30:
            return "APLAUSOS", 0.92

        # -------------------------------------------------------------
        # SINAL: 'POR FAVOR'
        # Ambas as mãos espalmadas unidas em frente ao peito (súplica/pedido)
        # -------------------------------------------------------------
        both_flat = (res1.letter == "B" or (res1.finger_states[1] and res1.finger_states[2])) and \
                    (res2.letter == "B" or (res2.finger_states[1] and res2.finger_states[2]))
        if both_flat and d_index_tips < 0.15 and d_wrists < 0.22:
            return "POR FAVOR", 0.94

        # -------------------------------------------------------------
        # SINAL: 'NOME'
        # Ambas as mãos em configuração 'U' tocando uma na outra
        # -------------------------------------------------------------
        both_u = (res1.letter in ("U", "V")) and (res2.letter in ("U", "V"))
        if both_u and d_index_tips < 0.18:
            return "NOME", 0.93

        return None

    def _evaluate_facial_articulation(self, hand: HandDetection, face: FaceKeypoints) -> Optional[Tuple[str, float]]:
        """Determina o significado do sinal cruzando configuração da mão com o ponto de contato no rosto."""
        res = self.static_clf.classify(hand)

        tip_index = hand.landmarks_norm[8, :2]
        tip_thumb = hand.landmarks_norm[4, :2]
        wrist = hand.landmarks_norm[0, :2]

        d_index_chin = float(np.linalg.norm(tip_index - face.chin[:2])) / face.face_size
        d_thumb_chin = float(np.linalg.norm(tip_thumb - face.chin[:2])) / face.face_size
        d_index_mouth = float(np.linalg.norm(tip_index - face.mouth[:2])) / face.face_size
        d_hand_forehead = float(np.linalg.norm(tip_index - face.forehead[:2])) / face.face_size

        # -------------------------------------------------------------
        # SINAL: 'ÁGUA' (Configuração 'L' encostando / batendo no queixo)
        # -------------------------------------------------------------
        if res.letter == "L" and (d_index_chin < 0.50 or d_thumb_chin < 0.50):
            return "AGUA", 0.95

        # -------------------------------------------------------------
        # SINAL: 'DESCULPA' (Configuração 'Y' no queixo / mandíbula)
        # -------------------------------------------------------------
        if res.letter == "Y" and (d_thumb_chin < 0.55 or d_index_chin < 0.60):
            return "DESCULPA", 0.94

        # -------------------------------------------------------------
        # SINAL: 'OBRIGADO' (Mão espalmada tocando a testa / queixo e saindo para frente)
        # -------------------------------------------------------------
        if res.letter == "B" and d_hand_forehead < 0.38:
            return "OBRIGADO", 0.94

        # -------------------------------------------------------------
        # SINAL: 'BOM' / 'ALIMENTO' (Mão em concha/fechada na boca)
        # -------------------------------------------------------------
        if (res.letter in ("O", "C", "B")) and d_index_mouth < 0.40:
            return "BOM", 0.92

        # -------------------------------------------------------------
        # SINAL: 'SABER' / 'APRENDER' (Mão tocando a lateral da testa)
        # -------------------------------------------------------------
        if (res.letter in ("B", "L", "A")) and d_hand_forehead < 0.45:
            return "SABER", 0.93

        return None

    def _evaluate_unimanual_lexical(self, hand: HandDetection) -> Optional[Tuple[str, float]]:
        """Avalia sinais dinâmicos rápidos com 1 mão (OI, SIM, NÃO, EU, VOCÊ, TE AMO, LEGAL)."""
        res = self.static_clf.classify(hand)
        now = time.time()

        lm = hand.landmarks_norm
        index_tip = lm[8]
        wrist = lm[0]

        # -------------------------------------------------------------
        # SINAL: 'TE AMO' (Configuração ILY: Polegar, Indicador e Mínimo erguidos)
        # -------------------------------------------------------------
        if res.finger_states[0] and res.finger_states[1] and not res.finger_states[2] and not res.finger_states[3] and res.finger_states[4]:
            return "TE AMO", 0.95

        # -------------------------------------------------------------
        # SINAL: 'LEGAL' / 'POSITIVO' (Polegar ereto para cima isolado com punho fechado)
        # -------------------------------------------------------------
        if res.finger_states[0] and not any(res.finger_states[1:]) and lm[4, 1] < lm[2, 1] - 0.08:
            return "LEGAL", 0.93

        # -------------------------------------------------------------
        # SINAL: 'EU' (Dedo indicador apontando para o próprio peito)
        # Z do indicador substancialmente mais positivo (para trás) ou apontando para o corpo
        # -------------------------------------------------------------
        if res.letter == "D" or (res.finger_states[1] and not res.finger_states[2]):
            # Vetor do pulso para o indicador aponta para baixo/peito
            if lm[8, 1] > lm[0, 1] - 0.05 and lm[8, 2] > 0.02:
                return "EU", 0.92

        # -------------------------------------------------------------
        # SINAL: 'VOCÊ' (Dedo indicador apontando diretamente para a câmera)
        # Z do indicador muito negativo (projetado para frente)
        # -------------------------------------------------------------
        if res.letter == "D" or (res.finger_states[1] and not res.finger_states[2]):
            if lm[8, 2] < -0.05:
                return "VOCE", 0.93

        # -------------------------------------------------------------
        # SINAL: 'NÃO' (Indicador balançando horizontalmente esquerda-direita)
        # -------------------------------------------------------------
        if res.finger_states[1] and not res.finger_states[2] and not res.finger_states[3]:
            self.oscillation_history.append((float(index_tip[0]), now))
            if len(self.oscillation_history) >= 12:
                xs = [item[0] for item in self.oscillation_history]
                # Conta reversões de sentido de movimento
                diffs = np.diff(xs)
                signs = np.sign(diffs)
                sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
                span_x = max(xs) - min(xs)
                if sign_changes >= 3 and span_x > 0.04:
                    self.oscillation_history.clear()
                    return "NAO", 0.94

        # -------------------------------------------------------------
        # SINAL: 'SIM' (Punho 'S' balançando verticalmente como aceno)
        # -------------------------------------------------------------
        if res.letter == "S" or (not any(res.finger_states[1:])):
            self.oscillation_history.append((float(wrist[1]), now))
            if len(self.oscillation_history) >= 12:
                ys = [item[0] for item in self.oscillation_history]
                diffs = np.diff(ys)
                signs = np.sign(diffs)
                sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
                span_y = max(ys) - min(ys)
                if sign_changes >= 2 and span_y > 0.035:
                    self.oscillation_history.clear()
                    return "SIM", 0.94

        # -------------------------------------------------------------
        # SINAL: 'OI' (Transição rápida de O para I com aceno)
        # -------------------------------------------------------------
        self.oi_history.append((res.letter, now))
        if len(self.oi_history) >= 10:
            letters = [item[0] for item in self.oi_history]
            has_o = any(l == "O" for l in letters[:len(letters)//2])
            has_i = any(l in ("I", "Y") for l in letters[len(letters)//2:])
            if has_o and has_i:
                self.oi_history.clear()
                return "OI", 0.95

        return None

