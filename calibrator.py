"""
Módulo de Calibração Anatômica Rápida para o Tradutor de Libras.
Permite adaptar os limiares de distância e abertura para a mão específica do usuário
(mãos infantis, mãos adultas, proporções anatômicas variadas) em apenas 5 segundos.
"""

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple
import numpy as np

from hand_tracker import HandDetection


@dataclass
class CalibrationProfile:
    """Perfil biométrico calibrado do usuário."""
    palm_scale_multiplier: float = 1.0
    finger_extension_threshold: float = 1.10
    finger_spread_threshold: float = 0.35
    is_calibrated: bool = False


class HandCalibrator:
    """
    Máquina de estados para calibração interativa (Mão Aberta -> Punho Fechado).
    """

    STATE_IDLE = 0
    STATE_OPEN = 1
    STATE_CLOSED = 2
    STATE_DONE = 3

    STEP_DURATION = 2.5  # Segundos por etapa

    def __init__(self):
        self.state = self.STATE_IDLE
        self.step_start_time = 0.0
        self.profile = CalibrationProfile()

        # Amostras coletadas durante a calibração
        self.open_palm_samples = []
        self.open_spread_samples = []
        self.closed_ratio_samples = []

    def start_calibration(self):
        """Inicia o processo de calibração."""
        self.state = self.STATE_OPEN
        self.step_start_time = time.time()
        self.open_palm_samples.clear()
        self.open_spread_samples.clear()
        self.closed_ratio_samples.clear()
        print("[Calibrador] Calibração iniciada. Etapa 1: Mão Aberta.")

    def cancel(self):
        """Cancela calibração."""
        self.state = self.STATE_IDLE

    def update(self, detection: Optional[HandDetection]) -> Optional[CalibrationProfile]:
        """
        Processa frame da calibração. Retorna o perfil calibrado ao concluir.
        """
        if self.state == self.STATE_IDLE:
            return None

        now = time.time()
        elapsed = now - self.step_start_time

        # -------------------------------------------------------------
        # ETAPA 1: MÃO ABERTA
        # -------------------------------------------------------------
        if self.state == self.STATE_OPEN:
            if detection is not None:
                lm = detection.landmarks_norm
                palm_sz = np.linalg.norm(lm[0, :2] - lm[9, :2])
                spread = np.linalg.norm(lm[8, :2] - lm[12, :2]) / max(1e-4, palm_sz)
                self.open_palm_samples.append(palm_sz)
                self.open_spread_samples.append(spread)

            if elapsed >= self.STEP_DURATION:
                self.state = self.STATE_CLOSED
                self.step_start_time = now
                print("[Calibrador] Etapa 1 concluída. Etapa 2: Punho Fechado.")

            return None

        # -------------------------------------------------------------
        # ETAPA 2: PUNHO FECHADO
        # -------------------------------------------------------------
        elif self.state == self.STATE_CLOSED:
            if detection is not None:
                lm = detection.landmarks_norm
                palm_sz = np.linalg.norm(lm[0, :2] - lm[9, :2])
                # Razão da ponta do indicador em punho fechado
                tip_ratio = np.linalg.norm(lm[0, :2] - lm[8, :2]) / max(1e-4, palm_sz)
                self.closed_ratio_samples.append(tip_ratio)

            if elapsed >= self.STEP_DURATION:
                self.state = self.STATE_DONE
                self._compute_profile()
                print("[Calibrador] Calibração concluída com sucesso!")
                return self.profile

        elif self.state == self.STATE_DONE:
            # Reseta para ociosidade após notificar
            self.state = self.STATE_IDLE

        return None

    def _compute_profile(self):
        """Calcula os limiares personalizados a partir das amostras."""
        if self.open_spread_samples:
            avg_spread = float(np.median(self.open_spread_samples))
            # O limiar de separação fica em 70% do spread máximo observado
            self.profile.finger_spread_threshold = max(0.25, min(0.48, avg_spread * 0.70))

        if self.closed_ratio_samples:
            avg_closed = float(np.median(self.closed_ratio_samples))
            # Ajusta limiar de extensão
            self.profile.finger_extension_threshold = max(1.05, min(1.20, avg_closed * 1.30))

        self.profile.is_calibrated = True

    def is_active(self) -> bool:
        """Verifica se a calibração está em andamento."""
        return self.state in (self.STATE_OPEN, self.STATE_CLOSED)

    def get_hud_instruction(self) -> Tuple[str, float]:
        """
        Retorna (texto_instrução, progresso_0_a_1) para exibição no HUD.
        """
        if self.state == self.STATE_OPEN:
            elapsed = time.time() - self.step_start_time
            prog = min(1.0, elapsed / self.STEP_DURATION)
            return "PASSO 1/2: ABRA A MAO E ESPALME OS DEDOS", prog

        elif self.state == self.STATE_CLOSED:
            elapsed = time.time() - self.step_start_time
            prog = min(1.0, elapsed / self.STEP_DURATION)
            return "PASSO 2/2: FECHE O PUNHO COMPLETAMENTE", prog

        return "", 0.0
