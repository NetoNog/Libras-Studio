"""
Filtro Temporal One-Euro (1€ Filter) para Suavização de Landmarks 3D.
Elimina o jitter e tremor natural do MediaPipe mantendo latência quase zero durante movimentos rápidos.
Padrão da indústria em realidade virtual e rastreamento biométrico.
"""

import math
import time
from typing import Optional
import numpy as np


class LowPassFilter:
    """Filtro passa-baixa de primeira ordem com constante de tempo dinâmica."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.y: Optional[np.ndarray] = None

    def reset(self):
        self.y = None

    def filter(self, value: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            self.y = value.copy()
        else:
            self.y = self.alpha * value + (1.0 - self.alpha) * self.y
        return self.y


class OneEuroFilter:
    """
    Filtro One-Euro para matrizes de landmarks (N, 3).
    Ajusta dinamicamente a frequência de corte baseando-se na velocidade do movimento.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Filtra a matriz de coordenadas x de formato (21, 3).
        """
        now = time.time() if timestamp is None else timestamp

        if self.last_time is None:
            self.last_time = now
            return self.x_filter.filter(x)

        dt = now - self.last_time
        self.last_time = now

        # Evita divisão por zero ou dt aberrante
        if dt <= 1e-5:
            dt = 1.0 / 30.0
        rate = 1.0 / dt

        # Estimativa da velocidade (derivada dx)
        prev_x = self.x_filter.y
        if prev_x is None:
            dx = np.zeros_like(x)
        else:
            dx = (x - prev_x) * rate

        # Filtra a velocidade estimada
        alpha_d = self._alpha(rate, self.d_cutoff)
        edx = self.dx_filter.filter(dx, alpha_d)

        # Frequência de corte dinâmica proporcional à magnitude da velocidade
        speed = np.linalg.norm(edx, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(rate, np.mean(cutoff))

        return self.x_filter.filter(x, alpha)
