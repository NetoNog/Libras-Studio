"""
Configurações e parâmetros do Tradutor de Libras em Tempo Real.
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AppConfig:
    # Captura de Vídeo (Resolução Padrão HD 720p - Alta Definição)
    CAMERA_INDEX: int = 0
    TARGET_WIDTH: int = 1280               # 1280x720 (HD 16:9) | 1920x1080 (FHD) | 640x480 (SD)
    TARGET_HEIGHT: int = 720
    USE_DSHOW: bool = True                 # DirectShow no Windows (evita atraso de 80s do MSMF)
    FLIP_HORIZONTAL: bool = True
    MIN_FPS: int = 30


    # MediaPipe Hands (Bimanual: 2 Mãos Simultâneas)
    MAX_NUM_HANDS: int = 2
    MIN_DETECTION_CONFIDENCE: float = 0.6  # Entre 0.5 e 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.6   # Entre 0.5 e 0.7
    MODEL_PATH: str = "hand_landmarker.task"

    # Recursos Avançados de Interpretação de Libras
    ENABLE_FACE: bool = True               # Rastreamento de ponto de articulação no rosto
    ENABLE_SMART_PAUSE: bool = True        # Detecção de pausa automática para finalizar palavras
    SMART_PAUSE_SEC: float = 1.4           # Tempo sem mãos no campo para finalizar palavra
    ENABLE_SYNTAX_TRANSLATOR: bool = True  # Tradutor gramatical Libras -> Português fluente

    # Buffer de Composição de Texto e Debounce
    COMMIT_INTERVAL_SEC: float = 0.8       # Tempo sustentando sinal para confirmar (ágeis 0.8s)
    HISTORY_WINDOW_SIZE: int = 18          # Janela móvel de frames para cálculo de estabilidade
    MIN_STABILITY_THRESHOLD: float = 0.60  # Estabilidade mínima (60%) para acumular progresso
    AUTO_REPEAT_INTERVAL_SEC: float = 1.8  # Intervalo para repetir a mesma letra se mantida

    # Interface Visual (HUD) - Paleta de Cores BGR
    COLOR_BG_DARK: Tuple[int, int, int] = (20, 22, 28)
    COLOR_PANEL_BG: Tuple[int, int, int] = (32, 35, 45)
    COLOR_PANEL_BORDER: Tuple[int, int, int] = (70, 75, 95)
    COLOR_PRIMARY: Tuple[int, int, int] = (245, 175, 40)      # Azul Ciano / Dourado moderno
    COLOR_ACCENT: Tuple[int, int, int] = (80, 220, 100)       # Verde Neon para Confirmação
    COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)         # Branco
    COLOR_TEXT_DIM: Tuple[int, int, int] = (170, 175, 190)    # Cinza claro
    COLOR_WARNING: Tuple[int, int, int] = (40, 120, 245)      # Laranja de aviso
    COLOR_SKELETON_JOINT: Tuple[int, int, int] = (0, 220, 255) # Ciano brilhante
    COLOR_SKELETON_BONE: Tuple[int, int, int] = (255, 180, 50) # Azul profundo
    COLOR_BAR_FILL: Tuple[int, int, int] = (60, 210, 120)     # Barra de progresso verde

    # Áudio, Voz (TTS) e Predição
    ENABLE_AUDIO: bool = True              # Bips de confirmação
    ENABLE_TTS: bool = True                # Síntese de voz em Português
    ENABLE_PREDICTOR: bool = True          # Sugestões de autocompletar palavras

    # Letras do Alfabeto Estático e Dinâmico de Libras Suportadas (A-Z Completo)
    SUPPORTED_LETTERS: Tuple[str, ...] = (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'ESPAÇO', 'APAGAR'
    )

