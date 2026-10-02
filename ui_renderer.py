"""
Motor Gráfico de Alta Fidelidade Visual (HUD Pro & Glassmorphism) para Libras.
Renderiza tipografia anti-serrilhada moderna (Segoe UI), painéis translúcidos
com desfoque de fundo (Frosted Glass), esqueletos anatômicos neon com gradientes
e efeitos visuais de pulso/partículas na confirmação de sinais.
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class UIRenderer:
    """
    Renderizador visual de última geração para interfaces gráficas sobre vídeo.
    Combina aceleração NumPy/OpenCV para gráficos vetoriais com a tipografia
    anti-aliased da biblioteca Pillow.
    """

    FONT_REGULAR_PATH = r"C:\Windows\Fonts\segoeui.ttf"
    FONT_BOLD_PATH = r"C:\Windows\Fonts\segoeuib.ttf"
    FONT_FALLBACK = r"C:\Windows\Fonts\arial.ttf"

    def __init__(self):
        self._font_cache: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}
        self._pill_mask_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    def get_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Obtém ou carrega fonte do cache com tamanho e peso especificados."""
        key = (size, bold)
        if key not in self._font_cache:
            path = self.FONT_BOLD_PATH if bold else self.FONT_REGULAR_PATH
            if not os.path.exists(path):
                path = self.FONT_FALLBACK
            try:
                self._font_cache[key] = ImageFont.truetype(path, size)
            except Exception:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def get_text_size(self, text: str, font_size: int, bold: bool = False) -> Tuple[int, int]:
        """Calcula largura e altura em pixels do texto renderizado."""
        font = self.get_font(font_size, bold)
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    def draw_glass_panel(self,
                         img: np.ndarray,
                         x: int, y: int, w: int, h: int,
                         bg_color: Tuple[int, int, int] = (16, 20, 28),
                         border_color: Tuple[int, int, int] = (60, 80, 115),
                         alpha: float = 0.88,
                         radius: int = 12,
                         frosted_blur: bool = True) -> np.ndarray:
        """
        Desenha um painel translúcido de efeito Frosted Glass (Vidro Fosco)
        com cantos arredondados, leve desfoque de fundo e borda iluminada.
        """
        img_h, img_w, _ = img.shape
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return img

        roi = img[y1:y2, x1:x2]

        # 1. Aplica desfoque de vidro fosco (Frosted Glass)
        if frosted_blur and roi.shape[0] > 6 and roi.shape[1] > 6:
            roi_blurred = cv2.blur(roi, (11, 11))
        else:
            roi_blurred = roi.copy()

        # 2. Mescla cor sólida translúcida
        overlay = np.full_like(roi_blurred, bg_color, dtype=np.uint8)
        tinted = cv2.addWeighted(overlay, alpha, roi_blurred, 1.0 - alpha, 0)

        # 3. Cria máscara com cantos arredondados
        mask = self._get_rounded_mask(x2 - x1, y2 - y1, radius)
        mask_3d = mask[:, :, np.newaxis]

        # 4. Aplica máscara na região da imagem
        img[y1:y2, x1:x2] = (tinted * mask_3d + roi * (1.0 - mask_3d)).astype(np.uint8)

        # 5. Desenha borda sutil iluminada
        self._draw_rounded_border(img, x1, y1, x2 - x1, y2 - y1, border_color, radius)

        return img

    def _get_rounded_mask(self, w: int, h: int, radius: int) -> np.ndarray:
        """Gera ou recupera máscara de cantos arredondados normalizada [0.0, 1.0]."""
        key = (w, h, radius)
        if key in self._pill_mask_cache:
            return self._pill_mask_cache[key]

        mask = np.zeros((h, w), dtype=np.float32)
        r = min(radius, w // 2, h // 2)

        # Retângulos centrais
        cv2.rectangle(mask, (r, 0), (w - r, h), 1.0, -1)
        cv2.rectangle(mask, (0, r), (w, h - r), 1.0, -1)

        # 4 cantos circulares anti-aliased
        if r > 0:
            cv2.circle(mask, (r, r), r, 1.0, -1, cv2.LINE_AA)
            cv2.circle(mask, (w - r - 1, r), r, 1.0, -1, cv2.LINE_AA)
            cv2.circle(mask, (r, h - r - 1), r, 1.0, -1, cv2.LINE_AA)
            cv2.circle(mask, (w - r - 1, h - r - 1), r, 1.0, -1, cv2.LINE_AA)

        if len(self._pill_mask_cache) > 40:
            self._pill_mask_cache.clear()
        self._pill_mask_cache[key] = mask
        return mask

    def _draw_rounded_border(self,
                             img: np.ndarray,
                             x: int, y: int, w: int, h: int,
                             color: Tuple[int, int, int],
                             radius: int):
        """Desenha uma linha de borda suave com cantos arredondados."""
        r = min(radius, w // 2, h // 2)
        # Linhas retas
        cv2.line(img, (x + r, y), (x + w - r, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, 1, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, 1, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, 1, cv2.LINE_AA)

        # Arcos nos 4 cantos
        if r > 0:
            cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, 1, cv2.LINE_AA)
            cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, 1, cv2.LINE_AA)
            cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, 1, cv2.LINE_AA)
            cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, 1, cv2.LINE_AA)

    def draw_cyber_skeleton(self,
                            img: np.ndarray,
                            landmarks_px: np.ndarray,
                            connections: List[Tuple[int, int]],
                            color_joint: Tuple[int, int, int] = (0, 230, 255),
                            color_bone: Tuple[int, int, int] = (255, 160, 40),
                            active_fingers: Optional[List[bool]] = None):
        """
        Desenha o esqueleto anatômico da mão com iluminação cyber-glow,
        gradientes nos ossos e anéis pulsantes nas pontas dos dedos.
        """
        # 1. Conexões dos Ossos (Linhas com efeito neon)
        for pt1_idx, pt2_idx in connections:
            pt1 = tuple(landmarks_px[pt1_idx])
            pt2 = tuple(landmarks_px[pt2_idx])
            # Linha de brilho externa (mais grossa e translúcida)
            cv2.line(img, pt1, pt2, (int(color_bone[0]*0.4), int(color_bone[1]*0.4), int(color_bone[2]*0.4)), 3, cv2.LINE_AA)
            # Linha de núcleo (branca/brilhante)
            cv2.line(img, pt1, pt2, color_bone, 1, cv2.LINE_AA)

        # 2. Juntas Anatômicas
        for idx, pt in enumerate(landmarks_px):
            center = tuple(pt)
            is_tip = idx in (4, 8, 12, 16, 20)
            radius = 5 if is_tip else 3
            # Halo suave
            cv2.circle(img, center, radius + 2, (30, 30, 40), -1, cv2.LINE_AA)
            # Centro brilhante
            cv2.circle(img, center, radius, color_joint, -1, cv2.LINE_AA)
            cv2.circle(img, center, 1, (255, 255, 255), -1, cv2.LINE_AA)

            # Anel concêntrico pulsante nas pontas ativas
            if is_tip and active_fingers:
                finger_idx = [4, 8, 12, 16, 20].index(idx)
                if finger_idx < len(active_fingers) and active_fingers[finger_idx]:
                    pulse_r = radius + int(3 + math.sin(time.time() * 8) * 2)
                    cv2.circle(img, center, pulse_r, (80, 240, 120), 1, cv2.LINE_AA)

    def draw_ripple_effect(self,
                           img: np.ndarray,
                           center: Tuple[int, int],
                           progress: float,
                           color: Tuple[int, int, int] = (80, 240, 120),
                           max_radius: int = 55):
        """Desenha onda de choque circular suave no ponto de confirmação."""
        if progress <= 0.0 or progress >= 1.0:
            return
        radius = int(max_radius * progress)
        alpha = 1.0 - progress
        glow_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        thickness = max(1, int(3 * (1.0 - progress)))
        cv2.circle(img, center, radius, glow_color, thickness, cv2.LINE_AA)

    def render_texts_on_frame(self,
                              img: np.ndarray,
                              text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int], bool, bool]]) -> np.ndarray:
        """
        Renderiza múltiplos textos com anti-aliasing tipográfico de alta resolução em lote.
        Cada item na lista: (texto, (x, y), tamanho_da_fonte, cor_rgb, negrito, sombra)
        """
        if not text_items:
            return img

        # Converte OpenCV BGR para PIL Image RGB
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for text, (x, y), size, color_rgb, bold, shadow in text_items:
            font = self.get_font(size, bold=bold)
            if shadow:
                # Sombra suave de alto contraste
                draw.text((x + 1, y + 1), text, font=font, fill=(10, 12, 18))
            draw.text((x, y), text, font=font, fill=color_rgb)

        # Converte de volta para OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
