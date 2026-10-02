"""
Interface Visual (HUD Pro) de Alto Desempenho e Estética de Software Comercial.
Renderiza painéis translúcidos com efeito Frosted Glass (cantos arredondados e desfoque),
tipografia moderna anti-serrilhada (Segoe UI), medidores de estabilidade,
modo de legendas cinematográficas flutuantes (Subtitle Mode) e animações de pulso/ripple.
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from classifier import ClassificationResult
from config import AppConfig
from hand_tracker import HandDetection
from text_buffer import TextBuffer
from ui_renderer import UIRenderer


class UIHUD:
    """
    Renderizador do Head-Up Display (HUD Pro) sobre o feed da webcam.
    """

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.renderer = UIRenderer()
        self.last_blink_time = time.time()
        self.cursor_visible = True
        self.commit_flash_time = 0.0
        self.last_committed_letter = ""
        self.last_commit_coords: Optional[Tuple[int, int]] = None

        # Modo de Exibição: False = HUD Completo/Técnico | True = Modo Legenda de Cinema
        self.subtitle_mode: bool = False

        # Notificações Toast no HUD
        self.toast_message = ""
        self.toast_expire_time = 0.0

    def toggle_subtitle_mode(self) -> bool:
        """Alterna entre o HUD Técnico Completo e o Modo Legendas Flutuantes."""
        self.subtitle_mode = not self.subtitle_mode
        mode_name = "LEGENDA DE CINEMA" if self.subtitle_mode else "HUD COMPLETO"
        self.trigger_toast(f"MODO VISUAL: {mode_name}")
        return self.subtitle_mode

    def trigger_toast(self, message: str, duration: float = 2.0):
        """Exibe uma notificação temporária flutuante de alto contraste."""
        self.toast_message = message
        self.toast_expire_time = time.time() + duration

    def trigger_commit_feedback(self, letter: str, coords: Optional[Tuple[int, int]] = None):
        """Ativa um efeito visual de brilho e onda de choque no HUD ao confirmar um sinal."""
        self.commit_flash_time = time.time()
        self.last_committed_letter = letter
        self.last_commit_coords = coords

    def draw_rounded_panel(self,
                           img: np.ndarray,
                           x: int, y: int, w: int, h: int,
                           bg_color: Tuple[int, int, int] = (16, 20, 28),
                           border_color: Tuple[int, int, int] = (60, 80, 115),
                           alpha: float = 0.88,
                           radius: int = 12) -> np.ndarray:
        """Desenha um painel translúcido elegante de alto contraste com cantos arredondados."""
        return self.renderer.draw_glass_panel(img, x, y, w, h, bg_color, border_color, alpha, radius)

    def render(self,
               frame: np.ndarray,
               detection: Optional[HandDetection],
               clf_result: Optional[ClassificationResult],
               text_buffer: TextBuffer,
               fps: float,
               suggestions: Optional[List[str]] = None,
               motion_trail: Optional[List[Tuple[int, int]]] = None,
               calibrator = None,
               all_detections: Optional[List[HandDetection]] = None,
               fluent_translation: Optional[str] = None) -> np.ndarray:
        """
        Renderiza todas as camadas de interface com estética profissional sobre o frame.
        """
        h, w, _ = frame.shape
        now = time.time()

        # Atualiza piscar do cursor
        if now - self.last_blink_time > 0.5:
            self.cursor_visible = not self.cursor_visible
            self.last_blink_time = now

        # Coletor de textos para renderização tipográfica anti-serrilhada em lote (Pillow)
        # Cada item: (texto, (x, y), font_size, (r, g, b), bold, shadow)
        texts_to_render: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int], bool, bool]] = []

        # 1. TRILHA DE MOVIMENTO CINEMÁTICO (SINAIS DINÂMICOS J / Z)
        if motion_trail and len(motion_trail) > 1:
            self._render_motion_trail(frame, motion_trail)

        # 2. ONDA DE CHOQUE / RIPPLE DE CONFIRMAÇÃO
        time_since_commit = now - self.commit_flash_time
        if time_since_commit < 0.6:
            ripple_center = self.last_commit_coords if self.last_commit_coords else (w // 2, h - 90)
            self.renderer.draw_ripple_effect(frame, ripple_center, time_since_commit / 0.6)

        # -----------------------------------------------------------------
        # SELEÇÃO DE MODO VISUAL: LEGENDA DE CINEMA vs HUD COMPLETO
        # -----------------------------------------------------------------
        if self.subtitle_mode:
            # MODO LEGENDA CINEMATOGRÁFICA (Visual minimalista, broadcast e videochamadas)
            self._render_cinematic_subtitles(frame, w, h, fluent_translation, text_buffer, texts_to_render)
        else:
            # MODO HUD COMPLETO (Dashboard analítico profissional de visão computacional)
            self._render_header(frame, w, detection, all_detections, fps, texts_to_render)

            if clf_result is not None:
                self._render_finger_dashboard(frame, clf_result, texts_to_render)

            if fluent_translation:
                self._render_fluent_translation_bar(frame, w, h, fluent_translation, texts_to_render)
            elif suggestions and len(suggestions) > 0:
                self._render_suggestions_bar(frame, w, h, suggestions, texts_to_render)

            self._render_bottom_hud(frame, w, h, clf_result, text_buffer, texts_to_render)
            self._render_shortcuts_footer(frame, w, h, texts_to_render)

        # OVERLAY DE CALIBRAÇÃO (SE ATIVO)
        if calibrator is not None and calibrator.is_active():
            self._render_calibration_overlay(frame, w, h, calibrator, texts_to_render)

        # NOTIFICAÇÕES TOAST FLUTUANTES
        if now < self.toast_expire_time:
            self._render_toast(frame, w, h, texts_to_render)

        # RENDERIZAÇÃO FINAL TIPOGRÁFICA ANTI-SERRILHADA EM LOTE (SEGOE UI)
        if texts_to_render:
            frame = self.renderer.render_texts_on_frame(frame, texts_to_render)

        return frame

    def _render_header(self,
                        frame: np.ndarray,
                        w: int,
                        detection: Optional[HandDetection],
                        all_detections: Optional[List[HandDetection]],
                        fps: float,
                        texts: List):
        """Barra superior com branding premium, FPS e dados das mãos."""
        header_h = 42
        self.draw_rounded_panel(frame, 10, 8, w - 20, header_h,
                                bg_color=(15, 18, 26), border_color=(45, 60, 88), alpha=0.90, radius=10)

        # Branding Principal (Tipografia Segoe UI Bold)
        texts.append(("TRADUTOR DE LIBRAS", (24, 18), 16, (255, 255, 255), True, True))
        texts.append(("IA PRO • BIMANUAL", (195, 21), 11, (80, 230, 120), True, True))

        # Status das Mãos
        if all_detections and len(all_detections) >= 2:
            hand_str = "MÃOS: 2 (BIMANUAL)"
            badge_color = (0, 230, 255)
        elif detection is not None:
            hand_pt = "DIR" if detection.handedness == "Right" else "ESQ"
            hand_str = f"MÃO: {hand_pt} ({int(detection.confidence * 100)}%)"
            badge_color = (60, 220, 100)
        else:
            hand_str = "MÃO: AGUARDANDO"
            badge_color = (150, 160, 180)

        texts.append((hand_str, (w - 250, 20), 12, badge_color, True, True))
        texts.append((f"{fps:.0f} FPS", (w - 65, 20), 12, (255, 210, 70), True, True))

    def _render_finger_dashboard(self, frame: np.ndarray, clf: ClassificationResult, texts: List):
        """Painel lateral compacto com o vetor dos 5 dedos em LEDs de alto contraste."""
        p_x, p_y = 10, 56
        p_w, p_h = 160, 114
        self.draw_rounded_panel(frame, p_x, p_y, p_w, p_h,
                                bg_color=(15, 18, 26), border_color=(40, 52, 75), alpha=0.86, radius=10)

        texts.append(("VETOR DE DEDOS", (p_x + 12, p_y + 10), 11, (160, 175, 200), True, True))

        labels = [("Polegar", clf.finger_states[0]),
                  ("Indicador", clf.finger_states[1]),
                  ("Médio", clf.finger_states[2]),
                  ("Anelar", clf.finger_states[3]),
                  ("Mínimo", clf.finger_states[4])]

        for i, (name, is_ext) in enumerate(labels):
            y_offset = p_y + 32 + (i * 15)
            led_color = (60, 220, 100) if is_ext else (65, 70, 85)
            cv2.circle(frame, (p_x + 18, y_offset + 5), 3, led_color, -1, cv2.LINE_AA)

            text_color = (255, 255, 255) if is_ext else (130, 135, 145)
            status_text = "ABERTO" if is_ext else "FECHADO"
            texts.append((f"{name}:", (p_x + 28, y_offset), 10, text_color, False, False))
            status_color = (80, 230, 120) if is_ext else (130, 135, 145)
            texts.append((status_text, (p_x + 95, y_offset), 9, status_color, True, False))

    def _render_bottom_hud(self,
                           frame: np.ndarray,
                           w: int, h: int,
                           clf: Optional[ClassificationResult],
                           text_buffer: TextBuffer,
                           texts: List):
        """Caixa principal inferior com sinal atual, medidor de estabilidade e transcrição."""
        hud_h = 140
        hud_y = h - hud_h - 24
        hud_x = 10
        hud_w = w - 20

        now = time.time()
        is_flash = (now - self.commit_flash_time) < 0.45

        border_col = (80, 255, 120) if is_flash else (45, 60, 85)
        bg_col = (20, 36, 28) if is_flash else (14, 17, 24)

        self.draw_rounded_panel(frame, hud_x, hud_y, hud_w, hud_h,
                                bg_color=bg_col, border_color=border_col, alpha=0.92, radius=14)

        # Coluna 1: Letra Reconhecida
        col1_w = 120
        cv2.line(frame, (hud_x + col1_w, hud_y + 12),
                 (hud_x + col1_w, hud_y + hud_h - 12), (38, 48, 68), 1, cv2.LINE_AA)

        texts.append(("SINAL ATUAL", (hud_x + 22, hud_y + 14), 11, (160, 175, 200), True, True))

        detected_char = clf.letter if (clf and clf.letter) else "-"
        char_color = (60, 230, 120) if detected_char != "-" else (100, 110, 125)
        font_size = 46 if len(detected_char) <= 2 else 20
        y_pos = hud_y + 40 if len(detected_char) <= 2 else hud_y + 55

        texts.append((detected_char, (hud_x + 35, y_pos), font_size, char_color, True, True))

        conf_pct = int(clf.confidence * 100) if clf else 0
        texts.append((f"Confiança: {conf_pct}%", (hud_x + 18, hud_y + 114), 10, (170, 180, 200), False, True))

        # Coluna 2: Barra de Progresso, Estabilidade e Texto
        col2_x = hud_x + col1_w + 16
        col2_w = hud_w - col1_w - 28

        stability_pct = int(text_buffer.current_stability * 100)
        st_color = (60, 220, 100) if stability_pct >= 60 else (50, 160, 240) if stability_pct > 25 else (130, 135, 145)

        texts.append(("ESTABILIDADE:", (col2_x, hud_y + 14), 11, (160, 175, 200), False, True))
        texts.append((f"{stability_pct}%", (col2_x + 105, hud_y + 13), 12, st_color, True, True))

        # Barra de Progresso Suave
        progress = text_buffer.current_progress
        bar_x = col2_x
        bar_y = hud_y + 34
        bar_w = col2_w - 10
        bar_h = 8

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (28, 34, 46), -1)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            fill_color = (60, 230, 120) if progress >= 0.95 else (245, 180, 40)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 68, 90), 1, cv2.LINE_AA)

        # Palavra Sendo Composta
        current_word = text_buffer.get_current_word()
        word_display = current_word if current_word else "(iniciando...)"
        texts.append(("PALAVRA:", (col2_x, hud_y + 56), 11, (160, 175, 200), False, True))
        texts.append((word_display, (col2_x + 75, hud_y + 55), 13, (255, 215, 70), True, True))

        # Caixa do Texto Acumulado (Console Glow)
        text_box_y = hud_y + 78
        text_box_h = 46
        self.draw_rounded_panel(frame, col2_x, text_box_y, bar_w, text_box_h,
                                bg_color=(10, 12, 16), border_color=(45, 58, 80), alpha=0.95, radius=8)

        display_str = text_buffer.get_display_text(max_length=32)
        if self.cursor_visible:
            display_str += "│"

        texts.append((f">  {display_str}", (col2_x + 12, text_box_y + 12), 17, (255, 255, 255), True, True))

    def _render_fluent_translation_bar(self, frame: np.ndarray, w: int, h: int, translation: str, texts: List):
        """Barra de tradução sintática fluente acima da caixa principal."""
        bar_h = 32
        bar_y = h - 140 - 24 - bar_h - 6
        bar_x = 10
        bar_w = w - 20

        self.draw_rounded_panel(frame, bar_x, bar_y, bar_w, bar_h,
                                bg_color=(15, 30, 22), border_color=(60, 210, 110), alpha=0.92, radius=8)

        texts.append(("PORTUGUÊS FLUENTE:", (bar_x + 14, bar_y + 8), 11, (80, 230, 120), True, True))
        clean_text = translation if len(translation) <= 52 else translation[:49] + "..."
        texts.append((f'"{clean_text}"', (bar_x + 175, bar_y + 7), 13, (255, 255, 255), True, True))

    def _render_suggestions_bar(self, frame: np.ndarray, w: int, h: int, suggestions: List[str], texts: List):
        """Barra de palavras sugeridas com atalhos interativos [1], [2], [3]."""
        bar_h = 30
        bar_y = h - 140 - 24 - bar_h - 6
        bar_x = 10
        bar_w = w - 20

        self.draw_rounded_panel(frame, bar_x, bar_y, bar_w, bar_h,
                                bg_color=(18, 22, 32), border_color=(50, 70, 100), alpha=0.90, radius=8)

        texts.append(("SUGESTÕES:", (bar_x + 14, bar_y + 8), 11, (160, 175, 200), False, True))

        offset_x = bar_x + 105
        for idx, word in enumerate(suggestions[:3]):
            key_tag = f"[{idx + 1}] "
            texts.append((key_tag, (offset_x, bar_y + 7), 11, (80, 230, 120), True, True))
            texts.append((word, (offset_x + 22, bar_y + 7), 12, (255, 255, 255), True, True))
            offset_x += int(len(word) * 11) + 40

    def _render_cinematic_subtitles(self,
                                    frame: np.ndarray,
                                    w: int, h: int,
                                    fluent_translation: Optional[str],
                                    text_buffer: TextBuffer,
                                    texts: List):
        """
        Modo Cinema / Legendas Flutuantes: visual limpo sem poluição técnica.
        Exibe uma legenda translúcida elegante no terço inferior, ideal para transmissões.
        """
        active_text = fluent_translation if (fluent_translation and fluent_translation.strip()) else text_buffer.current_text.strip()
        if not active_text:
            active_text = "Sinalize em Libras para gerar legendas em tempo real..."

        card_w = min(int(w * 0.92), 740)
        card_h = 58
        card_x = (w - card_w) // 2
        card_y = h - card_h - 28

        # Painel Frosted Glass arredondado
        self.draw_rounded_panel(frame, card_x, card_y, card_w, card_h,
                                bg_color=(12, 14, 20), border_color=(60, 220, 120), alpha=0.92, radius=16)

        texts.append(("• LEGENDA DE LIBRAS EM TEMPO REAL", (card_x + 20, card_y + 8), 9, (80, 230, 120), True, True))

        # Texto da legenda centralizado e destacado
        clean_text = active_text if len(active_text) <= 50 else active_text[:47] + "..."
        texts.append((clean_text, (card_x + 20, card_y + 24), 18, (255, 255, 255), True, True))

        # Indicador discreto de tecla para alternar
        texts.append(("[L] Modo Completo", (card_x + card_w - 110, card_y + 8), 9, (140, 150, 165), False, True))

    def _render_motion_trail(self, frame: np.ndarray, trail: List[Tuple[int, int]]):
        """Desenha a trilha cinemática de neon dos sinais com trajetória no ar."""
        n = len(trail)
        for i in range(1, n):
            alpha = i / n
            thickness = max(1, int(4 * alpha))
            color = (int(0 + 255 * alpha), int(220 * alpha), int(255))
            cv2.line(frame, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

    def _render_toast(self, frame: np.ndarray, w: int, h: int, texts: List):
        """Renderiza uma notificação toast flutuante no topo central."""
        card_w = int(len(self.toast_message) * 9.5) + 38
        card_h = 34
        card_x = (w - card_w) // 2
        card_y = 56

        self.draw_rounded_panel(frame, card_x, card_y, card_w, card_h,
                                bg_color=(20, 38, 26), border_color=(80, 230, 120), alpha=0.94, radius=10)

        texts.append((self.toast_message, (card_x + 18, card_y + 9), 12, (255, 255, 255), True, True))

    def _render_calibration_overlay(self, frame: np.ndarray, w: int, h: int, calibrator, texts: List):
        """Renderiza o painel central interativo de calibração biométrica."""
        instruction, progress = calibrator.get_hud_instruction()
        if not instruction:
            return

        card_w = int(w * 0.80)
        card_h = 90
        card_x = (w - card_w) // 2
        card_y = (h - card_h) // 2 - 20

        self.draw_rounded_panel(frame, card_x, card_y, card_w, card_h,
                                bg_color=(12, 16, 26), border_color=(0, 220, 255), alpha=0.95, radius=12)

        texts.append(("CALIBRAÇÃO ANATÔMICA GUIADA", (card_x + 22, card_y + 14), 13, (0, 220, 255), True, True))
        texts.append((instruction, (card_x + 22, card_y + 38), 12, (255, 255, 255), False, True))

        prog_w = card_w - 44
        prog_y = card_y + 66
        cv2.rectangle(frame, (card_x + 22, prog_y), (card_x + 22 + prog_w, prog_y + 8), (30, 40, 55), -1)
        fill_w = int(prog_w * progress)
        if fill_w > 0:
            cv2.rectangle(frame, (card_x + 22, prog_y), (card_x + 22 + fill_w, prog_y + 8), (0, 230, 180), -1)
        cv2.rectangle(frame, (card_x + 22, prog_y), (card_x + 22 + prog_w, prog_y + 8), (65, 85, 115), 1)

    def _render_shortcuts_footer(self, frame: np.ndarray, w: int, h: int, texts: List):
        """Legenda sutil de teclas de atalho no rodapé."""
        footer_y = h - 18
        shortcuts_text = "[ESPAÇO] Palavra | [1-3] Sugestões | [T] Copiar | [L] Legendas | [V] Falar | [K] Calibrar | [Q] Sair"
        texts.append((shortcuts_text, (14, footer_y), 9, (160, 170, 185), False, True))
