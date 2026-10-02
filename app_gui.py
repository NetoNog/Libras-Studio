"""
Interface Desktop Profissional (Libras Studio Pro) desenvolvida com CustomTkinter.
Oferece um estúdio completo de acessibilidade com feed de vídeo em tempo real,
dashboard lateral com transcrição, sugestões de palavras clicáveis,
telemetria analítica (WPM e confiança), exportação de legendas (.SRT / .TXT) e controles de áudio.
"""

from datetime import datetime
import os
import sys
import time
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk

from audio_feedback import AudioFeedback
from calibrator import HandCalibrator
from classifier import LibrasClassifier
from config import AppConfig
from face_tracker import FaceTracker
from gesture_guide import create_gesture_guide_image
from hand_tracker import HandTracker
from lexical_classifier import LexicalClassifier
from profile_manager import extract_hand_features
from syntax_translator import SyntaxTranslator
from text_buffer import TextBuffer
from trajectory_tracker import TrajectoryTracker
from ui_hud import UIHUD
from video_pipeline import ThreadedCamera
from word_predictor import WordPredictor


class LibrasStudioApp(ctk.CTk):
    """
    Aplicação Desktop comercial com interface gráfica moderna integrada.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()

        self.cfg = config if config else AppConfig()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Libras Studio Pro • Reconhecimento em Tempo Real e Acessibilidade")
        self.geometry("1360x840")
        self.minsize(1100, 720)


        # -------------------------------------------------------------
        # 1. INICIALIZAÇÃO DOS MOTORES DE IA E PROCESSAMENTO
        # -------------------------------------------------------------
        self.camera = ThreadedCamera(
            camera_index=self.cfg.CAMERA_INDEX,
            width=self.cfg.TARGET_WIDTH,
            height=self.cfg.TARGET_HEIGHT,
            fps=self.cfg.MIN_FPS,
            use_dshow=self.cfg.USE_DSHOW
        ).start()

        self.tracker = HandTracker(
            max_hands=self.cfg.MAX_NUM_HANDS,
            min_detection_confidence=self.cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.cfg.MIN_TRACKING_CONFIDENCE,
            model_path=self.cfg.MODEL_PATH
        )
        self.classifier = LibrasClassifier()
        self.text_buffer = TextBuffer(
            commit_interval_sec=self.cfg.COMMIT_INTERVAL_SEC,
            history_size=self.cfg.HISTORY_WINDOW_SIZE,
            min_stability=self.cfg.MIN_STABILITY_THRESHOLD,
            auto_repeat_interval_sec=self.cfg.AUTO_REPEAT_INTERVAL_SEC
        )
        self.hud = UIHUD(self.cfg)
        self.audio = AudioFeedback(enable_sound=self.cfg.ENABLE_AUDIO, enable_tts=self.cfg.ENABLE_TTS)
        self.predictor = WordPredictor()
        self.trajectory = TrajectoryTracker()
        self.calibrator = HandCalibrator()
        self.face_tracker = FaceTracker() if self.cfg.ENABLE_FACE else None
        self.lexical_clf = LexicalClassifier(static_classifier=self.classifier)
        self.syntax_translator = SyntaxTranslator() if self.cfg.ENABLE_SYNTAX_TRANSLATOR else None

        # Histórico de legendas para exportação SRT
        self.transcript_history: List[dict] = []
        self.session_start_time = time.time()
        self.last_hand_seen_time = time.time()
        self.total_words_count = 0
        self.prev_time = time.time()
        self.fps_history = []
        self.current_suggestions = []
        self.current_fluent = ""

        # Janela do guia de sinais
        self.guide_window: Optional[ctk.CTkToplevel] = None

        # Estado de calibração biométrica e gravação de assinaturas
        self.calib_modal: Optional["HandCalibrationModal"] = None
        self.recording_sign_active: bool = False
        self.recording_sign_target: Optional[str] = None
        self.recording_sign_samples_count: int = 0

        # -------------------------------------------------------------
        # 2. CONSTRUÇÃO DO LAYOUT DA INTERFACE (GRID MODERNO)
        # -------------------------------------------------------------
        self.grid_columnconfigure(0, weight=3) # Vídeo à esquerda
        self.grid_columnconfigure(1, weight=2) # Dashboard à direita
        self.grid_rowconfigure(0, weight=1)

        self._build_video_panel()
        self._build_dashboard_sidebar()
        self._bind_keyboard_shortcuts()

        # Inicia loop de vídeo assíncrono do Tkinter
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(20, self._process_video_frame)

    def _build_video_panel(self):
        """Painel esquerdo contendo o feed de vídeo ao vivo."""
        self.video_frame = ctk.CTkFrame(self, corner_radius=14, fg_color="#12151e")
        self.video_frame.grid(row=0, column=0, padx=(14, 8), pady=14, sticky="nsew")
        self.video_frame.grid_rowconfigure(1, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        # Barra superior do vídeo (Header)
        top_bar = ctk.CTkFrame(self.video_frame, height=44, fg_color="transparent")
        top_bar.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="ew")

        branding_lbl = ctk.CTkLabel(
            top_bar,
            text="● LIBRAS STUDIO PRO",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color="#4ee482"
        )
        branding_lbl.pack(side="left")

        # Botão alternar Modo Legendas / HUD
        self.btn_subtitles = ctk.CTkButton(
            top_bar,
            text="🎬 Modo Legendas",
            width=130,
            height=30,
            fg_color="#202636",
            hover_color="#303b54",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._toggle_subtitles
        )
        self.btn_subtitles.pack(side="right", padx=(6, 0))

        # Botão Calibrar Mão (Biometria & Assinaturas)
        self.btn_calibrate = ctk.CTkButton(
            top_bar,
            text="🎯 Calibrar Mão",
            width=125,
            height=30,
            fg_color="#1d3b2e",
            hover_color="#285641",
            border_width=1,
            border_color="#4ee482",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=self._open_calibration_window
        )
        self.btn_calibrate.pack(side="right", padx=(6, 0))

        # Seletor de Resolução Dinâmico
        self.res_options = ["720p HD (1280x720)", "1080p FHD (1920x1080)", "480p SD (640x480)"]
        default_res_str = f"{self.cfg.TARGET_WIDTH}x{self.cfg.TARGET_HEIGHT}"
        initial_val = self.res_options[0]
        for opt in self.res_options:
            if default_res_str in opt:
                initial_val = opt
                break

        self.res_menu = ctk.CTkOptionMenu(
            top_bar,
            values=self.res_options,
            width=175,
            height=30,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#202636",
            button_color="#2c354a",
            button_hover_color="#3d4966",
            command=self._on_resolution_change
        )
        self.res_menu.set(initial_val)
        self.res_menu.pack(side="right", padx=(6, 8))


        # Canvas/Label de Renderização de Vídeo
        self.video_label = ctk.CTkLabel(self.video_frame, text="", corner_radius=12)
        self.video_label.grid(row=1, column=0, padx=12, pady=(4, 12), sticky="nsew")

    def _build_dashboard_sidebar(self):
        """Painel direito contendo o dashboard executivo com histórico e controles."""
        self.dash_frame = ctk.CTkFrame(self, corner_radius=14, fg_color="#181b26")
        self.dash_frame.grid(row=0, column=1, padx=(8, 14), pady=14, sticky="nsew")
        self.dash_frame.grid_rowconfigure(2, weight=1) # Histórico expansível
        self.dash_frame.grid_columnconfigure(0, weight=1)

        # 1. Card de Tradução Fluente (Destaque Superior)
        trans_card = ctk.CTkFrame(self.dash_frame, corner_radius=12, fg_color="#1e2330", border_width=1, border_color="#35405a")
        trans_card.grid(row=0, column=0, padx=14, pady=(14, 8), sticky="ew")

        trans_hdr = ctk.CTkLabel(
            trans_card,
            text="TRADUÇÃO SINTÁTICA EM PORTUGUÊS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#52e690"
        )
        trans_hdr.pack(anchor="w", padx=14, pady=(10, 2))

        self.lbl_fluent = ctk.CTkLabel(
            trans_card,
            text='"Sinalize para traduzir..."',
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color="#ffffff",
            wraplength=380,
            justify="left"
        )
        self.lbl_fluent.pack(anchor="w", padx=14, pady=(4, 10))

        # Ações do Card Fluente
        btn_box = ctk.CTkFrame(trans_card, fg_color="transparent")
        btn_box.pack(fill="x", padx=14, pady=(0, 10))

        self.btn_speak = ctk.CTkButton(
            btn_box,
            text="🔊 Ouvir Frase (V)",
            width=120,
            height=30,
            fg_color="#2b7fff",
            hover_color="#1e60c8",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=self._speak_current_sentence
        )
        self.btn_speak.pack(side="left", padx=(0, 6))

        self.btn_copy = ctk.CTkButton(
            btn_box,
            text="📋 Copiar (T)",
            width=100,
            height=30,
            fg_color="#2e374d",
            hover_color="#424f6e",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._copy_to_clipboard
        )
        self.btn_copy.pack(side="left")

        # 2. Barra de Sugestões de Palavras (Pills Clicáveis)
        sug_box = ctk.CTkFrame(self.dash_frame, fg_color="transparent")
        sug_box.grid(row=1, column=0, padx=14, pady=(4, 8), sticky="ew")

        sug_title = ctk.CTkLabel(
            sug_box,
            text="SUGESTÕES DE AUTOCOMPLETAR:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#9da6ba"
        )
        sug_title.pack(anchor="w", pady=(0, 4))

        self.pills_frame = ctk.CTkFrame(sug_box, fg_color="transparent")
        self.pills_frame.pack(fill="x")
        self.pill_buttons: List[ctk.CTkButton] = []
        for idx in range(3):
            btn = ctk.CTkButton(
                self.pills_frame,
                text=f"[{idx+1}] -",
                width=115,
                height=28,
                fg_color="#202738",
                hover_color="#2f80ed",
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                command=lambda i=idx: self._apply_suggestion_by_index(i)
            )
            btn.pack(side="left", padx=(0, 6))
            self.pill_buttons.append(btn)

        # 3. Histórico e Transcrição de Diálogo (Área Rolante)
        hist_title = ctk.CTkLabel(
            self.dash_frame,
            text="HISTÓRICO DE COMUNICAÇÃO:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#9da6ba"
        )
        hist_title.grid(row=2, column=0, padx=14, pady=(6, 2), sticky="nw")

        self.txt_history = ctk.CTkTextbox(
            self.dash_frame,
            corner_radius=10,
            fg_color="#12151e",
            text_color="#f0f2f8",
            font=ctk.CTkFont(family="Segoe UI", size=13)
        )
        self.txt_history.grid(row=3, column=0, padx=14, pady=(0, 8), sticky="nsew")

        # 4. Telemetria e Estatísticas de Acessibilidade
        stats_frame = ctk.CTkFrame(self.dash_frame, fg_color="#1b202c", corner_radius=10)
        stats_frame.grid(row=4, column=0, padx=14, pady=(0, 10), sticky="ew")

        self.lbl_wpm = ctk.CTkLabel(
            stats_frame,
            text="Velocidade: 0 PPM",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#4ee482"
        )
        self.lbl_wpm.pack(side="left", padx=14, pady=8)

        self.lbl_signs_count = ctk.CTkLabel(
            stats_frame,
            text="Sinais: 0",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#abb4c7"
        )
        self.lbl_signs_count.pack(side="left", padx=14, pady=8)

        # 5. Ações Rápidas de Exportação e Controle
        actions_frame = ctk.CTkFrame(self.dash_frame, fg_color="transparent")
        actions_frame.grid(row=5, column=0, padx=14, pady=(0, 14), sticky="ew")

        self.btn_export_txt = ctk.CTkButton(
            actions_frame,
            text="💾 Salvar TXT",
            width=100,
            height=32,
            fg_color="#242c3e",
            hover_color="#36435f",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            command=self._export_txt
        )
        self.btn_export_txt.pack(side="left", padx=(0, 6))

        self.btn_export_srt = ctk.CTkButton(
            actions_frame,
            text="🎬 Exportar .SRT",
            width=110,
            height=32,
            fg_color="#242c3e",
            hover_color="#36435f",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            command=self._export_srt
        )
        self.btn_export_srt.pack(side="left", padx=(0, 6))

        self.btn_guide = ctk.CTkButton(
            actions_frame,
            text="📖 Guia (H)",
            width=90,
            height=32,
            fg_color="#242c3e",
            hover_color="#36435f",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            command=self._toggle_guide_window
        )
        self.btn_guide.pack(side="left", padx=(0, 6))

        self.btn_clear = ctk.CTkButton(
            actions_frame,
            text="🗑️ Limpar",
            width=80,
            height=32,
            fg_color="#3a2024",
            hover_color="#63262e",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            command=self._clear_text
        )
        self.btn_clear.pack(side="right")

    def _process_video_frame(self):
        """Loop contínuo de processamento e atualização de tela."""
        ret, frame = self.camera.read()
        if ret and frame is not None:
            if self.cfg.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            actual_h, actual_w, _ = frame.shape

            # 1. Rastreamento Bimanual (até 2 mãos)
            detections = self.tracker.process(frame)
            primary_hand = detections[0] if len(detections) > 0 else None
            self.tracker.draw_hands(frame, detections)

            # 2. Ponto de Articulação Facial
            face_kps = None
            if self.face_tracker is not None:
                face_kps = self.face_tracker.process(frame)
                if face_kps is not None:
                    self.face_tracker.draw_landmarks(frame, face_kps, actual_w, actual_h)

            clf_result = None

            # 3. Classificação Léxica / Bimanual / Facial Prioritária
            lexical_res = self.lexical_clf.evaluate(hands=detections, face=face_kps)
            if lexical_res:
                lex_word, lex_conf, lex_type = lexical_res
                self.text_buffer.commit_word(lex_word)
                hand_coords = tuple(primary_hand.landmarks_pixel[8]) if primary_hand else None
                self.hud.trigger_commit_feedback(lex_word, coords=hand_coords)
                self.audio.play_commit_sound()
                self.audio.speak_word(lex_word)
                self.hud.trigger_toast(f"[{lex_type}] {lex_word}")
                self._record_transcript_entry(lex_word)

            elif primary_hand is not None:
                # 4. Classificação Estática Canônica
                clf_result = self.classifier.classify(primary_hand)

                # 5. Sinais com Trajetória Dinâmica (J e Z)
                dynamic_res = self.trajectory.update(primary_hand, clf_result.finger_states if clf_result else None)
                if dynamic_res:
                    dyn_letter, dyn_conf = dynamic_res
                    self.text_buffer._apply_commit(dyn_letter)
                    self.hud.trigger_commit_feedback(dyn_letter)
                    self.hud.trigger_toast(f"DINÂMICO: {dyn_letter}")
                    self.audio.play_commit_sound()
                    # 6. Composição de Datilologia
                    committed = self.text_buffer.update(
                        clf_result.letter,
                        confidence=clf_result.confidence,
                        is_transition=clf_result.is_transition
                    )
                    if committed:
                        hand_coords = tuple(primary_hand.landmarks_pixel[8])
                        self.hud.trigger_commit_feedback(committed, coords=hand_coords)
                        if committed == "APAGAR":
                            self.audio.play_delete_sound()
                        elif committed == "ESPAÇO":
                            self.audio.play_commit_sound()
                            last_w = self.text_buffer.get_current_word()
                            if last_w:
                                self.audio.speak_word(last_w)
                                self._record_transcript_entry(last_w)
                        else:
                            self.audio.play_commit_sound()

            # 6.5. Processamento de Calibração Ativa (Anatômica ou Assinatura de Sinal)
            if self.calibrator.is_active():
                calib_profile = self.calibrator.update(primary_hand)
                instr, prog = self.calibrator.get_hud_instruction()
                if self.calib_modal and self.calib_modal.winfo_exists():
                    self.calib_modal.update_express_progress(instr, prog)
                if calib_profile:
                    self.classifier.set_calibration_profile(calib_profile)
                    self.audio.play_commit_sound()
                    self.hud.trigger_toast("CALIBRAÇÃO ANATÔMICA CONCLUÍDA!")
                    if self.calib_modal and self.calib_modal.winfo_exists():
                        self.calib_modal.on_anatomical_calib_done(calib_profile)

            elif self.recording_sign_active:
                if primary_hand is not None:
                    c_lm, _, _ = self.classifier._compute_canonical_frame(primary_hand.landmarks_norm)
                    feat = extract_hand_features(c_lm, primary_hand.landmarks_norm)
                    self.recording_sign_samples_count = self.classifier.profile_manager.add_recording_sample(feat)
                    if self.calib_modal and self.calib_modal.winfo_exists():
                        self.calib_modal.update_recording_progress(self.recording_sign_samples_count, 25)
                    if self.recording_sign_samples_count >= 25:
                        self.classifier.profile_manager.finish_recording()
                        saved_sign = self.recording_sign_target
                        self.recording_sign_active = False
                        self.audio.play_commit_sound()
                        self.hud.trigger_toast(f"ASSINATURA '{saved_sign}' SALVA!")
                        if self.calib_modal and self.calib_modal.winfo_exists():
                            self.calib_modal.on_sign_calib_done(saved_sign)

            # 7. Detecção Inteligente de Pausa
            now = time.time()
            if len(detections) > 0:
                self.last_hand_seen_time = now
            else:
                self.text_buffer.update(None)
                self.trajectory.update(None, None)
                if self.cfg.ENABLE_SMART_PAUSE and (now - self.last_hand_seen_time) > self.cfg.SMART_PAUSE_SEC:
                    curr_word = self.text_buffer.get_current_word()
                    if curr_word and not self.text_buffer.current_text.endswith(" "):
                        self.text_buffer.current_text += " "
                        self.audio.play_commit_sound()
                        self.audio.speak_word(curr_word)
                        self.hud.trigger_toast(f"PAUSA: '{curr_word}'")
                        self._record_transcript_entry(curr_word)
                        self.last_hand_seen_time = now

            # 8. Tradução Sintática Fluente
            fluent_translation = None
            if self.syntax_translator is not None:
                tokens = [w for w in self.text_buffer.current_text.strip().split(" ") if w]
                if tokens:
                    fluent_translation = self.syntax_translator.translate_sequence(tokens)
            self.current_fluent = fluent_translation if fluent_translation else self.text_buffer.current_text

            # 9. Autocompletar
            prefix = self.text_buffer.get_current_word()
            suggestions = self.predictor.get_suggestions(prefix, max_suggestions=3) if self.cfg.ENABLE_PREDICTOR else []
            self.current_suggestions = suggestions

            # 10. Cálculo de FPS
            dt = now - self.prev_time
            self.prev_time = now
            curr_fps = 1.0 / dt if dt > 0 else 30.0
            self.fps_history.append(curr_fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

            # 11. Renderização do HUD Pro
            frame = self.hud.render(
                frame=frame,
                detection=primary_hand,
                clf_result=clf_result,
                text_buffer=self.text_buffer,
                fps=avg_fps,
                suggestions=suggestions,
                motion_trail=self.trajectory.get_trail_pixels(),
                calibrator=self.calibrator,
                all_detections=detections,
                fluent_translation=fluent_translation
            )

            # Atualização do Dashboard e Feed de Vídeo na Tela
            self._update_gui_dashboard(frame, fluent_translation, suggestions)

        self.after(16, self._process_video_frame)

    def _on_resolution_change(self, selected_option: str):
        """Alterna dinamicamente a resolução da webcam em tempo de execução."""
        if "1920x1080" in selected_option:
            w, h = 1920, 1080
        elif "1280x720" in selected_option:
            w, h = 1280, 720
        else:
            w, h = 640, 480

        self.cfg.TARGET_WIDTH = w
        self.cfg.TARGET_HEIGHT = h
        actual_w, actual_h = self.camera.set_resolution(w, h)
        self.hud.trigger_toast(f"RESOLUÇÃO: {actual_w}x{actual_h}")
        print(f"[Libras Studio] Resolução alterada para {actual_w}x{actual_h}")

    def _update_gui_dashboard(self, frame_bgr: np.ndarray, fluent: Optional[str], suggestions: List[str]):
        """Atualiza a imagem no label do Tkinter com escala dinâmica proporcional e os widgets do dashboard."""
        frame_h, frame_w = frame_bgr.shape[:2]

        # Calcula o tamanho ideal de exibição baseado no espaço disponível na janela
        avail_w = self.video_frame.winfo_width() - 28
        avail_h = self.video_frame.winfo_height() - 72

        if avail_w > 120 and avail_h > 120:
            aspect = frame_w / frame_h
            disp_w = avail_w
            disp_h = int(disp_w / aspect)
            if disp_h > avail_h:
                disp_h = avail_h
                disp_w = int(disp_h * aspect)
            disp_w = max(480, disp_w)
            disp_h = max(360, disp_h)
        else:
            disp_w = 854
            disp_h = 480

        interp = cv2.INTER_AREA if (frame_w > disp_w) else cv2.INTER_LINEAR
        frame_resized = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=interp)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))
        self.video_label.configure(image=ctk_img)


        # Atualiza Card Fluente
        if fluent and fluent.strip():
            self.lbl_fluent.configure(text=f'"{fluent}"')
        elif self.text_buffer.current_text.strip():
            self.lbl_fluent.configure(text=f'"{self.text_buffer.current_text}"')
        else:
            self.lbl_fluent.configure(text='"Sinalize para traduzir..."')

        # Atualiza Pills de Sugestões
        for idx in range(3):
            if idx < len(suggestions):
                word = suggestions[idx]
                self.pill_buttons[idx].configure(text=f"[{idx+1}] {word}", state="normal")
            else:
                self.pill_buttons[idx].configure(text=f"[{idx+1}] -", state="disabled")

        # Telemetria: Palavras Por Minuto (WPM)
        elapsed_min = max(0.1, (time.time() - self.session_start_time) / 60.0)
        wpm = int(self.total_words_count / elapsed_min)
        self.lbl_wpm.configure(text=f"Velocidade: {wpm} PPM")
        self.lbl_signs_count.configure(text=f"Sinais Concluídos: {self.total_words_count}")

    def _record_transcript_entry(self, word: str):
        """Registra a palavra no histórico com timestamp para exportação de legendas."""
        now = time.time()
        self.total_words_count += 1
        time_str = datetime.now().strftime("%H:%M:%S")
        entry_text = f"[{time_str}] {word}\n"
        self.txt_history.insert("end", entry_text)
        self.txt_history.see("end")

        self.transcript_history.append({
            "timestamp": now,
            "time_str": time_str,
            "word": word,
            "full_sentence": self.current_fluent
        })

    def _apply_suggestion_by_index(self, index: int):
        """Aplica a sugestão do autocompletar ao clicar na pill correspondente."""
        if index < len(self.current_suggestions):
            selected = self.current_suggestions[index]
            self.text_buffer.current_text = self.predictor.apply_suggestion(self.text_buffer.current_text, selected)
            self.audio.play_commit_sound()
            self.audio.speak_word(selected)
            self.hud.trigger_toast(f"PALAVRA: {selected}")
            self._record_transcript_entry(selected)

    def _speak_current_sentence(self):
        """Pronuncia a frase inteira atual em voz alta via TTS."""
        phrase = self.current_fluent if self.current_fluent.strip() else self.text_buffer.current_text.strip()
        if phrase:
            self.audio.speak(phrase)
            self.hud.trigger_toast("FALANDO FRASE...")

    def _copy_to_clipboard(self):
        """Copia a tradução atual para o Clipboard do Windows."""
        try:
            import pyperclip
            text = self.current_fluent if self.current_fluent.strip() else self.text_buffer.current_text.strip()
            pyperclip.copy(text)
            self.audio.play_commit_sound()
            self.hud.trigger_toast("COPIADO PARA O CLIPBOARD!")
        except Exception as err:
            print(f"[ERRO] Falha ao copiar: {err}")

    def _clear_text(self):
        """Limpa o buffer de texto atual."""
        self.text_buffer.clear()
        self.audio.play_clear_sound()
        self.hud.trigger_toast("TEXTO LIMPO")

    def _toggle_subtitles(self):
        """Alterna entre o modo de legendas e o HUD completo."""
        is_sub = self.hud.toggle_subtitle_mode()
        self.btn_subtitles.configure(text="📊 HUD Completo" if is_sub else "🎬 Modo Legendas")

    def _toggle_guide_window(self):
        """Abre janela pop-up moderna do guia visual de sinais."""
        if self.guide_window is not None and self.guide_window.winfo_exists():
            self.guide_window.focus()
            return

        self.guide_window = ctk.CTkToplevel(self)
        self.guide_window.title("Guia Visual e Dicionário Anatômico de Libras")
        self.guide_window.geometry("1060x720")

        guide_img_bgr = create_gesture_guide_image()
        guide_rgb = cv2.cvtColor(guide_img_bgr, cv2.COLOR_BGR2RGB)
        pil_guide = Image.fromarray(guide_rgb)
        ctk_guide_img = ctk.CTkImage(light_image=pil_guide, dark_image=pil_guide, size=(1040, 680))

        lbl = ctk.CTkLabel(self.guide_window, text="", image=ctk_guide_img)
        lbl.pack(padx=10, pady=10, fill="both", expand=True)

    def _export_txt(self):
        """Exporta todo o histórico de comunicação para arquivo de texto."""
        content = self.txt_history.get("1.0", "end").strip()
        if not content:
            self.hud.trigger_toast("NENHUM TEXTO PARA EXPORTAR")
            return
        filename = f"transcricao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== TRANSCRIÇÃO DE LIBRAS (LIBRAS STUDIO PRO) ===\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            f.write(content + "\n")
        self.audio.play_commit_sound()
        self.hud.trigger_toast(f"SALVO: {filename}")

    def _export_srt(self):
        """Exporta o histórico no formato universal de legendas .SRT."""
        if not self.transcript_history:
            self.hud.trigger_toast("HISTÓRICO VAZIO")
            return

        filename = f"legendas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        t0 = self.session_start_time

        with open(filename, "w", encoding="utf-8") as f:
            for i, entry in enumerate(self.transcript_history, 1):
                start_sec = max(0.0, entry["timestamp"] - t0)
                end_sec = start_sec + 2.5 # Duração padrão de exibição da legenda

                def sec_to_srt(s):
                    h = int(s // 3600)
                    m = int((s % 3600) // 60)
                    sec = int(s % 60)
                    ms = int((s - int(s)) * 1000)
                    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

                f.write(f"{i}\n")
                f.write(f"{sec_to_srt(start_sec)} --> {sec_to_srt(end_sec)}\n")
                f.write(f"{entry['word']}\n\n")

        self.audio.play_commit_sound()
        self.hud.trigger_toast(f"LEGENDAS SRT SALVAS: {filename}")

    def _bind_keyboard_shortcuts(self):
        """Mapeia os atalhos de teclado do estúdio."""
        self.bind("<Key-1>", lambda e: self._apply_suggestion_by_index(0))
        self.bind("<Key-2>", lambda e: self._apply_suggestion_by_index(1))
        self.bind("<Key-3>", lambda e: self._apply_suggestion_by_index(2))
        self.bind("<Key-t>", lambda e: self._copy_to_clipboard())
        self.bind("<Key-T>", lambda e: self._copy_to_clipboard())
        self.bind("<Key-v>", lambda e: self._speak_current_sentence())
        self.bind("<Key-V>", lambda e: self._speak_current_sentence())
        self.bind("<Key-c>", lambda e: self._clear_text())
        self.bind("<Key-C>", lambda e: self._clear_text())
        self.bind("<Key-l>", lambda e: self._toggle_subtitles())
        self.bind("<Key-L>", lambda e: self._toggle_subtitles())
        self.bind("<Key-h>", lambda e: self._toggle_guide_window())
        self.bind("<Key-H>", lambda e: self._toggle_guide_window())
        self.bind("<space>", lambda e: self.text_buffer._apply_commit("ESPAÇO"))
        self.bind("<BackSpace>", lambda e: self.text_buffer.backspace())

    def _open_calibration_window(self):
        """Abre ou focaliza a central modal de calibração biométrica."""
        if self.calib_modal is not None and self.calib_modal.winfo_exists():
            self.calib_modal.focus()
            return
        self.calib_modal = HandCalibrationModal(self)

    def _on_close(self):
        """Finalização graciosa de todos os recursos da aplicação."""
        print("[StudioApp] Encerrando estúdio...")
        self.camera.release()
        self.tracker.release()
        self.audio.stop()
        self.destroy()
        sys.exit(0)


class HandCalibrationModal(ctk.CTkToplevel):
    """
    Central Modal de Calibração Biomecânica e Assinaturas Vetoriais de Libras.
    Permite calibrar a anatomia específica da mão do usuário e gravar amostras
    personalizadas para qualquer letra do alfabeto com correspondência contínua.
    """

    AVAILABLE_SIGNS = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
        "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "ESPAÇO"
    ]

    def __init__(self, parent: LibrasStudioApp):
        super().__init__(parent)
        self.app = parent
        self.title("🎯 Central de Calibração Biomecânica - Libras Studio Pro")
        self.geometry("640x590")
        self.resizable(False, False)
        self.configure(fg_color="#12151e")
        self.transient(parent)
        self.grab_set()

        self.selected_sign = self.AVAILABLE_SIGNS[0]
        self._build_ui()
        self._update_sign_badge(self.selected_sign)
        self._refresh_calibrated_summary()

    def _build_ui(self):
        """Constrói a interface moderna da central de calibração."""
        # 1. Cabeçalho
        hdr_frame = ctk.CTkFrame(self, fg_color="transparent")
        hdr_frame.pack(fill="x", padx=20, pady=(18, 10))

        title_lbl = ctk.CTkLabel(
            hdr_frame,
            text="● CENTRAL DE CALIBRAÇÃO BIOMÉTRICA",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color="#4ee482"
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ctk.CTkLabel(
            hdr_frame,
            text="Adapte o motor de Libras à anatomia exata da sua mão para precisão impecável.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#9da6ba"
        )
        sub_lbl.pack(anchor="w", pady=(2, 0))

        # 2. Card 1: Calibração Anatômica Expressa (5 seg)
        card1 = ctk.CTkFrame(self, corner_radius=12, fg_color="#181c28", border_width=1, border_color="#293246")
        card1.pack(fill="x", padx=20, pady=8)

        c1_title = ctk.CTkLabel(
            card1,
            text="⚡ CALIBRAÇÃO ANATÔMICA EXPRESSA (5 SEGUNDOS)",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#52e690"
        )
        c1_title.pack(anchor="w", padx=16, pady=(12, 2))

        c1_desc = ctk.CTkLabel(
            card1,
            text="Mede proporções da palma, abertura dos dedos e flexão para mãos adultas ou infantis.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#8c97af"
        )
        c1_desc.pack(anchor="w", padx=16, pady=(0, 6))

        self.lbl_express_status = ctk.CTkLabel(
            card1,
            text="Status: Limiares padrão de fábrica ativos",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#8c97af"
        )
        self.lbl_express_status.pack(anchor="w", padx=16, pady=(0, 4))

        self.lbl_express_step = ctk.CTkLabel(
            card1,
            text="Clique em Iniciar e siga as instruções na câmera (Mão Aberta -> Punho Fechado).",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#d0d7e6"
        )
        self.lbl_express_step.pack(anchor="w", padx=16, pady=(0, 6))

        self.prog_express = ctk.CTkProgressBar(card1, height=8, corner_radius=4, progress_color="#4ee482")
        self.prog_express.pack(fill="x", padx=16, pady=(0, 10))
        self.prog_express.set(0.0)

        self.btn_start_express = ctk.CTkButton(
            card1,
            text="▶ Iniciar Calibração Expressa (5s)",
            height=32,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color="#1f538d",
            hover_color="#163f6c",
            command=self._start_express_calib
        )
        self.btn_start_express.pack(fill="x", padx=16, pady=(0, 14))

        # 3. Card 2: Assinaturas Vetoriais por Sinal
        card2 = ctk.CTkFrame(self, corner_radius=12, fg_color="#181c28", border_width=1, border_color="#293246")
        card2.pack(fill="x", padx=20, pady=8)

        c2_title = ctk.CTkLabel(
            card2,
            text="🔤 ASSINATURAS PERSONALIZADAS POR SINAL (A-Z)",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#64a5ff"
        )
        c2_title.pack(anchor="w", padx=16, pady=(12, 2))

        c2_desc = ctk.CTkLabel(
            card2,
            text="Grave como você faz uma letra específica. O sistema gerará uma assinatura vetorial biométrica.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#8c97af"
        )
        c2_desc.pack(anchor="w", padx=16, pady=(0, 8))

        row_sel = ctk.CTkFrame(card2, fg_color="transparent")
        row_sel.pack(fill="x", padx=16, pady=(0, 8))

        lbl_s = ctk.CTkLabel(row_sel, text="Letra/Sinal:", font=ctk.CTkFont(family="Segoe UI", size=12))
        lbl_s.pack(side="left", padx=(0, 8))

        self.sign_menu = ctk.CTkOptionMenu(
            row_sel,
            values=self.AVAILABLE_SIGNS,
            width=110,
            height=30,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color="#202738",
            button_color="#2c354a",
            button_hover_color="#3d4966",
            command=self._on_sign_selected
        )
        self.sign_menu.set(self.selected_sign)
        self.sign_menu.pack(side="left", padx=(0, 12))

        self.lbl_sign_status = ctk.CTkLabel(
            row_sel,
            text="● Padrão de Referência",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#9da6ba"
        )
        self.lbl_sign_status.pack(side="left")

        self.prog_sign = ctk.CTkProgressBar(card2, height=8, corner_radius=4, progress_color="#2b7fff")
        self.prog_sign.pack(fill="x", padx=16, pady=(0, 10))
        self.prog_sign.set(0.0)

        # Botões de Ação do Sinal
        btn_box = ctk.CTkFrame(card2, fg_color="transparent")
        btn_box.pack(fill="x", padx=16, pady=(0, 8))

        self.btn_record_sign = ctk.CTkButton(
            btn_box,
            text="📸 Gravar Assinatura (2s)",
            height=32,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color="#2b7fff",
            hover_color="#1e60c8",
            command=self._start_sign_recording
        )
        self.btn_record_sign.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_reset_sign = ctk.CTkButton(
            btn_box,
            text="🗑️ Redefinir",
            height=32,
            width=100,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#2a3244",
            hover_color="#3a455c",
            command=self._reset_selected_sign
        )
        self.btn_reset_sign.pack(side="left")

        self.lbl_calibrated_summary = ctk.CTkLabel(
            card2,
            text="Carregando resumo de sinais...",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#7b869c",
            wraplength=580,
            justify="left"
        )
        self.lbl_calibrated_summary.pack(anchor="w", padx=16, pady=(2, 12))

        # 4. Rodapé
        ftr_frame = ctk.CTkFrame(self, fg_color="transparent")
        ftr_frame.pack(fill="x", padx=20, pady=(6, 14))

        btn_close = ctk.CTkButton(
            ftr_frame,
            text="Fechar Central de Calibração",
            height=34,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#232938",
            hover_color="#333c52",
            command=self.destroy
        )
        btn_close.pack(fill="x")

    def _start_express_calib(self):
        """Dispara a calibração anatômica expressa de 5 segundos."""
        self.app.calibrator.start_calibration()
        self.btn_start_express.configure(state="disabled", text="⏳ Calibrando (mantenha a mão no enquadramento)...")
        self.lbl_express_step.configure(text="Etapa 1/2: Mão aberta e espalmada...")
        self.prog_express.set(0.0)

    def update_express_progress(self, instruction: str, progress: float):
        """Atualiza a barra e texto da calibração expressa em tempo real."""
        self.lbl_express_step.configure(text=instruction)
        self.prog_express.set(progress)

    def on_anatomical_calib_done(self, profile):
        """Callback acionado quando a calibração expressa é concluída."""
        self.lbl_express_status.configure(
            text=f"✅ Calibrado! Abertura={profile.finger_spread_threshold:.2f}, Extensão={profile.finger_extension_threshold:.2f}",
            text_color="#4ee482"
        )
        self.lbl_express_step.configure(text="✅ Calibração anatômica concluída com sucesso!")
        self.prog_express.set(1.0)
        self.btn_start_express.configure(state="normal", text="🔄 Recalibrar Expressa (5s)")

    def _on_sign_selected(self, sign: str):
        """Alterna o sinal selecionado para visualização/gravação."""
        self.selected_sign = sign
        self._update_sign_badge(sign)
        self.prog_sign.set(0.0)

    def _update_sign_badge(self, sign: str):
        """Atualiza o badge de status do sinal selecionado."""
        is_calib = self.app.classifier.profile_manager.is_calibrated(sign)
        if is_calib:
            self.lbl_sign_status.configure(text="● 🟢 Personalizado para Você", text_color="#4ee482")
        else:
            self.lbl_sign_status.configure(text="● ⚪ Padrão de Referência", text_color="#9da6ba")

    def _start_sign_recording(self):
        """Inicia a gravação de 25 amostras para a letra selecionada."""
        sign = self.selected_sign
        self.app.classifier.profile_manager.start_recording(sign)
        self.app.recording_sign_active = True
        self.app.recording_sign_target = sign
        self.app.recording_sign_samples_count = 0
        self.prog_sign.set(0.0)
        self.btn_record_sign.configure(state="disabled", text=f"📸 Gravando '{sign}' (mantenha a mão fixa)...")
        self.lbl_sign_status.configure(text="● 🟡 Capturando amostras da câmera...", text_color="#f1c40f")

    def update_recording_progress(self, current: int, total: int):
        """Atualiza o progresso visual de captura de amostras."""
        ratio = min(1.0, current / total)
        self.prog_sign.set(ratio)
        self.lbl_sign_status.configure(text=f"● 🟡 Capturando amostras: {current}/{total}", text_color="#f1c40f")

    def on_sign_calib_done(self, sign: str):
        """Callback acionado ao término da gravação da assinatura do sinal."""
        self.prog_sign.set(1.0)
        self.btn_record_sign.configure(state="normal", text="📸 Gravar Assinatura (2s)")
        self._update_sign_badge(sign)
        self._refresh_calibrated_summary()

    def _reset_selected_sign(self):
        """Remove a assinatura personalizada da letra selecionada."""
        sign = self.selected_sign
        self.app.classifier.profile_manager.reset_sign(sign)
        self.prog_sign.set(0.0)
        self._update_sign_badge(sign)
        self._refresh_calibrated_summary()

    def _refresh_calibrated_summary(self):
        """Atualiza a lista textual de todos os sinais calibrados pelo usuário."""
        calib_list = self.app.classifier.profile_manager.get_calibrated_signs()
        if calib_list:
            self.lbl_calibrated_summary.configure(
                text=f"Sinais com biometria personalizada ({len(calib_list)}): " + ", ".join(calib_list),
                text_color="#a0aec0"
            )
        else:
            self.lbl_calibrated_summary.configure(
                text="Nenhum sinal com biometria personalizada salvo ainda. Use o botão acima para gravar.",
                text_color="#718096"
            )


def run_studio(config: Optional[AppConfig] = None):
    """Ponto de entrada para execução da aplicação desktop."""
    app = LibrasStudioApp(config=config)
    app.mainloop()


if __name__ == "__main__":
    run_studio()
