"""
Módulo de Feedback Auditivo e Síntese de Voz (TTS) em Português do Brasil.
Executa sons de confirmação e vocalização de texto em threads separadas,
garantindo latência zero e sem interromper a taxa de quadros (FPS) da câmera.
"""

import queue
import sys
import threading
import time
from typing import Optional

try:
    import winsound
except ImportError:
    winsound = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class AudioFeedback:
    """
    Controlador de áudio não-bloqueante para bips táteis e sintetizador de voz PT-BR.
    """

    def __init__(self, enable_sound: bool = True, enable_tts: bool = True):
        self.enable_sound = enable_sound
        self.enable_tts = enable_tts
        self.is_muted = not enable_sound

        # Fila de mensagens para síntese de voz assíncrona
        self.speech_queue = queue.Queue()
        self.tts_thread: Optional[threading.Thread] = None
        self.running = True

        if self.enable_tts:
            self._init_tts_worker()

    def _init_tts_worker(self):
        """Inicializa a thread trabalhadora de Text-to-Speech."""
        self.tts_thread = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self.tts_thread.start()

    def _tts_worker_loop(self):
        """Loop contínuo em segundo plano para vocalizar palavras sem travar o vídeo."""
        if pyttsx3 is None:
            return

        try:
            engine = pyttsx3.init()
            # Ajusta taxa de fala e volume
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.95)

            # Seleciona voz em Português se disponível (ex.: Microsoft Maria Desktop)
            voices = engine.getProperty("voices")
            pt_voice_id = None
            for voice in voices:
                v_name = voice.name.lower()
                v_lang = str(getattr(voice, "languages", "")).lower()
                if "brazil" in v_name or "portuguese" in v_name or "pt" in v_lang or "maria" in v_name:
                    pt_voice_id = voice.id
                    break

            if pt_voice_id:
                engine.setProperty("voice", pt_voice_id)

            while self.running:
                try:
                    text = self.speech_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if text is None:
                    break

                if not self.is_muted and text.strip():
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        print(f"[AudioFeedback] Erro na síntese TTS: {err}")

                self.speech_queue.task_done()

        except Exception as e:
            print(f"[AudioFeedback] Não foi possível inicializar motor pyttsx3: {e}")

    def toggle_mute(self) -> bool:
        """Alterna estado de mudo. Retorna True se estiver mudo, False se desmutado."""
        self.is_muted = not self.is_muted
        return self.is_muted

    def play_commit_sound(self):
        """Bip sutil ascendente de confirmação (1200 Hz, 45 ms)."""
        if self.is_muted or winsound is None or not sys.platform.startswith("win"):
            return

        def _beep():
            try:
                winsound.Beep(1200, 45)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()

    def play_delete_sound(self):
        """Bip grave de exclusão / backspace (480 Hz, 60 ms)."""
        if self.is_muted or winsound is None or not sys.platform.startswith("win"):
            return

        def _beep():
            try:
                winsound.Beep(480, 60)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()

    def play_clear_sound(self):
        """Bip duplo de limpeza total do buffer."""
        if self.is_muted or winsound is None or not sys.platform.startswith("win"):
            return

        def _beep():
            try:
                winsound.Beep(600, 40)
                time.sleep(0.04)
                winsound.Beep(400, 50)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()

    def speak(self, text: str):
        """Enfileira frase ou palavra para vocalização em segundo plano."""
        if not self.enable_tts or self.is_muted or not text.strip():
            return
        self.speech_queue.put(text)

    def speak_word(self, word: str):
        """Fala uma única palavra finalizada (ignora comandos especiais)."""
        clean_word = word.strip().upper()
        if clean_word in ("", "ESPAÇO", "ESPACO", "APAGAR", "LIMPAR"):
            return
        self.speak(clean_word.lower())

    def stop(self):
        """Encerra a thread de áudio."""
        self.running = False
        self.speech_queue.put(None)
