"""
Aplicação Principal: Tradutor de Libras em Tempo Real.
Captura webcam, rastreia landmarks via MediaPipe, infere letras do alfabeto
através de heurísticas geométricas vetoriais e compõe palavras no HUD.
"""

import argparse
import sys
import time
import cv2
import numpy as np

from audio_feedback import AudioFeedback
from calibrator import HandCalibrator
from classifier import LibrasClassifier
from config import AppConfig
from face_tracker import FaceTracker
from gesture_guide import create_gesture_guide_image
from hand_tracker import HandTracker
from lexical_classifier import LexicalClassifier
from syntax_translator import SyntaxTranslator
from text_buffer import TextBuffer
from trajectory_tracker import TrajectoryTracker
from ui_hud import UIHUD
from word_predictor import WordPredictor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tradutor de Libras em Tempo Real via Webcam")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (padrão: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Largura da captura (padrão: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Altura da captura (padrão: 720)")
    parser.add_argument("--interval", type=float, default=None, help="Intervalo de sustentação para confirmação (segundos)")
    parser.add_argument("--guide", action="store_true", help="Abrir janela do guia de sinais na inicialização")
    parser.add_argument("--opencv", action="store_true", help="Executar no modo clássico de janela única OpenCV")
    return parser.parse_args()


def open_camera(camera_index: int, use_dshow: bool = True) -> cv2.VideoCapture:
    """Abre a câmera utilizando DirectShow no Windows para evitar o atraso de ~80s do MSMF."""
    if use_dshow and sys.platform.startswith("win"):
        print(f"[*] Conectando à câmera {camera_index} via DirectShow (Windows)...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        print("[AVISO] DirectShow não abriu. Tentando backend padrão...")

    print(f"[*] Conectando à câmera {camera_index} via backend padrão...")
    return cv2.VideoCapture(camera_index)


def main():
    args = parse_arguments()
    config = AppConfig()
    config.CAMERA_INDEX = args.camera
    config.TARGET_WIDTH = args.width
    config.TARGET_HEIGHT = args.height
    if args.interval is not None:
        config.COMMIT_INTERVAL_SEC = args.interval

    # 1. Modo Estúdio Profissional Desktop (Padrão)
    if not args.opencv:
        from app_gui import run_studio
        print("[*] Iniciando Libras Studio Pro com Dashboard Executivo Integrado...")
        run_studio(config)
        return

    # 2. Modo Clássico OpenCV (quando passado --opencv)
    print("=" * 65)
    print("   TRADUTOR DE LIBRAS PROFISSIONAL (MODO OPENCV HUD)")
    print("=" * 65)
    print(f"[*] Câmera selecionada: Índice {config.CAMERA_INDEX}")
    print(f"[*] Resolução alvo: {config.TARGET_WIDTH}x{config.TARGET_HEIGHT}")
    print(f"[*] Modo Bimanual: Ativado (até {config.MAX_NUM_HANDS} mãos simultâneas)")
    print(f"[*] Ponto de Articulação Facial: {'Ativado' if config.ENABLE_FACE else 'Desativado'}")
    print(f"[*] Tradutor Sintático Fluente: {'Ativado' if config.ENABLE_SYNTAX_TRANSLATOR else 'Desativado'}")
    print("-" * 65)

    # Abertura da Webcam com Captura Assíncrona Desacoplada
    from video_pipeline import ThreadedCamera
    try:
        cap = ThreadedCamera(
            camera_index=config.CAMERA_INDEX,
            width=config.TARGET_WIDTH,
            height=config.TARGET_HEIGHT,
            fps=config.MIN_FPS,
            use_dshow=config.USE_DSHOW
        ).start()
        actual_w = cap.actual_w
        actual_h = cap.actual_h
        print(f"[SUCESSO] Câmera conectada com sucesso via ThreadedCamera! Resolução: {actual_w}x{actual_h}")
    except Exception as err:
        print(f"[AVISO] ThreadedCamera falhou ({err}). Tentando fallback padrão...")
        cap = open_camera(config.CAMERA_INDEX, use_dshow=config.USE_DSHOW)
        if not cap.isOpened():
            print(f"\n[ERRO FATAL] Não foi possível abrir a câmera no índice {config.CAMERA_INDEX}.")
            print("-> Verifique se a webcam não está sendo usada por outro aplicativo.\n")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TARGET_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.MIN_FPS)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[SUCESSO] Câmera conectada via fallback! Resolução: {actual_w}x{actual_h}")


    # 2. Inicialização dos Módulos Principais e Avançados
    tracker = HandTracker(
        max_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        model_path=config.MODEL_PATH
    )
    classifier = LibrasClassifier()
    text_buffer = TextBuffer(
        commit_interval_sec=config.COMMIT_INTERVAL_SEC,
        history_size=config.HISTORY_WINDOW_SIZE,
        min_stability=config.MIN_STABILITY_THRESHOLD,
        auto_repeat_interval_sec=config.AUTO_REPEAT_INTERVAL_SEC
    )
    hud = UIHUD(config)

    # Módulos de Áudio, Predição, Trajetória e Calibração
    audio = AudioFeedback(enable_sound=config.ENABLE_AUDIO, enable_tts=config.ENABLE_TTS)
    predictor = WordPredictor()
    trajectory = TrajectoryTracker()
    calibrator = HandCalibrator()

    # Módulos Linguísticos Avançados: Facial, Bimanual e Sintático
    face_tracker = FaceTracker() if config.ENABLE_FACE else None
    lexical_clf = LexicalClassifier(static_classifier=classifier)
    syntax_translator = SyntaxTranslator() if config.ENABLE_SYNTAX_TRANSLATOR else None

    last_hand_seen_time = time.time()

    window_name = "Tradutor de Libras - Tempo Real"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.TARGET_WIDTH, config.TARGET_HEIGHT)

    # Janela opcional do guia de sinais
    show_guide = args.guide
    guide_window_name = "Guia de Sinais de Libras"
    if show_guide:
        guide_img = create_gesture_guide_image()
        cv2.imshow(guide_window_name, guide_img)

    fps_history = []
    prev_time = time.time()

    print("[*] Sistema pronto com Bimanual, Ponto de Articulação Facial e Tradução Sintática!")
    print("[*] Pressione 'h' para o Guia de Sinais | 't' para Copiar | 'k' para Calibrar | '1-3' para Sugestões.")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[AVISO] Falha na leitura do frame da webcam.")
                time.sleep(0.01)
                continue

            # 1. Espelhamento horizontal para manter a ergonomia visual do usuário
            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # 2. Rastreamento das Mãos (Bimanual: até 2 mãos)
            detections = tracker.process(frame)
            primary_hand = detections[0] if len(detections) > 0 else None

            # Renderiza esqueletos de todas as mãos detectadas
            tracker.draw_hands(frame, detections)

            # 3. Rastreamento de Referências Faciais (Ponto de Articulação)
            face_keypoints = None
            if face_tracker is not None:
                face_keypoints = face_tracker.process(frame)
                if face_keypoints is not None:
                    face_tracker.draw_landmarks(frame, face_keypoints, actual_w, actual_h)

            clf_result = None

            # 4. Avaliação Prioritária: Sinais Lexicais, Bimanuais ou com Ponto no Rosto
            lexical_res = lexical_clf.evaluate(hands=detections, face=face_keypoints)
            if lexical_res:
                lex_word, lex_conf, lex_type = lexical_res
                # Insere a palavra inteira diretamente no buffer
                text_buffer.commit_word(lex_word)
                hud.trigger_commit_feedback(lex_word)
                audio.play_commit_sound()
                audio.speak_word(lex_word)
                hud.trigger_toast(f"[{lex_type}] {lex_word}")
                print(f"-> [SINAL {lex_type}]: '{lex_word}' | Texto: '{text_buffer.current_text}'")

            elif primary_hand is not None:
                # 5. Classificação Heurística Unimanual de Alta Precisão (Canônica)
                clf_result = classifier.classify(primary_hand)

                # 6. Sinais Dinâmicos com Trajetória ('J' e 'Z')
                dynamic_res = trajectory.update(primary_hand, clf_result.finger_states if clf_result else None)
                if dynamic_res:
                    dyn_letter, dyn_conf = dynamic_res
                    text_buffer._apply_commit(dyn_letter)
                    hud.trigger_commit_feedback(dyn_letter)
                    hud.trigger_toast(f"SINAL DINAMICO: {dyn_letter}")
                    audio.play_commit_sound()
                    print(f"-> [SINAL DINAMICO]: '{dyn_letter}' | Texto: '{text_buffer.current_text}'")
                    # 7. Buffer Temporal de Composição de Datilologia
                    committed_char = text_buffer.update(
                        clf_result.letter,
                        confidence=clf_result.confidence,
                        is_transition=clf_result.is_transition
                    )
                    if committed_char:
                        hud.trigger_commit_feedback(committed_char)
                        if committed_char == "APAGAR":
                            audio.play_delete_sound()
                        elif committed_char == "ESPAÇO":
                            audio.play_commit_sound()
                            words = text_buffer.current_text.strip().split(" ")
                            if words and words[-1]:
                                audio.speak_word(words[-1])
                        else:
                            audio.play_commit_sound()

                        print(f"-> [CONFIRMADO]: '{committed_char}' | Texto atual: '{text_buffer.current_text}'")

            # 8. Detecção Inteligente de Pausa (Finalização Automática ao Abaixar as Mãos)
            now = time.time()
            if len(detections) > 0:
                last_hand_seen_time = now
            else:
                text_buffer.update(None)
                trajectory.update(None, None)

                if config.ENABLE_SMART_PAUSE and (now - last_hand_seen_time) > config.SMART_PAUSE_SEC:
                    current_word = text_buffer.get_current_word()
                    if current_word and not text_buffer.current_text.endswith(" "):
                        text_buffer.current_text += " "
                        audio.play_commit_sound()
                        audio.speak_word(current_word)
                        hud.trigger_toast(f"PAUSA: '{current_word}' CONCLUIDA")
                        last_hand_seen_time = now

            # 9. Tradução Sintática Fluente (Gramática de Libras -> Português Fluente)
            fluent_translation = None
            if syntax_translator is not None:
                token_list = [w for w in text_buffer.current_text.strip().split(" ") if w]
                if token_list:
                    fluent_translation = syntax_translator.translate_sequence(token_list)

            # 10. Atualização de Calibração (se ativa)
            if calibrator.is_active():
                calib_profile = calibrator.update(primary_hand)
                if calib_profile:
                    classifier.set_calibration_profile(calib_profile)
                    audio.play_commit_sound()
                    hud.trigger_toast("CALIBRACAO CONCLUIDA COM SUCESSO!")

            # 11. Sugestões de Palavras em Português
            current_prefix = text_buffer.get_current_word()
            suggestions = predictor.get_suggestions(current_prefix, max_suggestions=3) if config.ENABLE_PREDICTOR else []

            # Cálculo de taxa de quadros (FPS)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            current_fps = 1.0 / dt if dt > 0 else 30.0
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 12. Renderização do HUD de alta definição sobre o frame
            frame = hud.render(
                frame=frame,
                detection=primary_hand,
                clf_result=clf_result,
                text_buffer=text_buffer,
                fps=avg_fps,
                suggestions=suggestions,
                motion_trail=trajectory.get_trail_pixels(),
                calibrator=calibrator,
                all_detections=detections,
                fluent_translation=fluent_translation
            )

            cv2.imshow(window_name, frame)

            # Tratamento de Teclado
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                # ESC ou 'q': encerrar aplicação
                break
            elif key in (ord('c'), ord('C')):
                # Limpar texto acumulado
                text_buffer.clear()
                audio.play_clear_sound()
                hud.trigger_toast("TEXTO LIMPO")
                print("[INFO] Texto do buffer limpo.")
            elif key == 8:
                # Backspace: apagar último caractere
                text_buffer.backspace()
                audio.play_delete_sound()
                print(f"[INFO] Caractere apagado. Texto: '{text_buffer.current_text}'")
            elif key == 32:
                # Espaço: inserir espaço manualmente
                words = text_buffer.current_text.strip().split(" ")
                if words and words[-1]:
                    audio.speak_word(words[-1])
                text_buffer._apply_commit("ESPAÇO")
                audio.play_commit_sound()
                print(f"[INFO] Espaço inserido. Texto: '{text_buffer.current_text}'")
            elif key in (ord('1'), ord('2'), ord('3')):
                # Selecionar sugestão de autocompletar
                idx = key - ord('1')
                if idx < len(suggestions):
                    selected = suggestions[idx]
                    text_buffer.current_text = predictor.apply_suggestion(text_buffer.current_text, selected)
                    audio.play_commit_sound()
                    audio.speak_word(selected)
                    hud.trigger_toast(f"PALAVRA: {selected}")
                    print(f"[AUTOCOMPLETAR] '{selected}' aplicado. Texto: '{text_buffer.current_text}'")
            elif key in (ord('t'), ord('T')):
                # Copiar para a Área de Transferência (Clipboard)
                try:
                    import pyperclip
                    clip_text = fluent_translation if (fluent_translation and fluent_translation.strip()) else text_buffer.current_text
                    pyperclip.copy(clip_text)
                    audio.play_commit_sound()
                    hud.trigger_toast("TEXTO COPIADO PARA O CLIPBOARD!")
                    print(f"[CLIPBOARD] Texto copiado: '{clip_text}'")
                except Exception as err:
                    print(f"[ERRO] Falha ao copiar para o clipboard: {err}")
            elif key in (ord('m'), ord('M')):
                # Mudo / Ativar áudio
                is_muted = audio.toggle_mute()
                hud.trigger_toast("AUDIO: MUDO" if is_muted else "AUDIO: ATIVADO")
            elif key in (ord('v'), ord('V')):
                # Falar frase inteira (priorizando tradução sintática fluente se disponível)
                phrase_to_speak = fluent_translation if (fluent_translation and fluent_translation.strip()) else text_buffer.current_text.strip()
                if phrase_to_speak:
                    audio.speak(phrase_to_speak)
                    hud.trigger_toast("FALANDO FRASE...")
            elif key in (ord('k'), ord('K')):
                # Iniciar / Cancelar Calibração
                if not calibrator.is_active():
                    calibrator.start_calibration()
                    hud.trigger_toast("INICIANDO CALIBRACAO...")
                else:
                    calibrator.cancel()
                    hud.trigger_toast("CALIBRACAO CANCELADA")
            elif key in (ord('s'), ord('S')):
                # Salvar texto em arquivo local
                filename = "transcricao_libras.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text_buffer.current_text + "\n")
                audio.play_commit_sound()
                hud.trigger_toast("ARQUIVO SALVO: transcricao_libras.txt")
                print(f"[SUCESSO] Texto salvo em '{filename}': '{text_buffer.current_text}'")
            elif key in (ord('l'), ord('L')):
                # Alternar Modo de Legendas Flutuantes vs HUD Completo
                hud.toggle_subtitle_mode()
            elif key in (ord('h'), ord('H')):
                # Alternar exibição do guia visual de sinais
                show_guide = not show_guide
                if show_guide:
                    guide_img = create_gesture_guide_image()
                    cv2.imshow(guide_window_name, guide_img)
                else:
                    try:
                        cv2.destroyWindow(guide_window_name)
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\n[INFO] Interrupção manual recebida.")

    finally:
        print("\nFinalizando aplicação...")
        audio.stop()
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Transcrição final obtida:")
        print(f"'{text_buffer.current_text}'")
        print("Programa encerrado com sucesso.")


if __name__ == "__main__":
    main()
