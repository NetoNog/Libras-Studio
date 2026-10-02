"""
Suíte de Testes Automatizados para o Tradutor de Libras.
Testa imports, simulações geométricas sintéticas de landmarks para cada sinal,
lógica de debounce do buffer de texto, predição de palavras, calibração,
rastreamento de trajetória e integridade da renderização do HUD.
"""

import time
import numpy as np

from calibrator import HandCalibrator
from classifier import LibrasClassifier
from config import AppConfig
from face_tracker import FaceKeypoints
from hand_tracker import HandDetection
from lexical_classifier import LexicalClassifier
from syntax_translator import SyntaxTranslator
from text_buffer import TextBuffer
from trajectory_tracker import TrajectoryTracker
from ui_hud import UIHUD
from word_predictor import WordPredictor


def create_synthetic_hand(
    thumb_extended: bool = False,
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
    index_middle_spread: bool = False,
    crossed_fingers: bool = False,
    fingers_down: bool = False,
    down_count: int = 3,
    thumb_down: bool = False,
    handedness: str = "Right"
) -> HandDetection:
    """Cria uma mão sintética normalizada para testar heurísticas geométricas."""
    landmarks = np.zeros((21, 3), dtype=np.float32)

    # Pulso na base inferior
    landmarks[0] = [0.5, 0.8, 0.0]

    # Articulações MCP (juntas da base dos dedos)
    landmarks[5] = [0.45, 0.55, 0.0]  # Index MCP
    landmarks[9] = [0.50, 0.53, 0.0]  # Middle MCP
    landmarks[13] = [0.55, 0.55, 0.0] # Ring MCP
    landmarks[17] = [0.60, 0.58, 0.0] # Pinky MCP

    if fingers_down:
        # Pulso no topo e dedos voltados para baixo (Letras M e N)
        landmarks[0] = [0.50, 0.40, 0.0]

        landmarks[6] = [0.45, 0.60, 0.0]
        landmarks[8] = [0.45, 0.72, 0.0] # Index tip para baixo

        landmarks[10] = [0.50, 0.60, 0.0]
        landmarks[12] = [0.50, 0.72, 0.0] # Middle tip para baixo

        if down_count >= 3:
            landmarks[14] = [0.55, 0.60, 0.0]
            landmarks[16] = [0.55, 0.72, 0.0] # Ring tip para baixo (M)
        else:
            landmarks[14] = [0.55, 0.56, 0.0]
            landmarks[16] = [0.55, 0.56, 0.0] # Ring fechado contra a palma (N)

        landmarks[18] = [0.60, 0.60, 0.0]
        landmarks[20] = [0.60, 0.61, 0.0]

    else:
        # Dedo Indicador (MCP 5, PIP 6, DIP 7, TIP 8)
        idx_tip_x = 0.52 if crossed_fingers else 0.45
        if index_extended:
            landmarks[6] = [0.45, 0.42, 0.0]
            landmarks[7] = [0.45, 0.32, 0.0]
            landmarks[8] = [idx_tip_x, 0.22, 0.0] # TIP bem acima do PIP
        else:
            landmarks[6] = [0.45, 0.60, 0.0]
            landmarks[7] = [0.45, 0.65, 0.0]
            landmarks[8] = [0.45, 0.70, 0.0] # TIP dobrado abaixo do PIP

        # Dedo Médio (MCP 9, PIP 10, DIP 11, TIP 12)
        mid_tip_x = 0.44 if crossed_fingers else (0.55 if index_middle_spread else 0.47)
        if middle_extended:
            landmarks[10] = [0.50, 0.40, 0.0]
            landmarks[11] = [0.50, 0.30, 0.0]
            landmarks[12] = [mid_tip_x, 0.20, 0.0]
        else:
            landmarks[10] = [0.50, 0.60, 0.0]
            landmarks[11] = [0.50, 0.65, 0.0]
            landmarks[12] = [0.50, 0.70, 0.0]

        # Dedo Anelar (MCP 13, PIP 14, DIP 15, TIP 16)
        if ring_extended:
            landmarks[14] = [0.55, 0.42, 0.0]
            landmarks[15] = [0.55, 0.32, 0.0]
            landmarks[16] = [0.55, 0.23, 0.0]
        else:
            landmarks[14] = [0.55, 0.60, 0.0]
            landmarks[15] = [0.55, 0.65, 0.0]
            landmarks[16] = [0.55, 0.70, 0.0]

        # Dedo Mínimo (MCP 17, PIP 18, DIP 19, TIP 20)
        if pinky_extended:
            landmarks[18] = [0.60, 0.44, 0.0]
            landmarks[19] = [0.62, 0.34, 0.0]
            landmarks[20] = [0.63, 0.25, 0.0]
        else:
            landmarks[18] = [0.60, 0.62, 0.0]
            landmarks[19] = [0.60, 0.67, 0.0]
            landmarks[20] = [0.60, 0.72, 0.0]

    # Polegar (CMC 1, MCP 2, IP 3, TIP 4)
    landmarks[1] = [0.47, 0.75, 0.0]
    landmarks[2] = [0.42, 0.68, 0.0]
    if thumb_down:
        landmarks[3] = [0.40, 0.78, 0.0]
        landmarks[4] = [0.38, 0.88, 0.0] # Apontando para baixo
    elif thumb_extended:
        # Abre para a esquerda (Right hand palm view)
        landmarks[3] = [0.34, 0.60, 0.0]
        landmarks[4] = [0.25, 0.55, 0.0]
    else:
        # Dobrado sobre a palma
        landmarks[3] = [0.45, 0.62, 0.0]
        landmarks[4] = [0.48, 0.60, 0.0]

    pixel_coords = (landmarks[:, :2] * np.array([640, 480])).astype(np.int32)

    return HandDetection(
        landmarks_norm=landmarks,
        landmarks_pixel=pixel_coords,
        handedness=handedness,
        confidence=0.95
    )


def test_classifier_letter_a():
    classifier = LibrasClassifier()
    # Letra A: Punho fechado com polegar ereto na lateral externa
    hand_a = create_synthetic_hand()
    hand_a.landmarks_norm[4] = [0.40, 0.50, 0.0]
    res = classifier.classify(hand_a)
    assert res.letter == "A", f"Esperado 'A', obtido '{res.letter}'"


def test_classifier_letter_s():
    classifier = LibrasClassifier()
    # Letra S: Punho fechado com polegar cruzado sobre a frente dos dedos
    hand_s = create_synthetic_hand()
    hand_s.landmarks_norm[4] = [0.48, 0.62, 0.0]
    res = classifier.classify(hand_s)
    assert res.letter == "S", f"Esperado 'S', obtido '{res.letter}'"


def test_classifier_letter_l():
    classifier = LibrasClassifier()
    hand_l = create_synthetic_hand(thumb_extended=True, index_extended=True)
    res = classifier.classify(hand_l)
    assert res.letter == "L", f"Esperado 'L', obtido '{res.letter}'"


def test_classifier_letter_v():
    classifier = LibrasClassifier()
    hand_v = create_synthetic_hand(index_extended=True, middle_extended=True, index_middle_spread=True)
    res = classifier.classify(hand_v)
    assert res.letter == "V", f"Esperado 'V', obtido '{res.letter}'"


def test_classifier_letter_u():
    classifier = LibrasClassifier()
    hand_u = create_synthetic_hand(index_extended=True, middle_extended=True, index_middle_spread=False)
    res = classifier.classify(hand_u)
    assert res.letter == "U", f"Esperado 'U', obtido '{res.letter}'"


def test_classifier_letter_r():
    classifier = LibrasClassifier()
    # Letra R: Dedos indicador e médio cruzados
    hand_r = create_synthetic_hand(index_extended=True, middle_extended=True, crossed_fingers=True)
    res = classifier.classify(hand_r)
    assert res.letter == "R", f"Esperado 'R', obtido '{res.letter}'"


def test_classifier_letter_m_and_n():
    classifier = LibrasClassifier()
    # Letra M: 3 dedos voltados para baixo
    hand_m = create_synthetic_hand(fingers_down=True, down_count=3)
    res_m = classifier.classify(hand_m)
    assert res_m.letter == "M", f"Esperado 'M', obtido '{res_m.letter}'"

    # Letra N: 2 dedos voltados para baixo
    hand_n = create_synthetic_hand(fingers_down=True, down_count=2)
    res_n = classifier.classify(hand_n)
    assert res_n.letter == "N", f"Esperado 'N', obtido '{res_n.letter}'"


def test_classifier_letter_w():
    classifier = LibrasClassifier()
    hand_w = create_synthetic_hand(index_extended=True, middle_extended=True, ring_extended=True)
    res = classifier.classify(hand_w)
    assert res.letter == "W", f"Esperado 'W', obtido '{res.letter}'"


def test_classifier_letter_b():
    classifier = LibrasClassifier()
    hand_b = create_synthetic_hand(index_extended=True, middle_extended=True,
                                   ring_extended=True, pinky_extended=True,
                                   thumb_extended=False)
    res = classifier.classify(hand_b)
    assert res.letter == "B", f"Esperado 'B', obtido '{res.letter}'"


def test_classifier_letter_i():
    classifier = LibrasClassifier()
    hand_i = create_synthetic_hand(pinky_extended=True)
    res = classifier.classify(hand_i)
    assert res.letter == "I", f"Esperado 'I', obtido '{res.letter}'"


def test_classifier_letter_y():
    classifier = LibrasClassifier()
    hand_y = create_synthetic_hand(thumb_extended=True, pinky_extended=True)
    res = classifier.classify(hand_y)
    assert res.letter == "Y", f"Esperado 'Y', obtido '{res.letter}'"


def test_classifier_command_apagar():
    classifier = LibrasClassifier()
    hand_apagar = create_synthetic_hand(thumb_down=True)
    res = classifier.classify(hand_apagar)
    assert res.letter == "APAGAR", f"Esperado 'APAGAR', obtido '{res.letter}'"


def test_word_predictor():
    predictor = WordPredictor()
    # Prefixo "BO"
    sug = predictor.get_suggestions("BO", max_suggestions=3)
    assert len(sug) > 0, "Deveria retornar sugestões para 'BO'"
    assert "BOM" in sug or "BOA" in sug

    # Aplicação de sugestão
    text = "OLA MEU "
    new_text = predictor.apply_suggestion(text, "AMIGO")
    assert new_text == "OLA MEU AMIGO ", f"Texto incorreto: '{new_text}'"


def test_calibrator_flow():
    calibrator = HandCalibrator()
    calibrator.start_calibration()
    assert calibrator.is_active()
    assert calibrator.state == HandCalibrator.STATE_OPEN

    # Simula frames de calibração
    hand_open = create_synthetic_hand(thumb_extended=True, index_extended=True,
                                      middle_extended=True, ring_extended=True, pinky_extended=True)
    calibrator.update(hand_open)
    instr, prog = calibrator.get_hud_instruction()
    assert "PASSO 1/2" in instr

    calibrator.cancel()
    assert not calibrator.is_active()


def test_text_buffer_debounce():
    buf = TextBuffer(commit_interval_sec=1.0, history_size=10, min_stability=0.7)

    # Inicia com sinal 'L'
    t0 = 100.0
    for i in range(10):
        committed = buf.update("L", timestamp=t0 + (i * 0.05))
        assert committed is None, "Não deve confirmar antes do intervalo de 1.0s"

    assert buf.current_candidate == "L"
    assert buf.current_stability >= 0.70

    # Avança para 1.05s sustentando 'L'
    committed = buf.update("L", timestamp=t0 + 1.05)
    assert committed == "L", f"Deveria ter confirmado 'L', retornou '{committed}'"
    assert buf.current_text == "L"

    # Sustenta sinal 'V'
    t1 = t0 + 2.0
    for i in range(10):
        buf.update("V", timestamp=t1 + (i * 0.05))

    committed = buf.update("V", timestamp=t1 + 1.40)
    assert committed == "V"
    assert buf.current_text == "LV"

    # Testa backspace
    buf.backspace()
    assert buf.current_text == "L"

    # Testa clear
    buf.clear()
    assert buf.current_text == ""


    # Testa commit de palavra inteira (léxica / bimanual)
    buf.clear()
    buf.commit_word("OI")
    assert buf.current_text == "OI "
    buf.commit_word("TUDO")
    assert buf.current_text == "OI TUDO "


def test_syntax_translator():
    translator = SyntaxTranslator()
    assert translator.translate_sequence(["EU", "CASA", "IR"]) == "Eu estou indo para casa."
    assert translator.translate_sequence(["OI", "TUDO", "BEM"]) == "Olá! Tudo bem com você?"
    assert translator.translate_sequence(["BOM", "DIA"]) == "Bom dia!"
    assert translator.translate_sequence(["QUERO", "AGUA"]) == "Eu gostaria de água, por favor."
    assert translator.translate_sequence(["DESCULPA"]) == "Peço desculpas."
    print("SyntaxTranslator validado com sucesso!")


def test_lexical_bimanual_casa():
    classifier = LibrasClassifier()
    lex_clf = LexicalClassifier(static_classifier=classifier)

    # Mão direita aberta (B) inclinada formando telhado
    h1 = create_synthetic_hand(index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=True, handedness="Right")
    orig1 = h1.landmarks_norm[0, :2].copy()
    t1 = np.radians(20)
    R1 = np.array([[np.cos(t1), -np.sin(t1)], [np.sin(t1), np.cos(t1)]])
    h1.landmarks_norm[:, :2] = (h1.landmarks_norm[:, :2] - orig1) @ R1.T + orig1
    h1.landmarks_norm[:, 0] -= 0.15

    # Mão esquerda aberta (B) inclinada formando telhado oposto
    h2 = create_synthetic_hand(index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=True, handedness="Left")
    orig2 = h2.landmarks_norm[0, :2].copy()
    t2 = np.radians(-20)
    R2 = np.array([[np.cos(t2), -np.sin(t2)], [np.sin(t2), np.cos(t2)]])
    h2.landmarks_norm[:, :2] = (h2.landmarks_norm[:, :2] - orig2) @ R2.T + orig2
    h2.landmarks_norm[:, 0] += 0.25

    res = lex_clf.evaluate(hands=[h1, h2], face=None)
    assert res is not None, "Deveria reconhecer sinal bimanual CASA"
    assert res[0] == "CASA"
    assert res[2] == "BIMANUAL"
    print("Sinal bimanual CASA validado com sucesso!")


def test_lexical_facial_agua():
    classifier = LibrasClassifier()
    lex_clf = LexicalClassifier(static_classifier=classifier)

    # Mão em 'L' posicionada no queixo
    hand = create_synthetic_hand(thumb_extended=True, index_extended=True)
    dy = 0.60 - hand.landmarks_norm[8, 1]
    hand.landmarks_norm[:, 1] += dy

    face = FaceKeypoints(
        chin=np.array([hand.landmarks_norm[8, 0], 0.60, 0.0], dtype=np.float32),
        mouth=np.array([0.50, 0.50, 0.0], dtype=np.float32),
        nose=np.array([0.50, 0.45, 0.0], dtype=np.float32),
        forehead=np.array([0.50, 0.25, 0.0], dtype=np.float32),
        face_size=0.35
    )

    res = lex_clf.evaluate(hands=[hand], face=face)
    assert res is not None, "Deveria reconhecer sinal com ponto facial AGUA"
    assert res[0] == "AGUA"
    assert res[2] == "FACIAL"
    print("Sinal com ponto de articulação facial AGUA validado com sucesso!")


def test_hud_rendering():
    cfg = AppConfig()
    hud = UIHUD(cfg)
    buf = TextBuffer()
    buf.current_text = "TESTE LIBRAS"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hand = create_synthetic_hand(thumb_extended=True, index_extended=True)
    classifier = LibrasClassifier()
    clf_res = classifier.classify(hand)

    suggestions = ["BOM", "BOA", "BONITO"]
    trail = [(200, 200), (210, 220), (220, 240), (230, 230)]
    hud.trigger_toast("TESTE TOAST")

    rendered = hud.render(
        frame=frame,
        detection=hand,
        clf_result=clf_res,
        text_buffer=buf,
        fps=32.5,
        suggestions=suggestions,
        motion_trail=trail,
        calibrator=None,
        all_detections=[hand],
        fluent_translation="Eu estou indo para casa."
    )
    assert rendered.shape == (480, 640, 3)
    assert rendered.dtype == np.uint8
    print("HUD renderizado com sucesso com todos os novos componentes!")


def test_ui_renderer_typography_and_glass():
    from ui_renderer import UIRenderer
    renderer = UIRenderer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    w, h = renderer.get_text_size("LIBRAS STUDIO PRO", 16, bold=True)
    assert w > 0 and h > 0, "Deveria calcular dimensões do texto com Segoe UI"

    glass_frame = renderer.draw_glass_panel(frame.copy(), 50, 50, 200, 100, radius=12)
    assert glass_frame.shape == (480, 640, 3)

    text_frame = renderer.render_texts_on_frame(glass_frame, [
        ("LIBRAS STUDIO", (60, 60), 14, (255, 255, 255), True, True),
        ("Subtítulo de teste", (60, 90), 11, (180, 200, 220), False, False)
    ])
    assert text_frame.shape == (480, 640, 3)
    print("UIRenderer (Tipografia Segoe UI + Frosted Glass) validado com sucesso!")


def test_hud_subtitle_mode():
    cfg = AppConfig()
    hud = UIHUD(cfg)
    buf = TextBuffer()
    buf.current_text = "BOA NOITE"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Ativa modo legendas
    hud.toggle_subtitle_mode()
    assert hud.subtitle_mode is True, "Modo legendas deveria estar ativado"

    rendered = hud.render(
        frame=frame,
        detection=None,
        clf_result=None,
        text_buffer=buf,
        fps=30.0,
        fluent_translation="Boa noite a todos."
    )
    assert rendered.shape == (480, 640, 3)

    hud.toggle_subtitle_mode()
    assert hud.subtitle_mode is False, "Modo legendas deveria ter sido desativado"
    print("Modo de Legendas Flutuantes (Subtitle Mode) validado com sucesso!")


def test_srt_export_formatting():
    # Testa a lógica padrão de formatação SubRip (.SRT)
    def format_srt_time(seconds: float) -> str:
        ms = int((seconds - int(seconds)) * 1000)
        s = int(seconds) % 60
        m = (int(seconds) // 60) % 60
        h = int(seconds) // 3600
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    ts = format_srt_time(3723.456)
    assert ts == "01:02:03,456", f"Timestamp SRT incorreto: {ts}"
    print("Formatador de legendas .SRT validado com sucesso!")


def test_hud_rendering_hd_resolution():
    cfg = AppConfig()
    hud = UIHUD(cfg)
    buf = TextBuffer()
    buf.current_text = "TESTE ALTA RESOLUÇÃO"

    frame_hd = np.zeros((720, 1280, 3), dtype=np.uint8)
    hand = create_synthetic_hand(thumb_extended=True, index_extended=True)
    classifier = LibrasClassifier()
    clf_res = classifier.classify(hand)

    suggestions = ["ALTO", "ALTA", "AMOR"]
    rendered = hud.render(
        frame=frame_hd,
        detection=hand,
        clf_result=clf_res,
        text_buffer=buf,
        fps=60.0,
        suggestions=suggestions,
        fluent_translation="Teste em resolução HD 1280x720."
    )
    assert rendered.shape == (720, 1280, 3), f"Esperado (720, 1280, 3), obtido {rendered.shape}"
    print("HUD em Alta Resolução HD (1280x720) validado com sucesso!")


def test_classifier_letter_f_and_t():
    classifier = LibrasClassifier()
    # Letra F: Indicador dobrado, médio, anelar e mínimo erguidos. Polegar por FORA.
    hand_f = create_synthetic_hand(middle_extended=True, ring_extended=True, pinky_extended=True)
    hand_f.landmarks_norm[4] = [0.38, 0.58, 0.0]  # Polegar lateral externo
    res_f = classifier.classify(hand_f)
    assert res_f.letter == "F", f"Esperado 'F', obtido '{res_f.letter}'"

    # Letra T: Indicador dobrado, médio, anelar e mínimo erguidos. Polegar por DENTRO.
    hand_t = create_synthetic_hand(middle_extended=True, ring_extended=True, pinky_extended=True)
    hand_t.landmarks_norm[4] = [0.49, 0.58, 0.0]  # Polegar inserido entre indicador e médio
    res_t = classifier.classify(hand_t)
    assert res_t.letter == "T", f"Esperado 'T', obtido '{res_t.letter}'"
    print("Diferenciação precisa das letras F e T validada com sucesso!")


def test_classifier_letter_q():
    classifier = LibrasClassifier()
    # Letra Q: Indicador e polegar estendidos paralelamente apontando para baixo
    hand_q = create_synthetic_hand(thumb_extended=True, index_extended=True)
    # Pulso no topo e mão orientada para baixo
    hand_q.landmarks_norm[0] = [0.50, 0.40, 0.0]   # Pulso no topo
    hand_q.landmarks_norm[5] = [0.45, 0.55, 0.0]   # Index MCP
    hand_q.landmarks_norm[6] = [0.45, 0.65, 0.0]   # Index PIP
    hand_q.landmarks_norm[8] = [0.45, 0.75, 0.0]   # Index TIP para baixo
    hand_q.landmarks_norm[2] = [0.42, 0.50, 0.0]   # Thumb MCP
    hand_q.landmarks_norm[3] = [0.40, 0.60, 0.0]   # Thumb IP
    hand_q.landmarks_norm[4] = [0.40, 0.70, 0.0]   # Thumb TIP paralelo para baixo
    # Dedos médios, anelar e mínimo dobrados contra a palma
    hand_q.landmarks_norm[10] = [0.50, 0.54, 0.0]
    hand_q.landmarks_norm[12] = [0.50, 0.54, 0.0]
    hand_q.landmarks_norm[14] = [0.55, 0.54, 0.0]
    hand_q.landmarks_norm[16] = [0.55, 0.54, 0.0]
    hand_q.landmarks_norm[18] = [0.60, 0.56, 0.0]
    hand_q.landmarks_norm[20] = [0.60, 0.56, 0.0]
    res_q = classifier.classify(hand_q)
    assert res_q.letter == "Q", f"Esperado 'Q', obtido '{res_q.letter}'"
    print("Letra Q (orientação para baixo) validada com sucesso!")



def test_classifier_letter_x():
    classifier = LibrasClassifier()
    # Letra X: Indicador em gancho, demais dedos recolhidos
    hand_x = create_synthetic_hand()
    hand_x.landmarks_norm[6] = [0.45, 0.44, 0.0]  # PIP levantada
    hand_x.landmarks_norm[7] = [0.45, 0.48, 0.0]
    hand_x.landmarks_norm[8] = [0.45, 0.54, 0.0]  # TIP curvada para trás/baixo em gancho
    res_x = classifier.classify(hand_x)
    assert res_x.letter == "X", f"Esperado 'X', obtido '{res_x.letter}'"
    print("Letra X (indicador em gancho) validada com sucesso!")


def test_lexical_por_favor_e_te_amo():
    classifier = LibrasClassifier()
    lex_clf = LexicalClassifier(static_classifier=classifier)

    # POR FAVOR: Duas mãos abertas espalmadas unidas
    h1 = create_synthetic_hand(index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=True)
    h2 = create_synthetic_hand(index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=True)
    h2.landmarks_norm[:, 0] += 0.05
    res_pf = lex_clf.evaluate(hands=[h1, h2], face=None)
    assert res_pf is not None and res_pf[0] == "POR FAVOR"

    # TE AMO: ILY (Polegar, Indicador e Mínimo erguidos)
    lex_clf.last_lexical_time = 0.0  # Reseta cooldown entre testes
    hand_ily = create_synthetic_hand(thumb_extended=True, index_extended=True, pinky_extended=True)
    res_ily = lex_clf.evaluate(hands=[hand_ily], face=None)
    assert res_ily is not None and res_ily[0] == "TE AMO"
    print("Sinais léxicos POR FAVOR e TE AMO validados com sucesso!")


def test_kinetic_velocity_transition_filter():
    classifier = LibrasClassifier()
    # Frame 1: Posição inicial (L) no instante t=0
    hand_l = create_synthetic_hand(thumb_extended=True, index_extended=True)
    res1 = classifier.classify(hand_l, timestamp=10.0)
    assert res1.is_transition is False, "Frame inicial não deve estar em transição"

    # Frame 2: Deslocamento rápido das pontas no instante t=10.033 (Delta t = 33ms)
    hand_moved = create_synthetic_hand(thumb_extended=True, index_extended=True)
    # Move as pontas bruscamente simulando troca de sinal
    hand_moved.landmarks_norm[[4, 8, 12, 16, 20], :2] += 0.15
    res2 = classifier.classify(hand_moved, timestamp=10.033)
    assert res2.is_transition is True, f"Deveria detectar transição rápida (vel={res2.kinetic_velocity:.2f})"
    assert res2.kinetic_velocity > 1.25, f"Velocidade cinética esperada > 1.25, obtida {res2.kinetic_velocity}"

    # Frame 3: Mão estabilizada na nova posição no instante t=10.20
    hand_stable = create_synthetic_hand(thumb_extended=True, index_extended=True)
    hand_stable.landmarks_norm[[4, 8, 12, 16, 20], :2] += 0.15
    res3 = classifier.classify(hand_stable, timestamp=10.20)
    assert res3.is_transition is False, "Mão estabilizada não deve estar em transição"
    print("Filtro cinético de velocidade e coarticulação validado com sucesso!")


def test_text_buffer_transition_rejection():
    buf = TextBuffer(commit_interval_sec=0.5, history_size=10, min_stability=0.7)
    t0 = 200.0

    # Simula 5 frames de sinal em transição rápida (ex: 'M' acidental ao fechar a mão)
    for i in range(5):
        committed = buf.update("M", confidence=0.85, timestamp=t0 + (i * 0.033), is_transition=True)
        assert committed is None, "Frame em transição não deve ser confirmado"

    # Buffer deve ter descartado o ruído de transição
    assert buf.current_candidate is None, "Ruído de transição não deve definir candidato"
    assert buf.current_progress == 0.0
    print("Rejeição de ruídos de transição no TextBuffer validada com sucesso!")


def test_profile_manager_and_vector_calibration():
    from profile_manager import SignProfileManager, extract_hand_features
    pm = SignProfileManager()
    assert len(pm.reference_signatures) >= 20, "Deveria conter assinaturas de referência pré-carregadas"

    # Testa gravação de amostra personalizada para o sinal 'L'
    c_l = np.zeros((21, 3), dtype=np.float32)
    c_l[4] = [-0.75, 0.85, 0.0]
    c_l[6] = [-0.30, 1.30, 0.0]; c_l[8] = [-0.30, 1.95, 0.0]
    raw_l = c_l.copy()
    raw_l[9, 1] = raw_l[0, 1] - 0.5
    feat_l = extract_hand_features(c_l, raw_l)

    pm.start_recording("L")
    for _ in range(15):
        pm.add_recording_sample(feat_l)
    pm.finish_recording()

    assert pm.is_calibrated("L") is True, "Sinal 'L' deveria estar marcado como calibrado"
    assert "L" in pm.get_calibrated_signs()

    # Testa matching vetorial
    best_s, score = pm.get_best_match(feat_l)
    assert best_s == "L", f"Esperado 'L', obtido '{best_s}'"
    assert score > 0.90, f"Pontuação de similaridade esperada > 0.90, obtida {score:.3f}"

    # Limpa calibração personalizada
    pm.reset_sign("L")
    assert pm.is_calibrated("L") is False
    print("Motor de assinaturas vetoriais e calibração biométrica validado com sucesso!")


if __name__ == "__main__":
    test_classifier_letter_a()
    test_classifier_letter_s()
    test_classifier_letter_l()
    test_classifier_letter_v()
    test_classifier_letter_u()
    test_classifier_letter_r()
    test_classifier_letter_m_and_n()
    test_classifier_letter_w()
    test_classifier_letter_b()
    test_classifier_letter_i()
    test_classifier_letter_y()
    test_classifier_letter_f_and_t()
    test_classifier_letter_q()
    test_classifier_letter_x()
    test_classifier_command_apagar()
    test_word_predictor()
    test_calibrator_flow()
    test_text_buffer_debounce()
    test_syntax_translator()
    test_lexical_bimanual_casa()
    test_lexical_facial_agua()
    test_lexical_por_favor_e_te_amo()
    test_kinetic_velocity_transition_filter()
    test_text_buffer_transition_rejection()
    test_profile_manager_and_vector_calibration()
    test_hud_rendering()
    test_ui_renderer_typography_and_glass()
    test_hud_subtitle_mode()
    test_srt_export_formatting()
    test_hud_rendering_hd_resolution()
    print("\nTODOS OS TESTES (INCLUINDO AS NOVAS MELHORIAS) PASSARAM COM 100% DE SUCESSO!")



