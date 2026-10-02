"""
Guia Visual e Tabela de Referência de Gestos de Libras em Alta Resolução.
Exibe uma janela auxiliar com instruções anatômicas detalhadas e categorizadas
em 2 colunas para cada sinal suportado pelo tradutor em tempo real.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np


ALPHABET_GESTURES: List[Tuple[str, str]] = [
    ("A", "Punho fechado, polegar encostado na lateral do indicador."),
    ("B", "4 dedos esticados e unidos, polegar dobrado sobre a palma."),
    ("C", "Mao curvada em formato de concha semicircular."),
    ("D", "Indicador para cima; polegar toca na ponta do dedo medio."),
    ("E", "Todas as pontas dos dedos dobradas sobre a palma."),
    ("F", "Indicador dobrado com polegar apoiado por FORA."),
    ("G", "Indicador ereto com polegar paralelo encostado."),
    ("I", "Apenas dedo minimo estendido para cima."),
    ("J", "Minimo desenhando gancho no ar (dinamico)."),
    ("K", "Indicador erguido, medio para frente, polegar no meio."),
    ("L", "Polegar e indicador a 90 graus (formato de L)."),
    ("M", "3 dedos voltados para baixo sobre o polegar."),
    ("N", "2 dedos voltados para baixo sobre o polegar."),
    ("O", "Pontas dos dedos unidas a ponta do polegar em circulo."),
    ("P", "Indicador para frente, medio e polegar em 'K' horizontal."),
    ("Q", "Indicador e polegar estendidos apontando para baixo."),
    ("R", "Indicador e medio estendidos e CRUZADOS em helice."),
    ("S", "Punho fechado com o polegar cruzado SOBRE os dedos."),
    ("T", "Indicador dobrado com polegar inserido por DENTRO."),
    ("U", "Indicador e medio estendidos para cima e BEM JUNTOS."),
    ("V", "Indicador e medio estendidos para cima e SEPARADOS em V."),
    ("W", "Indicador, medio e anelar estendidos e espalhados."),
    ("X", "Indicador curvado em gancho, demais dedos fechados."),
    ("Y", "Polegar e minimo estendidos (sinal Shaka)."),
    ("Z", "Dedo indicador desenhando 'Z' no ar (dinamico).")
]

LEXICAL_AND_BIMANUAL: List[Tuple[str, str]] = [
    ("OI", "Mao acenando com transicao rapida de 'O' para 'I'."),
    ("SIM", "Punho fechado 'S' balancando para baixo (aceno)."),
    ("NAO", "Dedo indicador oscilando horizontalmente."),
    ("EU", "Dedo indicador apontando para o proprio peito."),
    ("VOCE", "Dedo indicador apontando para a camera."),
    ("AGUA", "Mao em 'L' batendo com o indicador no queixo."),
    ("DESCULPA", "Mao em 'Y' encostada na lateral da mandibula."),
    ("BOM", "Mao em concha na boca abrindo para frente."),
    ("SABER", "Mao tocando a lateral da testa."),
    ("OBRIGADO", "Mao espalmada tocando a testa / queixo."),
    ("POR FAVOR", "2 maos espalmadas unidas em suplica."),
    ("CASA", "2 maos abertas unidas pelas pontas (telhado)."),
    ("AJUDA", "Mao base aberta plana e punho com polegar sobre ela."),
    ("NOME", "2 maos em 'U' tocando uma na outra."),
    ("TE AMO", "Polegar, indicador e minimo erguidos (sinal ILY)."),
    ("TRABALHO", "2 maos em 'L' voltadas para baixo no espaco neutro."),
    ("FAMILIA", "2 maos em 'F' unidas pelos polegares e indicadores."),
    ("APLAUSOS", "Ambas as maos no alto acenando com dedos abertos."),
    ("LEGAL", "Polegar ereto para cima (joinha) oscilando."),
    ("ESPACO", "Mao totalmente aberta espalmada para a camera."),
    ("APAGAR", "Punho fechado com polegar para baixo (Thumbs Down).")
]



def create_gesture_guide_image(width: int = 1040, height: int = 680) -> np.ndarray:
    """Gera uma imagem rica com os sinais organizados em 2 colunas para melhor visualizacao."""
    guide_img = np.zeros((height, width, 3), dtype=np.uint8)
    guide_img[:] = (20, 23, 30)

    # Topo / Banner Superior
    cv2.rectangle(guide_img, (0, 0), (width, 60), (28, 34, 48), -1)
    cv2.line(guide_img, (0, 60), (width, 60), (55, 65, 88), 1, cv2.LINE_AA)

    cv2.putText(guide_img, "GUIA DE SINAIS DE LIBRAS - DICIONARIO VISUAL", (24, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(guide_img, "Posicione as maos a ~50cm da camera", (width - 320, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 100), 1, cv2.LINE_AA)

    col_w = (width - 40) // 2
    row_h = 24
    y_start = 105

    # Cabeçalho da Coluna 1
    cv2.putText(guide_img, "ALFABETO DATILOLOGICO (A - Z)", (20, 85),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, (245, 180, 40), 1, cv2.LINE_AA)

    # Cabeçalho da Coluna 2
    cv2.putText(guide_img, "SINAIS LEXICAIS, BIMANUAIS & COMANDOS", (20 + col_w + 10, 85),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, (60, 230, 120), 1, cv2.LINE_AA)

    # Renderiza Coluna 1 (Alfabeto)
    for idx, (letter, desc) in enumerate(ALPHABET_GESTURES):
        y = y_start + (idx * row_h)
        x = 20
        if idx % 2 == 0:
            cv2.rectangle(guide_img, (x - 5, y - 16), (x + col_w - 15, y + 6), (25, 30, 40), -1)

        cv2.putText(guide_img, f"[{letter}]", (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (245, 180, 40), 1, cv2.LINE_AA)
        cv2.putText(guide_img, desc, (x + 55, y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 225, 235), 1, cv2.LINE_AA)

    # Renderiza Coluna 2 (Léxicos e Bimanuais)
    for idx, (word, desc) in enumerate(LEXICAL_AND_BIMANUAL):
        y = y_start + (idx * row_h)
        x = 20 + col_w + 10
        if idx % 2 == 0:
            cv2.rectangle(guide_img, (x - 5, y - 16), (x + col_w - 15, y + 6), (25, 30, 40), -1)

        tag_color = (60, 230, 120) if len(word) > 2 else (0, 220, 255)
        cv2.putText(guide_img, f"[{word}]", (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, tag_color, 1, cv2.LINE_AA)
        offset_desc = 95 if len(word) > 4 else 75
        cv2.putText(guide_img, desc, (x + offset_desc, y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 225, 235), 1, cv2.LINE_AA)

    # Rodapé
    cv2.line(guide_img, (0, height - 32), (width, height - 32), (40, 50, 70), 1, cv2.LINE_AA)
    footer_text = "Pressione [H] novamente na janela principal para fechar este guia."
    cv2.putText(guide_img, footer_text, (24, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 180), 1, cv2.LINE_AA)

    return guide_img
