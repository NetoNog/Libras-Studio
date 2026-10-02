# Libras Studio Pro — Tradutor e Plataforma de Estudo de Libras em Tempo Real
> **Visão Computacional Local (MediaPipe + OpenCV), Reconhecimento Bimanual, Tradução Sintática e Síntese de Voz (TTS)**

> [!NOTE]
> **Propósito Didático & Estudo de Libras**: Este projeto foi concebido exclusivamente para fins **educacionais, de estudo, pesquisa acadêmica e experimentação em visão computacional**. Ele **não** substitui a atuação de Tradutores e Intérpretes de Libras (TILS) certificados. Para uma análise detalhada da arquitetura, tecnologias e roadmap, consulte o arquivo [DOCUMENTACAO.md](DOCUMENTACAO.md).

Sistema avançado de alta precisão e baixa latência para rastreamento bimanual e facial, reconhecimento de sinais estáticos, dinâmicos e léxicos, tradução sintática gramatical (Libras para Português fluente) e **síntese de voz em português (TTS)** via webcam, com execução puramente local em CPU (sem necessidade de placas GPU dedicadas ou APIs em nuvem).


---

## 1. Visão Geral da Arquitetura

O sistema emprega um pipeline desacoplado e multimodal de alta performance:

```
                  [Webcam (30+ FPS com DirectShow no Windows)]
                                       │
                                       ▼
                  [Pré-processamento: cv2.flip horizontal]
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
[MediaPipe Hands (Bimanual: 2 Mãos)]             [MediaPipe Face Landmarker]
  ├── 21 Landmarks 3D por mão                       ├── Ponto de Articulação Fonológica
  ├── 1€ Filter (Eliminação de Jitter)              └── Queixo, Boca, Nariz e Testa
            │                                                     │
            ├──────────────────────────┬──────────────────────────┘
            ▼                          ▼
[Classificador Estático Canônico] [Classificador Léxico & Bimanual]
  ├── Sistema Local da Palma        ├── Sinais Bimanuais (CASA, AJUDA, TRABALHO, FAMÍLIA)
  ├── 100% Invariante a Rotação     ├── Ponto de Articulação (ÁGUA, DESCULPA, BOM, SABER)
  └── Alfabeto Datilológico (A-Z)   └── Sinais Dêiticos & Dinâmicos (OI, SIM, NÃO, EU, VOCÊ)
            │                                  │
            └─────────────────┬────────────────┘
                              ▼
                [Buffer Temporal de Composição]
                  ├── Debounce Temporal e Filtragem
                  ├── Smart Pause (Auto-commit ao abaixar as mãos)
                  └── Formação de Palavras
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
[Tradutor Sintático de Libras]         [Módulo de Áudio & TTS]
  ├── Gramática Tópico-Comentário        ├── Síntese de Voz Offline (PT-BR)
  └── Tradução para Português Fluente    └── Cues Auditivos Não-Bloqueantes
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
                 [Interface HUD de Alta Definição]
                   ├── Barra Superior com Modo Bimanual e FPS
                   ├── Painel do Vetor de 5 Dedos
                   ├── Barra de Tradução Sintática em Destaque
                   ├── Barra de Sugestões de Palavras ([1], [2], [3])
                   ├── Trilha Cinemática de Movimento Neon
                   └── Caixa de Texto com Cursor Pulsante
```

---

## 2. Estrutura dos Módulos

* [app_gui.py](file:///c:/Users/zigle/Desktop/libras/app_gui.py): Interface Desktop Studio Pro (`CustomTkinter`) com feed de vídeo integrado, card de tradução fluente, transcrição em tempo real, telemetria (WPM/sinais), central modal de calibração biométrica ("🎯 Calibrar Mão") e exportador .SRT/.TXT.
* [profile_manager.py](file:///c:/Users/zigle/Desktop/libras/profile_manager.py): Motor de assinaturas vetoriais densas de 44 dimensões invariantes e gerenciamento de perfis biométricos personalizados do usuário com similaridade de cosseno contínua e persistência JSON (`user_sign_signatures.json`).
* [ui_renderer.py](file:///c:/Users/zigle/Desktop/libras/ui_renderer.py): Motor gráfico de alta definição visual com tipografia anti-serrilhada (Segoe UI via Pillow), efeito Frosted Glass (vidro fosco com desfoque e cantos arredondados), anéis de pulso e iluminação cyber-glow.
* [video_pipeline.py](file:///c:/Users/zigle/Desktop/libras/video_pipeline.py): Pipeline assíncrono de captura de vídeo com thread desacoplada, buffer de 1 frame e seletor dinâmico de resolução (720p / 1080p / 480p).
* [config.py](file:///c:/Users/zigle/Desktop/libras/config.py): Configurações globais (DirectShow, resolução padrão HD 1280x720, FPS, modo bimanual, ponto facial, smart pause, cores e limiares).
* [hand_tracker.py](file:///c:/Users/zigle/Desktop/libras/hand_tracker.py): Rastreador bimanual (até 2 mãos) com filtro One-Euro para zero-jitter e renderização multi-mão.
* [face_tracker.py](file:///c:/Users/zigle/Desktop/libras/face_tracker.py): Rastreador de pontos de articulação faciais (queixo, boca, nariz, testa).
* [classifier.py](file:///c:/Users/zigle/Desktop/libras/classifier.py): Motor híbrido (regras geométricas canônicas 3D + modelo vetorial denso) com filtro cinemático de velocidade para supressão de ruído de coarticulação e distinção refinada de punhos (A, S, E), F/T e orientação de M/N/Q.
* [lexical_classifier.py](file:///c:/Users/zigle/Desktop/libras/lexical_classifier.py): Reconhecedor de palavras inteiras (sinais bimanuais, faciais e dêiticos unimanuais).
* [syntax_translator.py](file:///c:/Users/zigle/Desktop/libras/syntax_translator.py): Tradutor gramatical de sequências de Libras para sentenças naturais do português falado.
* [trajectory_tracker.py](file:///c:/Users/zigle/Desktop/libras/trajectory_tracker.py): Reconhecedor cinemático de sinais espaciais com cursores dinâmicos (**J** e **Z**) e trilha neon.
* [audio_feedback.py](file:///c:/Users/zigle/Desktop/libras/audio_feedback.py): Síntese de voz em português offline (`pyttsx3`) e bips sensoriais assíncronos.
* [word_predictor.py](file:///c:/Users/zigle/Desktop/libras/word_predictor.py): Autocompletar com vocabulário de alta frequência da língua portuguesa.
* [calibrator.py](file:///c:/Users/zigle/Desktop/libras/calibrator.py): Calibração biométrica expressa individual guiada em 2 etapas (Mão Aberta / Punho Fechado em 5s).
* [text_buffer.py](file:///c:/Users/zigle/Desktop/libras/text_buffer.py): Buffer temporal com histerese de candidatos, rejeição estrita de transições cinéticas e inserção de palavras inteiras.
* [ui_hud.py](file:///c:/Users/zigle/Desktop/libras/ui_hud.py): Interface futurista de alto contraste com tradução fluente, toasts, vetor de dedos, modo legendas cinemático (`L`) e efeitos de ripple.
* [gesture_guide.py](file:///c:/Users/zigle/Desktop/libras/gesture_guide.py): Guia visual de sinais em duas colunas de alta legibilidade.
* [main.py](file:///c:/Users/zigle/Desktop/libras/main.py): Orquestrador principal da aplicação com suporte a modo Studio Desktop e modo OpenCV clássico.
* [test_libras.py](file:///c:/Users/zigle/Desktop/libras/test_libras.py): Suíte de testes automatizados com cobertura de 100% de todos os módulos.

---

## 3. Catálogo de Sinais e Gestos Suportados

### Alfabeto Datilológico e Comandos (A - Z Completo)

| Sinal | Tipo | Descrição Anatômica & Regras |
| :---: | :---: | :--- |
| **A** | Estático | Punho fechado, polegar ereto encostado na lateral do indicador. |
| **B** | Estático | 4 dedos estendidos para cima unidos; polegar recolhido sobre a palma. |
| **C** | Estático | Mão formando arco semicircular em concha. |
| **D** | Estático | Indicador estendido para cima; polegar toca na ponta do dedo médio em anel. |
| **E** | Estático | Todas as pontas dos dedos dobradas para baixo apoiadas sobre a palma. |
| **F** | Estático | Indicador dobrado com polegar apoiado por **FORA** da lateral; outros 3 dedos erguidos. |
| **G** | Estático | Indicador apontando para cima com polegar paralelo colado. |
| **I** | Estático | Apenas o dedo mínimo estendido para cima; demais dedos recolhidos. |
| **J** | **Dinâmico** | Dedo mínimo erguido desenhando a curva de um gancho no ar. |
| **K** | Estático | Indicador para cima, médio projetado para frente, polegar entre eles. |
| **L** | Estático | Polegar e indicador abertos formando ângulo reto de 90°. |
| **M** | Estático | 3 dedos voltados estritamente para baixo ($v_{hand\_y} \ge -0.05$). |
| **N** | Estático | 2 dedos voltados estritamente para baixo ($v_{hand\_y} \ge -0.05$). |
| **O** | Estático | Todas as pontas dos dedos unidas à ponta do polegar em círculo. |
| **P** | Estático | Indicador para frente, médio e polegar formando 'K' com mão na horizontal. |
| **Q** | Estático | Indicador e polegar estendidos paralelamente apontando para **baixo** ($v_{hand\_y} \ge -0.05$). |
| **R** | Estático | Indicador e médio estendidos para cima e **cruzados em hélice**. |
| **S** | Estático | Punho fechado com o polegar cruzado **sobre a frente** dos dedos. |
| **T** | Estático | Indicador dobrado com polegar inserido por **DENTRO** (entre indicador e médio). |
| **U** | Estático | Indicador e médio estendidos para cima, paralelos e **juntos**. |
| **V** | Estático | Indicador e médio estendidos para cima e **separados em V**. |
| **W** | Estático | Indicador, médio e anelar estendidos para cima e espalhados. |
| **X** | Estático | Indicador curvado em gancho com demais dedos fechados. |
| **Y** | Estático | Polegar e mínimo estendidos para fora (sinal Hang Loose / Shaka). |
| **Z** | **Dinâmico** | Dedo indicador estendido desenhando um 'Z' no ar em 3 traços contínuos. |
| **ESPAÇO** | Comando | Mão totalmente aberta e espalmada de frente para a câmera. |
| **APAGAR** | Comando | Punho fechado com polegar apontando para baixo (*Thumbs Down*). |

### Sinais Léxicos, Bimanuais e Articulação Facial

| Palavra / Sinal | Categoria | Descrição Anatômica & Regras |
| :---: | :---: | :--- |
| **CASA** | **Bimanual** | Ambas as mãos abertas (B) unidas pelas pontas dos dedos formando o telhado de uma casa. |
| **AJUDA** | **Bimanual** | Uma mão aberta plana horizontal (base) e a outra em punho com polegar sobre ela. |
| **TRABALHO** | **Bimanual** | Duas mãos em configuração 'L' apontando para baixo movimentando-se no espaço neutro. |
| **FAMÍLIA** | **Bimanual** | Duas mãos em configuração 'F' unidas pelos polegares e indicadores. |
| **POR FAVOR** | **Bimanual** | Ambas as mãos abertas espalmadas unidas em frente ao peito (súplica/pedido). |
| **NOME** | **Bimanual** | Ambas as mãos em configuração 'U' tocando uma na outra horizontalmente. |
| **APLAUSOS** | **Bimanual** | Ambas as mãos no alto acenando com dedos abertos (palmas de Libras). |
| **ÁGUA** | **Facial** | Mão em formato de 'L' tocando com o dedo indicador na ponta do queixo. |
| **DESCULPA** | **Facial** | Mão em formato de 'Y' encostada na lateral do queixo / mandíbula. |
| **OBRIGADO** | **Facial** | Mão aberta espalmada tocando a testa / queixo e saindo para frente. |
| **BOM** | **Facial** | Mão em concha na boca abrindo para frente espalmada. |
| **SABER** | **Facial** | Mão tocando a lateral da testa com o indicador/palma. |
| **TE AMO** | **Dêitico** | Configuração ILY: polegar, indicador e mínimo erguidos simultaneamente. |
| **LEGAL** | **Dinâmico** | Polegar ereto para cima (joinha) oscilando no espaço neutro. |
| **OI** | Dinâmico | Transição rápida e fluida da configuração 'O' para a configuração 'I' com aceno. |
| **SIM** | Dinâmico | Punho fechado na configuração 'S' balançando verticalmente (aceno afirmativo). |
| **NÃO** | Dinâmico | Dedo indicador estendido oscilando horizontalmente de um lado para o outro. |
| **EU** | Dêitico | Dedo indicador apontando diretamente para o próprio peito. |
| **VOCÊ** | Dêitico | Dedo indicador apontando diretamente para a câmera/interlocutor. |

---


## 4. Atalhos Interativos de Teclado & Controles

| Tecla / Controle | Ação |
| :---: | :--- |
| **`1`, `2`, `3`** | Seleciona e insere instantaneamente uma das **palavras sugeridas** pelo autocompletar. |
| **`L`** | Alterna entre o **Modo Legendas Cinemático** (apenas legendas limpas) e o **HUD Técnico Completo**. |
| **`T`** | **Copia** o texto ou a frase em **português fluente** para a Área de Transferência (`Clipboard`). |
| **`V`** | **Vocaliza** em voz alta a **frase inteira traduzida** usando o sintetizador PT-BR offline. |
| **`M`** | Alterna entre **Mudo** e **Áudio Ativado** (bips sensoriais e sintetizador). |
| **`K`** | Inicia ou cancela o **Modo de Calibração Biométrica** individual da mão. |
| **`H`** | Abre ou fecha a janela do **Guia Visual de Sinais de Libras**. |
| **`ESPAÇO`** | Insere um espaço (e pronuncia a palavra recém-concluída). |
| **`BACKSPACE`** | Apaga o último caractere digitado no buffer. |
| **`C`** | Limpa todo o texto acumulado na sessão. |
| **`S`** | Salva a transcrição no arquivo local `transcricao_libras.txt`. |
| **`Q` ou `ESC`** | Encerra a aplicação com segurança. |

---

## 5. Como Executar

### Modo Padrão: Libras Studio Pro (Aplicação Desktop Executiva)
Inicia o aplicativo com interface desktop moderna, viewport de vídeo em tempo real, painel lateral de transcrição com timestamps, telemetria de WPM, botões de cópia e exportação para `.SRT` e `.TXT`:

```powershell
python main.py
```

### Modo Clássico: Janela Única OpenCV (HUD Direto)
Executa na janela tradicional do OpenCV acelerada com a nova tipografia Segoe UI e painéis Frosted Glass:

```powershell
python main.py --opencv
```

Opções de linha de comando adicionais:
* `--width <px>`: Largura da captura (padrão: `1280` para HD 720p).
* `--height <px>`: Altura da captura (padrão: `720` para HD 720p).
* `--guide`: Abre automaticamente a janela com o catálogo de sinais na inicialização.
* `--camera <id>`: Seleciona o índice da câmera (padrão: `0`).
* `--interval <seg>`: Ajusta o tempo de sustentação necessário para confirmação (padrão: `0.80s`).


---

## 6. Exportação de Legendas Profissionais (.SRT e .TXT)

No painel lateral do **Libras Studio Pro**, utilize:
* **"Salvar TXT"**: Grava o texto integral da transcrição em `transcricao_libras.txt`.
* **"Exportar .SRT"**: Gera um arquivo de legendas no formato padrão de radiodifusão e cinema (SubRip `.srt`) com marcações temporais automáticas de cada bloco proferido (`HH:MM:SS,mmm --> HH:MM:SS,mmm`), pronto para importar no Adobe Premiere, DaVinci Resolve, VLC ou YouTube.

---

## 7. Testes Automatizados

Para rodar a suíte completa de testes (heurística geométrica dos 26 sinais, detecção de punhos A/S/E, diferenciação estrita de M/N, sinais bimanuais, faciais, tradutor sintático, UIRenderer com Pillow, modo legendas e gerador SRT):

```powershell
python test_libras.py
```

