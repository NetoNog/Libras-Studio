"""
Buffer de Composição de Texto com Mecanismo de Debounce e Persistência Temporal.
Acumula sinais sustentados continuamente pelo intervalo configurado (ex.: 1.0s),
filtrando ruídos e instabilidades momentâneas através de uma janela móvel (deque).
"""

import time
from collections import Counter, deque
from typing import Optional, Tuple


class TextBuffer:
    """
    Gerencia a confirmação de letras sustentadas e formação de palavras/frases.
    """

    def __init__(self,
                 commit_interval_sec: float = 1.0,
                 history_size: int = 25,
                 min_stability: float = 0.70,
                 auto_repeat_interval_sec: float = 2.0):
        self.commit_interval = commit_interval_sec
        self.history_size = history_size
        self.min_stability = min_stability
        self.auto_repeat_interval = auto_repeat_interval_sec

        # Fila circular para filtragem de ruído de alta frequência
        self.history = deque(maxlen=history_size)

        # Estado da letra candidata atual
        self.current_candidate: Optional[str] = None
        self.candidate_start_time: float = 0.0
        self.last_commit_time: float = 0.0
        self.last_committed_letter: Optional[str] = None

        # Texto acumulado e palavras
        self.text_history = []  # Lista de caracteres
        self.current_text: str = ""

        # Métricas em tempo real para o HUD
        self.current_stability: float = 0.0
        self.current_progress: float = 0.0  # 0.0 a 1.0

    def update(self,
               detected_letter: Optional[str],
               confidence: float = 1.0,
               timestamp: Optional[float] = None,
               is_transition: bool = False) -> Optional[str]:
        """
        Recebe a detecção do frame atual e atualiza o estado temporal do buffer.
        Descarta frames com confiança menor que 0.70 ou marcados em transição cinética rápida.
        """
        now = time.time() if timestamp is None else timestamp

        # Se a confiança for baixa ou se a mão estiver em transição cinética, descarta como ruído
        filtered_letter = (detected_letter if (detected_letter and confidence >= 0.70 and not is_transition) else None)

        # Adiciona a detecção atual ao histórico deslizante
        self.history.append(filtered_letter)

        # 1. Análise de estabilidade na janela temporal recente
        valid_items = [item for item in self.history if item is not None]
        if not valid_items:
            # Mão ausente ou em transição
            self.current_stability = 0.0
            self.current_progress = 0.0
            self.current_candidate = None
            return None

        # Encontra a letra mais frequente na janela
        counts = Counter(valid_items)
        dominant_letter, count = counts.most_common(1)[0]
        stability = count / len(self.history)
        self.current_stability = stability

        if stability < self.min_stability:
            # Não atingiu o limiar de estabilidade: sinal em transição rápida
            # Decai o progresso suavemente sem zerar bruscamente o temporizador do sinal
            self.current_progress = max(0.0, self.current_progress - 0.05)
            return None

        # 2. Mecanismo de Debounce e Acúmulo Temporal (Sinal Estável com Histerese)
        if dominant_letter != self.current_candidate:
            # Inicia imediatamente se vazio, ou requer contagem mínima para trocar de candidato existente
            if self.current_candidate is None or count >= 3:
                self.current_candidate = dominant_letter
                self.candidate_start_time = now
                self.current_progress = 0.0
            return None



        # Mesmo sinal mantido continuamente com estabilidade
        elapsed = now - self.candidate_start_time
        self.current_progress = min(1.0, elapsed / self.commit_interval)

        # 3. Verificação de Confirmação (Sinal sustentado pelo intervalo configurado)
        if self.current_progress >= 1.0:
            # Condição para evitar commits repetidos indesejados da mesma letra segurada
            can_commit = False
            if dominant_letter != self.last_committed_letter:
                can_commit = True
            elif (now - self.last_commit_time) >= self.auto_repeat_interval:
                can_commit = True

            if can_commit:
                self._apply_commit(dominant_letter)
                self.last_commit_time = now
                self.last_committed_letter = dominant_letter
                self.candidate_start_time = now
                self.current_progress = 0.0
                return dominant_letter

        return None

    def _apply_commit(self, command_or_letter: str):
        """Aplica a letra ou comando ao texto acumulado."""
        if command_or_letter == "ESPAÇO":
            if self.current_text and not self.current_text.endswith(" "):
                self.current_text += " "
        elif command_or_letter == "APAGAR":
            self.backspace()
        elif command_or_letter == "LIMPAR":
            self.clear()
        else:
            # Letra convencional
            self.current_text += command_or_letter

    def backspace(self):
        """Remove o último caractere do texto."""
        if len(self.current_text) > 0:
            self.current_text = self.current_text[:-1]

    def clear(self):
        """Limpa todo o texto acumulado."""
        self.current_text = ""
        self.last_committed_letter = None
        self.current_candidate = None
        self.current_progress = 0.0
        self.history.clear()

    def commit_word(self, word: str):
        """
        Insere uma palavra inteira (sinal léxico ou bimanual),
        garantindo pontuação/espaçamento adequado e limpando buffers de ruído.
        """
        word = word.strip()
        if not word:
            return
        if self.current_text and not self.current_text.endswith(" "):
            self.current_text += " " + word + " "
        else:
            self.current_text += word + " "

        self.last_committed_letter = word
        self.current_candidate = None
        self.current_progress = 0.0
        self.history.clear()

    def get_current_word(self) -> str:
        """Retorna a última palavra sendo formada."""
        words = self.current_text.split(" ")
        return words[-1] if words else ""

    def get_display_text(self, max_length: int = 40) -> str:
        """Retorna o texto visível para o HUD, truncado com reticências se exceder."""
        if len(self.current_text) <= max_length:
            return self.current_text
        return "..." + self.current_text[-(max_length - 3):]
