"""
Módulo de Predição de Palavras e Autocompletar em Português para Libras.
Permite acelerar a datilologia sugerindo as palavras mais prováveis em tempo real.
"""

from typing import List, Tuple

# Dicionário de frequência em Português com foco em conversação diária e acessibilidade
COMMON_WORDS_PT = [
    # Saudações e Cortesia
    "BOM", "BOA", "DIA", "TARDE", "NOITE", "OLA", "OI", "TUDO", "BEM",
    "COMO", "VAI", "OBRIGADO", "OBRIGADA", "POR", "FAVOR", "DESCULPA", "LICENCA",
    "PRAZER", "CONHECER", "BEM-VINDO", "ATE", "LOGO", "AMANHA",
    # Identidade e Pessoas
    "MEU", "NOME", "SEU", "VOCE", "ELE", "ELA", "NOS", "ELES", "AMIGO",
    "AMIGA", "FAMILIA", "PAI", "MAE", "FILHO", "FILHA", "IRMAO", "IRMA",
    "PROFESSOR", "ALUNO", "SURDO", "OUVINTE", "INTERPRETE", "PESSOA", "GENTE",
    # Comunicação e Libras
    "LIBRAS", "SINAL", "APRENDER", "ENSINAR", "FALAR", "COMUNICAR", "ENTENDER",
    "SABER", "CONHECER", "PALAVRA", "FRASE", "TRADUZIR", "TEXTO", "LEITURA",
    # Sentimentos e Respostas
    "SIM", "NAO", "TALVEZ", "QUERO", "GOSTO", "AMOR", "PAZ", "ALEGRIA", "FELIZ",
    "TRISTE", "CANSADO", "CALMO", "CERTO", "ERRADO", "BELEZA", "LEGAL", "OTIMO",
    # Necessidades e Ações Diárias
    "AJUDA", "PRECISO", "ONDE", "QUANDO", "QUEM", "QUAL", "PORQUE", "HORA",
    "TEMPO", "HOJE", "AGORA", "DEPOIS", "CASA", "TRABALHO", "ESCOLA", "HOSPITAL",
    "MEDICO", "REMEDIO", "AGUA", "COMIDA", "ALMOCO", "JANTAR", "BANHEIRO", "CIDADE",
    "BRASIL", "RUA", "DINHEIRO", "TELEFONE", "MENSAGEM", "VIDEO", "CHAMADA"
]


class WordPredictor:
    """
    Motor de predição e autocompletar em tempo real.
    """

    def __init__(self, vocabulary: List[str] = None):
        self.words = vocabulary if vocabulary is not None else COMMON_WORDS_PT
        # Normaliza vocabulário em maiúsculas sem duplicatas mantendo ordem
        seen = set()
        self.unique_words = []
        for w in self.words:
            w_clean = w.strip().upper()
            if w_clean and w_clean not in seen:
                seen.add(w_clean)
                self.unique_words.append(w_clean)

    def get_suggestions(self, prefix: str, max_suggestions: int = 3) -> List[str]:
        """
        Retorna as principais palavras que iniciam com o prefixo fornecido.
        """
        prefix_clean = prefix.strip().upper()
        if not prefix_clean:
            return []

        suggestions = []
        # 1. Correspondência exata de prefixo
        for w in self.unique_words:
            if w.startswith(prefix_clean) and w != prefix_clean:
                suggestions.append(w)
                if len(suggestions) >= max_suggestions:
                    break

        return suggestions

    def apply_suggestion(self, current_text: str, selected_word: str) -> str:
        """
        Substitui a última palavra parcial pelo vocábulo sugerido e adiciona um espaço.
        """
        words = current_text.split(" ")
        if not words:
            return selected_word + " "

        # Substitui a última palavra em formação
        words[-1] = selected_word.upper()
        return " ".join(words) + " "
