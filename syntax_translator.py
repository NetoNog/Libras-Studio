"""
Tradutor Sintático de Gramática de Libras para Português Fluente.
A estrutura sintática da Libras frequentemente utiliza Tópico-Comentário e OSV (Objeto-Sujeito-Verbo),
sem preposições ou artigos soltos. Este módulo reordena e conjuga a sequência de sinais
em frases naturais e gramaticalmente corretas da língua portuguesa falada.
"""

from typing import Dict, List, Optional


class SyntaxTranslator:
    """
    Traduz sequências de sinais de Libras em sentenças fluentes de Português.
    """

    # Mapeamento de padrões sintáticos de Libras para frases naturais em Português
    PATTERNS: Dict[tuple, str] = {
        ("OI",): "Olá, tudo bem?",
        ("OI", "TUDO", "BEM"): "Olá! Tudo bem com você?",
        ("BOM", "DIA"): "Bom dia!",
        ("BOA", "TARDE"): "Boa tarde!",
        ("BOA", "NOITE"): "Boa noite!",
        ("OBRIGADO",): "Muito obrigado!",
        ("OBRIGADA",): "Muito obrigada!",
        ("POR FAVOR",): "Por favor.",
        ("POR", "FAVOR"): "Por favor.",
        ("TE AMO",): "Eu te amo muito!",
        ("TE", "AMO"): "Eu te amo muito!",
        ("LEGAL",): "Que legal!",
        ("NOME",): "Meu nome é...",
        ("SEU", "NOME"): "Qual é o seu nome?",
        ("DESCULPA",): "Peço desculpas.",

        ("SIM",): "Sim, com certeza.",
        ("NAO",): "Não.",
        ("EU", "CASA", "IR"): "Eu estou indo para casa.",
        ("EU", "CASA"): "Eu vou para casa.",
        ("EU", "AJUDA"): "Eu preciso de ajuda.",
        ("EU", "AJUDA", "PRECISO"): "Eu estou precisando de ajuda.",
        ("PRECISO", "AJUDA"): "Por favor, preciso de ajuda.",
        ("VOCE", "TUDO", "BEM"): "Tudo bem com você?",
        ("VOCE", "BEM"): "Você está bem?",
        ("MEU", "NOME"): "Meu nome é ",
        ("QUERO", "AGUA"): "Eu gostaria de água, por favor.",
        ("AGUA", "QUERO"): "Eu quero água, por favor.",
        ("AGUA",): "Água, por favor.",
        ("CASA",): "Em casa.",
        ("TRABALHO",): "No trabalho.",
        ("FAMILIA",): "Minha família.",
        ("APLAUSOS",): "Parabéns! (Aplausos)",
        ("EU", "TRABALHO"): "Eu estou no trabalho.",
        ("EU", "TRABALHO", "IR"): "Eu estou indo trabalhar.",
        ("EU", "SABER"): "Eu compreendi / Eu sei.",
        ("NAO", "SABER"): "Eu não sei.",
        ("VOCE", "AJUDA"): "Você precisa de ajuda?",
        ("VOCE", "CASA"): "Você está em casa?",
        ("FAMILIA", "BOM"): "Minha família está muito bem!",
        ("DESCULPA", "ATRASO"): "Peço desculpas pelo atraso.",
        ("BOM",): "Muito bom!",
        ("APRENDER", "LIBRAS"): "Estou aprendendo Libras.",
        ("LIBRAS", "LEGAL"): "Libras é muito legal!",
        ("LIBRAS", "BOM"): "Libras é ótimo!"
    }

    def translate_sequence(self, words: List[str]) -> str:
        """
        Recebe a lista de palavras/sinais e gera uma frase coerente em Português.
        """
        clean_words = [w.strip().upper() for w in words if w.strip()]
        if not clean_words:
            return ""

        # 1. Verifica correspondência exata de tupla de sinais
        tuple_key = tuple(clean_words)
        if tuple_key in self.PATTERNS:
            return self.PATTERNS[tuple_key]

        # 2. Busca o maior sub-padrão contido no final da frase
        for n in range(min(4, len(clean_words)), 0, -1):
            sub_key = tuple(clean_words[-n:])
            if sub_key in self.PATTERNS:
                prefix = " ".join(clean_words[:-n])
                if prefix:
                    return f"{prefix.capitalize()} {self.PATTERNS[sub_key]}"
                return self.PATTERNS[sub_key]

        # 3. Formatação padrão para sentenças livres
        sentence = " ".join(clean_words).capitalize()
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence
