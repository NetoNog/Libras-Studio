"""
Motor de Assinaturas Vetoriais Canônicas e Perfil Biométrico do Usuário (Profile Manager).
Extrai vetores de características densas da mão (posições canônicas 3D, distâncias interdigitais,
extensões relativas e ângulos) e realiza classificação contínua de alta precisão por proximidade de cosseno.
Permite salvar e carregar perfis de calibração personalizados para qualquer formato de mão.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from hand_tracker import HandDetection


def extract_hand_features(canonical_landmarks: np.ndarray,
                          raw_landmarks_norm: np.ndarray) -> np.ndarray:
    """
    Extrai um vetor normalizado de 44 características invariantes a escala e rotação:
    - 15 coords canônicas das pontas dos dedos (Thumb, Index, Middle, Ring, Pinky)
    - 15 coords canônicas das juntas PIP/MCP
    - 7 distâncias interdigitais normalizadas
    - 5 diferenciais de extensão longitudinal
    - 2 componentes do vetor diretor da mão na câmera
    """
    c = canonical_landmarks # Shape (21, 3)
    lm = raw_landmarks_norm # Shape (21, 3)

    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]

    # 1. Coordenadas 3D das pontas e juntas
    tip_coords = c[tips].flatten() # 15 floats
    pip_coords = c[pips].flatten() # 15 floats

    # 2. Distâncias interdigitais normalizadas
    d_ti = float(np.linalg.norm(c[4] - c[8]))
    d_tm = float(np.linalg.norm(c[4] - c[12]))
    d_tp = float(np.linalg.norm(c[4] - c[20]))
    d_im = float(np.linalg.norm(c[8] - c[12]))
    d_mr = float(np.linalg.norm(c[12] - c[16]))
    d_rp = float(np.linalg.norm(c[16] - c[20]))
    d_wi = float(np.linalg.norm(c[0] - c[8]))
    distances = np.array([d_ti, d_tm, d_tp, d_im, d_mr, d_rp, d_wi], dtype=np.float32)

    # 3. Diferenciais de extensão longitudinal (Y_tip - Y_pip)
    extensions = np.array([
        c[4, 1] - c[2, 1],
        c[8, 1] - c[6, 1],
        c[12, 1] - c[10, 1],
        c[16, 1] - c[14, 1],
        c[20, 1] - c[18, 1]
    ], dtype=np.float32)

    # 4. Vetor diretor no espaço de imagem (câmera)
    v_dir = np.array([
        lm[9, 1] - lm[0, 1], # Y da câmera (altura do pulso para o nódulo do meio)
        lm[9, 0] - lm[0, 0]  # X da câmera (inclinação lateral)
    ], dtype=np.float32)

    vec = np.concatenate([tip_coords, pip_coords, distances, extensions, v_dir])
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec.astype(np.float32)


class SignProfileManager:
    """
    Gerenciador de assinaturas de sinais de Libras e calibração por usuário.
    Compara o vetor atual com as assinaturas de referência usando similaridade de cosseno.
    """

    PROFILE_PATH = "user_sign_signatures.json"

    def __init__(self):
        # Assinaturas de referência (Nome do Sinal -> Vetor de 44 características normalizado)
        self.reference_signatures: Dict[str, np.ndarray] = {}
        # Assinaturas personalizadas gravadas pelo usuário
        self.user_signatures: Dict[str, np.ndarray] = {}
        # Buffer de gravação rápida
        self.recording_sign: Optional[str] = None
        self.recording_samples: List[np.ndarray] = []

        self.generate_baseline_reference_signatures()
        self.load_profile()

    def generate_baseline_reference_signatures(self):
        """
        Gera assinaturas de referência pré-computadas baseadas na morfologia canônica
        de Libras para os sinais principais do alfabeto e comandos.
        """
        signs = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M',
            'N', 'O', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'ESPAÇO'
        ]
        for s in signs:
            c, raw = self._build_prototypical_canonical_landmarks(s)
            feat = extract_hand_features(c, raw)
            self.set_reference_signature(s, feat)

    def _build_prototypical_canonical_landmarks(self, sign: str) -> Tuple[np.ndarray, np.ndarray]:
        """Constrói uma representação anatômica sintética canônica para um sinal."""
        c = np.zeros((21, 3), dtype=np.float32)
        c[0] = [0.0, 0.0, 0.0]
        c[1] = [-0.22, 0.25, -0.05]
        c[2] = [-0.42, 0.50, -0.05]
        c[3] = [-0.55, 0.72, -0.02]
        c[5] = [-0.30, 0.90, 0.0]
        c[9] = [0.0, 1.0, 0.0]
        c[13] = [0.26, 0.92, 0.0]
        c[17] = [0.46, 0.84, 0.0]

        # Padrão: dedos recolhidos contra a palma (fist fechado)
        c[6] = [-0.30, 1.15, 0.15]; c[7] = [-0.30, 0.90, 0.25]; c[8] = [-0.30, 0.70, 0.20]
        c[10] = [0.0, 1.25, 0.15]; c[11] = [0.0, 0.95, 0.25]; c[12] = [0.0, 0.72, 0.20]
        c[14] = [0.26, 1.15, 0.15]; c[15] = [0.26, 0.90, 0.25]; c[16] = [0.26, 0.70, 0.20]
        c[18] = [0.46, 1.05, 0.15]; c[19] = [0.46, 0.85, 0.25]; c[20] = [0.46, 0.68, 0.20]
        c[4] = [-0.15, 0.70, 0.10]

        v_dir_y = -0.5

        if sign == 'A':
            c[4] = [-0.35, 0.92, 0.08]
        elif sign == 'S':
            c[4] = [0.05, 0.82, 0.20]
        elif sign == 'E':
            c[4] = [-0.10, 0.65, 0.15]
        elif sign == 'B':
            c[6]=[-0.30,1.30,0.0]; c[7]=[-0.30,1.65,0.0]; c[8]=[-0.30,1.95,0.0]
            c[10]=[0.0,1.45,0.0]; c[11]=[0.0,1.82,0.0]; c[12]=[0.0,2.15,0.0]
            c[14]=[0.26,1.33,0.0]; c[15]=[0.26,1.66,0.0]; c[16]=[0.26,1.95,0.0]
            c[18]=[0.46,1.18,0.0]; c[19]=[0.46,1.44,0.0]; c[20]=[0.46,1.68,0.0]
            c[4]=[-0.05, 0.65, 0.10]
        elif sign == 'L':
            c[4] = [-0.75, 0.85, 0.0]
            c[6]=[-0.30,1.30,0.0]; c[7]=[-0.30,1.65,0.0]; c[8]=[-0.30,1.95,0.0]
        elif sign == 'I':
            c[18]=[0.46,1.18,0.0]; c[19]=[0.46,1.44,0.0]; c[20]=[0.46,1.68,0.0]
        elif sign == 'Y':
            c[4] = [-0.75, 0.85, 0.0]
            c[18]=[0.46,1.18,0.0]; c[19]=[0.46,1.44,0.0]; c[20]=[0.46,1.68,0.0]
        elif sign == 'V':
            c[6]=[-0.30,1.30,0.0]; c[7]=[-0.30,1.65,0.0]; c[8]=[-0.30,1.95,0.0]
            c[10]=[0.08,1.45,0.0]; c[11]=[0.16,1.82,0.0]; c[12]=[0.24,2.15,0.0]
        elif sign == 'U':
            c[6]=[-0.30,1.30,0.0]; c[7]=[-0.30,1.65,0.0]; c[8]=[-0.30,1.95,0.0]
            c[10]=[-0.05,1.45,0.0]; c[11]=[-0.05,1.82,0.0]; c[12]=[-0.05,2.15,0.0]
        elif sign == 'W':
            c[6]=[-0.30,1.30,0.0]; c[7]=[-0.30,1.65,0.0]; c[8]=[-0.30,1.95,0.0]
            c[10]=[0.0,1.45,0.0]; c[11]=[0.0,1.82,0.0]; c[12]=[0.0,2.15,0.0]
            c[14]=[0.26,1.33,0.0]; c[15]=[0.26,1.66,0.0]; c[16]=[0.26,1.95,0.0]
        elif sign == 'D':
            c[6]=[-0.30,1.30,0.0]; c[7]=[-0.30,1.65,0.0]; c[8]=[-0.30,1.95,0.0]
            c[4]=[-0.05, 0.70, 0.15]
        elif sign == 'F':
            c[10]=[0.0,1.45,0.0]; c[11]=[0.0,1.82,0.0]; c[12]=[0.0,2.15,0.0]
            c[14]=[0.26,1.33,0.0]; c[15]=[0.26,1.66,0.0]; c[16]=[0.26,1.95,0.0]
            c[18]=[0.46,1.18,0.0]; c[19]=[0.46,1.44,0.0]; c[20]=[0.46,1.68,0.0]
            c[4]=[-0.45, 0.70, 0.10]
        elif sign == 'T':
            c[10]=[0.0,1.45,0.0]; c[11]=[0.0,1.82,0.0]; c[12]=[0.0,2.15,0.0]
            c[14]=[0.26,1.33,0.0]; c[15]=[0.26,1.66,0.0]; c[16]=[0.26,1.95,0.0]
            c[18]=[0.46,1.18,0.0]; c[19]=[0.46,1.44,0.0]; c[20]=[0.46,1.68,0.0]
            c[4]=[-0.10, 0.72, 0.15]
        elif sign == 'M':
            c[6]=[-0.30,0.60,0.0]; c[7]=[-0.30,0.30,0.0]; c[8]=[-0.30,0.05,0.0]
            c[10]=[0.0,0.65,0.0]; c[11]=[0.0,0.32,0.0]; c[12]=[0.0,0.05,0.0]
            c[14]=[0.26,0.60,0.0]; c[15]=[0.26,0.30,0.0]; c[16]=[0.26,0.05,0.0]
            v_dir_y = 0.2
        elif sign == 'N':
            c[6]=[-0.30,0.60,0.0]; c[7]=[-0.30,0.30,0.0]; c[8]=[-0.30,0.05,0.0]
            c[10]=[0.0,0.65,0.0]; c[11]=[0.0,0.32,0.0]; c[12]=[0.0,0.05,0.0]
            v_dir_y = 0.2
        elif sign == 'Q':
            c[6]=[-0.30,0.60,0.0]; c[7]=[-0.30,0.30,0.0]; c[8]=[-0.30,0.05,0.0]
            c[4]=[-0.40, 0.35, 0.0]
            v_dir_y = 0.2
        elif sign == 'X':
            c[6]=[-0.30, 1.25, 0.0]; c[7]=[-0.30, 1.35, 0.20]; c[8]=[-0.30, 1.10, 0.25]
        elif sign == 'C':
            c[6]=[-0.30, 1.25, 0.10]; c[7]=[-0.30, 1.40, 0.25]; c[8]=[-0.30, 1.30, 0.35]
            c[10]=[0.0, 1.35, 0.10]; c[11]=[0.0, 1.50, 0.25]; c[12]=[0.0, 1.38, 0.35]
            c[4]=[-0.45, 0.90, 0.25]
        elif sign == 'O':
            c[6]=[-0.30, 1.25, 0.15]; c[7]=[-0.30, 1.35, 0.30]; c[8]=[-0.20, 1.10, 0.35]
            c[10]=[0.0, 1.25, 0.15]; c[11]=[0.0, 1.35, 0.30]; c[12]=[-0.15, 1.10, 0.35]
            c[4]=[-0.20, 1.05, 0.30]
        elif sign == 'G':
            c[6]=[-0.30, 1.30, 0.0]; c[7]=[-0.30, 1.65, 0.0]; c[8]=[-0.30, 1.95, 0.0]
            c[4]=[-0.50, 1.20, 0.0]
        elif sign == 'R':
            c[6]=[-0.30, 1.30, 0.0]; c[7]=[-0.15, 1.65, 0.05]; c[8]=[-0.05, 1.95, 0.10]
            c[10]=[0.0, 1.30, 0.0]; c[11]=[-0.15, 1.65, -0.05]; c[12]=[-0.25, 1.95, -0.10]
        elif sign == 'ESPAÇO':
            c[6]=[-0.30,1.30,0.0]; c[7]=[-0.30,1.65,0.0]; c[8]=[-0.30,1.95,0.0]
            c[10]=[0.0,1.45,0.0]; c[11]=[0.0,1.82,0.0]; c[12]=[0.0,2.15,0.0]
            c[14]=[0.26,1.33,0.0]; c[15]=[0.26,1.66,0.0]; c[16]=[0.26,1.95,0.0]
            c[18]=[0.46,1.18,0.0]; c[19]=[0.46,1.44,0.0]; c[20]=[0.46,1.68,0.0]
            c[4]=[-0.60, 0.85, 0.0]
            v_dir_y = 0.05

        raw = c.copy()
        raw[9, 1] = raw[0, 1] + v_dir_y
        return c, raw

    def set_reference_signature(self, sign_name: str, feature_vector: np.ndarray):
        """Registra a assinatura de referência padrão para uma letra ou comando."""
        norm = np.linalg.norm(feature_vector)
        if norm > 1e-6:
            feature_vector = feature_vector / norm
        self.reference_signatures[sign_name] = feature_vector.astype(np.float32)

    def is_calibrated(self, sign_name: str) -> bool:
        """Verifica se o usuário possui calibração pessoal salva para um sinal."""
        return sign_name in self.user_signatures

    def get_calibrated_signs(self) -> List[str]:
        """Retorna a lista de sinais que possuem calibração personalizada ativa."""
        return sorted(list(self.user_signatures.keys()))

    def reset_sign(self, sign_name: str) -> bool:
        """Remove a assinatura personalizada do usuário para um sinal específico."""
        if sign_name in self.user_signatures:
            del self.user_signatures[sign_name]
            self.save_profile()
            return True
        return False

    def clear_all_user_signatures(self):
        """Limpa todas as assinaturas personalizadas e salva o perfil limpo."""
        self.user_signatures.clear()
        self.save_profile()

    def start_recording(self, sign_name: str):
        """Inicia a gravação de amostras personalizadas para um sinal."""
        self.recording_sign = sign_name
        self.recording_samples.clear()
        print(f"[ProfileManager] Gravando assinatura para '{sign_name}'...")

    def add_recording_sample(self, feature_vector: np.ndarray) -> int:
        """Adiciona uma amostra da mão durante a calibração do sinal."""
        if self.recording_sign:
            self.recording_samples.append(feature_vector.copy())
            return len(self.recording_samples)
        return 0

    def finish_recording(self) -> bool:
        """Calcula o vetor médio das amostras e salva para o sinal gravado."""
        if not self.recording_sign or not self.recording_samples:
            self.recording_sign = None
            return False

        samples_mat = np.array(self.recording_samples)
        mean_vec = np.mean(samples_mat, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-6:
            mean_vec = mean_vec / norm

        self.user_signatures[self.recording_sign] = mean_vec.astype(np.float32)
        print(f"[ProfileManager] Assinatura personalizada para '{self.recording_sign}' salva com sucesso!")
        self.recording_sign = None
        self.recording_samples.clear()
        self.save_profile()
        return True

    def save_profile(self, path: Optional[str] = None):
        """Persiste as assinaturas personalizadas do usuário em arquivo JSON."""
        file_path = path if path else self.PROFILE_PATH
        data = {k: v.tolist() for k, v in self.user_signatures.items()}
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[ProfileManager] Perfil salvo em '{file_path}'.")
        except Exception as e:
            print(f"[ProfileManager] Erro ao salvar perfil: {e}")

    def load_profile(self, path: Optional[str] = None) -> bool:
        """Carrega assinaturas personalizadas gravadas anteriormente."""
        file_path = path if path else self.PROFILE_PATH
        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.user_signatures = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
            print(f"[ProfileManager] Perfil carregado com {len(self.user_signatures)} assinaturas personalizadas.")
            return True
        except Exception as e:
            print(f"[ProfileManager] Erro ao carregar perfil: {e}")
            return False

    def get_best_match(self,
                       current_features: np.ndarray,
                       candidate_signs: Optional[List[str]] = None) -> Tuple[Optional[str], float]:
        """
        Calcula a similaridade de cosseno com as assinaturas de referência/personalizadas.
        Retorna (melhor_sinal, pontuacao_similaridade_0_a_1).
        """
        # Prioriza a assinatura do usuário se existir, senão usa a referência
        db = {}
        db.update(self.reference_signatures)
        db.update(self.user_signatures)

        if not db:
            return None, 0.0

        if candidate_signs:
            db = {k: v for k, v in db.items() if k in candidate_signs}

        if not db:
            return None, 0.0

        best_sign = None
        best_sim = -1.0

        for sign_name, ref_vec in db.items():
            sim = float(np.dot(current_features, ref_vec))
            if sim > best_sim:
                best_sim = sim
                best_sign = sign_name

        # Mapeia [-1.0, 1.0] para [0.0, 1.0]
        conf = max(0.0, min(1.0, (best_sim + 1.0) / 2.0))
        return best_sign, conf
