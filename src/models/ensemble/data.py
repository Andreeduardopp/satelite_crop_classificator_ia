"""
Módulo compartilhado de dados para o ensemble.
Replica a lógica de carregamento do V7 para evitar dependências de import.
"""

import os
import re
import ast
import sqlite3
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# -- Paths ---------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(_HERE, '..', '..'))
ROOT_DIR = os.path.normpath(os.path.join(SRC_DIR, '..'))

# -- Constantes ----------------------------------------------------------------
CLASSES     = ['soja', 'milho', 'trigo', 'aveia', 'feijão']
TABELA      = 'culturas'
IMG_SIZE    = (224, 224)
MAX_SEQ_LEN = 3
MAX_DIA     = 100.0
SEED        = 42

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# Auto-detectar maior DB de treino disponível
_DB_CANDIDATES_TREINO = [
    'sample_treino_max9000.db', 'sample_treino_6k.db',
    'sample_treino_v2.db', 'sample_treino.db',
]
DB_TREINO = None
for _name in _DB_CANDIDATES_TREINO:
    _path = os.path.join(ROOT_DIR, _name)
    if os.path.exists(_path):
        DB_TREINO = _path
        break

_DB_CANDIDATES_TESTE = ['sample_teste_250.db', 'sample_teste.db']
DB_TESTE = None
for _name in _DB_CANDIDATES_TESTE:
    _path = os.path.join(ROOT_DIR, _name)
    if os.path.exists(_path):
        DB_TESTE = _path
        break

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'


def extrair_dia(caminho: str) -> int:
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_dados(db_path: str):
    """Retorna (registros, labels, meses) a partir do banco SQLite."""
    classe_para_id = {c: i for i, c in enumerate(CLASSES)}
    classe_para_id['feijao'] = classe_para_id['feijão']

    registros, labels, meses = [], [], []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT cultura, mes, imagens_processadas FROM {TABELA}"
        ).fetchall()

    for cultura, mes, imgs_str in rows:
        if cultura not in classe_para_id:
            continue
        try:
            paths = ast.literal_eval(imgs_str)
        except (ValueError, SyntaxError):
            continue

        validos = []
        for p in paths:
            abs_p = os.path.join(SRC_DIR, p) if not os.path.isabs(p) else p
            if os.path.exists(abs_p):
                validos.append((abs_p, extrair_dia(abs_p)))

        if validos:
            validos.sort(key=lambda x: x[1])
            registros.append(validos)
            labels.append(classe_para_id[cultura])
            meses.append(int(mes) if mes else 1)

    return registros, labels, meses


def preprocessar_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    if img is None:
        return np.zeros((3, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))


class TemporalCulturaDataset(Dataset):
    """Cada sample: (images[T,C,H,W], dias[T], mes, mask[T], label)."""

    def __init__(self, registros, labels, meses):
        self.registros = registros
        self.labels = labels
        self.meses = meses

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, idx):
        items = self.registros[idx]
        label = self.labels[idx]
        mes = self.meses[idx]
        seq_len = min(len(items), MAX_SEQ_LEN)

        images = np.zeros((MAX_SEQ_LEN, 3, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
        dias = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

        for i in range(seq_len):
            caminho, dia = items[i]
            images[i] = preprocessar_imagem(caminho)
            dias[i] = dia / MAX_DIA
            mask[i] = 1.0

        return (
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(dias, dtype=torch.float32),
            torch.tensor(mes - 1, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )
