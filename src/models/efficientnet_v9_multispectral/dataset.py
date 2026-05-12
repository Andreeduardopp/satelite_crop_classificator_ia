"""
Dataset PyTorch do V9 — TIFF multispectral 6 bandas.

Substitui o `TemporalCulturaDataset` do V7 (que lia PNG via cv2). As mudanças
relevantes:

  - Lê TIFF FLOAT32 6 bandas (B02, B03, B04, B08, B11, B12) com tifffile.
  - Resize via cv2 ainda funciona em arrays float32 multi-canal.
  - Normalização por banda usa BAND_MEAN/BAND_STD do dataset, não do ImageNet.
  - Augmentation D4 (flip H/V + rot90) preservada e ainda compartilhada por
    timesteps para manter alinhamento temporal.
  - Carrega registros do `dados_v9.db` / coluna `tiffs_processados`.
"""

import ast
import os
import re
import sqlite3

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BAND_NAMES = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
NUM_BANDS  = len(BAND_NAMES)

# ⚠️ PLACEHOLDERS — rodar `compute_band_stats.py` após baixar dataset e colar a saída.
# Os valores abaixo são chutes razoáveis para Sentinel-2 L2A reflectance em zonas
# agrícolas brasileiras; suficientes para um teste de sanidade mas não para um
# treino final. NÃO esqueça de substituir antes de rodar o treino sério.
BAND_MEAN = np.array([0.105, 0.110, 0.118, 0.260, 0.220, 0.150], dtype=np.float32)
BAND_STD  = np.array([0.060, 0.060, 0.075, 0.095, 0.085, 0.075], dtype=np.float32)

IMG_SIZE     = (224, 224)
MAX_SEQ_LEN  = 5            # ↑ vs 3 do V7 — pipeline_tiff baixa 9 DAPs/talhão
MAX_DIA      = 100.0

CLASSES = ['soja', 'milho', 'trigo', 'aveia', 'feijão']

DB_NOME    = 'dados_v9.db'
DB_TABELA  = 'culturas_tiff'


# ---------------------------------------------------------------------------
# Carregamento do DB
# ---------------------------------------------------------------------------

def extrair_dia(caminho: str) -> int:
    m = re.search(r'_d(\d+)\.tiff$', caminho)
    return int(m.group(1)) if m else 0


def carregar_dados(db_path: str = DB_NOME, src_dir: str = '.'):
    """
    Lê o banco V9 e retorna (registros, labels, meses).

    Cada registro é uma lista [(caminho_tiff, dia_normalizado), ...] ordenada
    crescentemente por DAP. Talhões sem nenhum TIFF válido em disco são
    descartados.
    """
    classe_para_id = {c: i for i, c in enumerate(CLASSES)}
    classe_para_id['feijao'] = classe_para_id['feijão']

    registros, labels, meses = [], [], []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT cultura, mes, tiffs_processados FROM {DB_TABELA}"
        ).fetchall()

    for cultura, mes, tiffs_str in rows:
        if cultura not in classe_para_id:
            continue
        try:
            paths = ast.literal_eval(tiffs_str)
        except (ValueError, SyntaxError):
            continue

        validos = []
        for p in paths:
            abs_p = os.path.join(src_dir, p) if not os.path.isabs(p) else p
            if os.path.exists(abs_p):
                validos.append((abs_p, extrair_dia(abs_p)))

        if validos:
            validos.sort(key=lambda x: x[1])
            registros.append(validos)
            labels.append(classe_para_id[cultura])
            meses.append(int(mes) if mes else 1)

    return registros, labels, meses


# ---------------------------------------------------------------------------
# Pré-processamento de uma imagem
# ---------------------------------------------------------------------------

def preprocessar_tiff(path: str) -> np.ndarray:
    """
    Lê um TIFF (H, W, C) FLOAT32 e devolve (C, H', W') normalizado, com
    H'×W' = IMG_SIZE.

    - Reamostra com cv2.resize (INTER_AREA: bom para reduzir resolução).
    - Normaliza por banda: (x - mean) / std, com BAND_MEAN/BAND_STD acima.
    - Em caso de falha de leitura, retorna zeros (o caller marca via mask).
    """
    try:
        img = tifffile.imread(path)        # (H, W, C) FLOAT32 esperado
    except Exception:
        return np.zeros((NUM_BANDS, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)

    if img.ndim == 2:
        img = img[:, :, None]
    elif img.shape[0] == NUM_BANDS and img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))

    if img.shape[-1] != NUM_BANDS:
        # Banda inesperada — não dá para normalizar, retorna zero
        return np.zeros((NUM_BANDS, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)

    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = (img - BAND_MEAN) / BAND_STD     # broadcast em (H, W, C)
    return np.transpose(img, (2, 0, 1)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TemporalCulturaTiffDataset(Dataset):
    """
    Cada sample: (images[T,C,H,W], dias[T], mes, mask[T], label).
    Sequências menores que MAX_SEQ_LEN recebem padding zero + mask 0.
    Augmentation D4 sorteada uma vez por sequência (preserva alinhamento
    temporal entre timesteps).
    """

    def __init__(self, registros, labels, meses, augment: bool = False):
        self.registros = registros
        self.labels    = labels
        self.meses     = meses
        self.augment   = augment

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, idx):
        items   = self.registros[idx]
        label   = self.labels[idx]
        mes     = self.meses[idx]
        seq_len = min(len(items), MAX_SEQ_LEN)

        images = np.zeros(
            (MAX_SEQ_LEN, NUM_BANDS, IMG_SIZE[0], IMG_SIZE[1]),
            dtype=np.float32,
        )
        dias = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

        if self.augment:
            flip_h = np.random.rand() < 0.5
            flip_v = np.random.rand() < 0.5
            rot_k  = np.random.randint(0, 4)
        else:
            flip_h = flip_v = False
            rot_k = 0

        # Se a sequência for maior que MAX_SEQ_LEN, amostra MAX_SEQ_LEN índices
        # uniformemente espaçados — preserva começo, meio e fim.
        if len(items) > MAX_SEQ_LEN:
            idxs = np.linspace(0, len(items) - 1, MAX_SEQ_LEN).astype(int)
            items_sel = [items[i] for i in idxs]
        else:
            items_sel = items[:seq_len]

        for i, (caminho, dia) in enumerate(items_sel):
            img = preprocessar_tiff(caminho)
            if flip_h:
                img = img[:, :, ::-1]
            if flip_v:
                img = img[:, ::-1, :]
            if rot_k:
                img = np.rot90(img, k=rot_k, axes=(1, 2))
            images[i] = np.ascontiguousarray(img)
            dias[i] = dia / MAX_DIA
            mask[i] = 1.0

        return (
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(dias, dtype=torch.float32),
            torch.tensor(mes - 1, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )
