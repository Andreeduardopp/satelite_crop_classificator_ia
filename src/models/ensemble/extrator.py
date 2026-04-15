"""
Extrator de features do ensemble: carrega os 3 modelos treinados e extrai,
para cada amostra, a representação pooled + probabilidades de cada modelo.

O XGBoost recebe features de 3 perspectivas diferentes (EfficientNet, ResNet,
ConvNeXt), cada uma capturando padrões visuais distintos. Ele aprende qual
modelo confiar para cada classe.

Feature layout final (todos os backbones projetados para PROJ_DIM=512):
    [0    :  512]  EfficientNetB0 pooled            — 512
    [ 512 :  517]  EfficientNetB0 probabilities     — 5
    [ 517 : 1029]  ResNet50 pooled                  — 512
    [1029 : 1034]  ResNet50 probabilities            — 5
    [1034 : 1546]  ConvNeXt-Tiny pooled              — 512
    [1546 : 1551]  ConvNeXt-Tiny probabilities       — 5
    [1551 : 1554]  Dias                              — 3
    [1554 : 1555]  Mes                               — 1
    [1555 : 1556]  Count imagens                     — 1
    Total: 1556

Uso:
    python extrator.py
"""

import os
import sys
import time
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from data import (
    carregar_dados, TemporalCulturaDataset,
    CLASSES, DB_TREINO, DB_TESTE, DEVICE, USE_AMP, SEED
)
from model import TemporalCulturaModel, BACKBONE_CONFIG, PROJ_DIM

# -- Configurações -------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(_HERE, 'artifacts')
OUT_DIR       = os.path.join(_HERE, 'features')

BATCH_SIZE  = 64
NUM_WORKERS = 4

BACKBONES = list(BACKBONE_CONFIG.keys())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)


def carregar_modelo(backbone_name: str):
    """Instancia modelo e carrega pesos treinados."""
    peso_path = os.path.join(ARTIFACTS_DIR, f'pesos_{backbone_name}.pt')

    if not os.path.exists(peso_path):
        log.warning(f"Pesos nao encontrados para {backbone_name}: {peso_path}")
        return None

    modelo = TemporalCulturaModel(backbone_name, len(CLASSES)).to(DEVICE)
    modelo.load_state_dict(torch.load(peso_path, map_location=DEVICE))
    modelo.eval()
    log.info(f"  {backbone_name}: carregado (backbone_dim={modelo.feature_dim}, proj_dim={PROJ_DIM})")
    return modelo


@torch.no_grad()
def extrair_batch(modelo, images, dias, mes, mask):
    """Extrai pooled features e probabilidades de um modelo para um batch."""
    pooled, logits = modelo.forward_features(images, dias, mes, mask)
    probs = F.softmax(logits, dim=-1)
    return pooled.cpu().float().numpy(), probs.cpu().float().numpy()


def extrair_features(db_path: str, prefixo: str, modelos: dict):
    """
    Extrai features de todos os modelos do ensemble para um banco de dados.

    Args:
        db_path: Caminho para o banco SQLite.
        prefixo: 'treino' ou 'teste'.
        modelos: Dict {backbone_name: modelo} com modelos carregados.
    """
    log.info(f"[{prefixo}] Carregando dados: {db_path}")

    registros, labels, meses = carregar_dados(db_path)
    if not registros:
        log.error(f"[{prefixo}] Nenhum dado!")
        return

    log.info(f"[{prefixo}] {len(registros)} registros")

    dataset = TemporalCulturaDataset(registros, labels, meses)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE.type == 'cuda'))

    X_batches = []
    y_batches = []
    total = 0
    t0 = time.time()

    with autocast(device_type=DEVICE.type, enabled=USE_AMP):
        for images, dias, mes, mask, batch_labels in loader:
            B = images.shape[0]

            images_dev = images.to(DEVICE, non_blocking=True)
            dias_dev   = dias.to(DEVICE, non_blocking=True)
            mes_dev    = mes.to(DEVICE, non_blocking=True)
            mask_dev   = mask.to(DEVICE, non_blocking=True)

            # Extrair features de cada modelo
            parts = []
            for bb_name in BACKBONES:
                if bb_name not in modelos:
                    continue
                pooled, probs = extrair_batch(
                    modelos[bb_name], images_dev, dias_dev, mes_dev, mask_dev
                )
                parts.append(pooled)   # (B, PROJ_DIM)
                parts.append(probs)    # (B, 5)

            # Metadados
            dias_np = dias.numpy()                                   # (B, 3)
            mes_np  = (mes.numpy() + 1).reshape(B, 1).astype(np.float32)  # (B, 1)
            count   = mask.sum(dim=1).numpy().reshape(B, 1).astype(np.float32)  # (B, 1)

            parts.extend([dias_np, mes_np, count])

            batch_X = np.concatenate(parts, axis=1).astype(np.float32)

            X_batches.append(batch_X)
            y_batches.append(batch_labels.numpy())

            total += B
            if total % 500 == 0:
                elapsed = time.time() - t0
                log.info(f"[{prefixo}] {total} amostras ({total/elapsed:.0f}/s)")

    X = np.vstack(X_batches)
    y = np.concatenate(y_batches)

    elapsed = time.time() - t0
    log.info(f"[{prefixo}] Extração concluída: {X.shape} em {elapsed:.1f}s")

    # Validar
    assert np.all(np.isfinite(X)), f"NaN/Inf detectados em {prefixo}!"

    # Salvar
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f'X_{prefixo}.npy'), X)
    np.save(os.path.join(OUT_DIR, f'y_{prefixo}.npy'), y)
    log.info(f"[{prefixo}] Salvo em {OUT_DIR}/ ({X.nbytes/1e6:.1f} MB)")

    # Estatísticas
    unique, counts = np.unique(y, return_counts=True)
    for cid, cnt in zip(unique, counts):
        name = CLASSES[cid] if cid < len(CLASSES) else f"cls_{cid}"
        log.info(f"[{prefixo}]   {name}: {cnt} ({100*cnt/len(y):.1f}%)")


def main():
    log.info("="*70)
    log.info("Extrator de Features do Ensemble (3 Backbones)")
    log.info("="*70)

    # 1. Carregar todos os modelos treinados
    log.info("Carregando modelos treinados:")
    modelos = {}
    for bb in BACKBONES:
        m = carregar_modelo(bb)
        if m is not None:
            modelos[bb] = m

    if not modelos:
        log.error("Nenhum modelo treinado encontrado! Execute train.py --all primeiro.")
        return

    log.info(f"Modelos carregados: {list(modelos.keys())} ({len(modelos)}/{len(BACKBONES)})")

    if len(modelos) < len(BACKBONES):
        faltando = set(BACKBONES) - set(modelos.keys())
        log.warning(f"Modelos faltando: {faltando}")
        log.warning("O ensemble funcionará com os modelos disponíveis.")

    # Calcular tamanho esperado (pooled agora é PROJ_DIM para todos os backbones)
    total_features = sum(PROJ_DIM + len(CLASSES) for _ in modelos.values()) + 5
    log.info(f"Features por amostra: {total_features}")

    # 2. Extrair
    extrair_features(DB_TREINO, "treino", modelos)
    extrair_features(DB_TESTE,  "teste",  modelos)

    log.info("="*70)
    log.info("Extração concluída! Proximo: python train_xgboost.py")
    log.info("="*70)


if __name__ == '__main__':
    main()
