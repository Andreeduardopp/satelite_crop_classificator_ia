"""
Extrator Multi-Nível de Features via EfficientNet V7.

O extrator anterior só usava modelo.backbone() — features brutas (1280-dim),
ignorando completamente o FiLM conditioning e a temporal attention que o V7
aprendeu. XGBoost recebia features idênticas às que o modelo já usa, sem
informação nova → mesma acurácia.

Esta versão extrai features de MÚLTIPLOS níveis do modelo treinado:

  Nível 1 — Representação pooled (pós-attention): o que o classificador vê
  Nível 2 — Features pós-FiLM (condicionadas por dia/mês)
  Nível 3 — Multi-escala backbone (blocos intermediários do EfficientNet)
  Nível 4 — Engenharia temporal (mean, std, max, diffs entre timesteps)
  Nível 5 — Metadados (dia, mês, contagem de imagens)

Isso dá ao XGBoost informação COMPLEMENTAR que o classificador linear simples
do V7 (Linear→ReLU→Dropout→Linear) não consegue explorar.
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

# -- Injeção segura do ambiente V7 ------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(_HERE)
V7_DIR = os.path.join(MODELS_DIR, 'efficientnet_v7')
ROOT_DIR = os.path.normpath(os.path.join(MODELS_DIR, '..', '..'))

sys.path.append(V7_DIR)
try:
    from train import (
        EfficientNetTemporalV6, carregar_dados, TemporalCulturaDataset,
        CLASSES, BATCH_SIZE, NUM_WORKERS, DEVICE, USE_AMP, MAX_SEQ_LEN
    )
except ImportError as e:
    raise ImportError(f"Falha ao importar do train.py da V7: {e}")

# -- Configurações ----------------------------------------------------------------
DB_TREINO  = os.path.join(ROOT_DIR, 'sample_treino_max9000.db')
DB_TESTE   = os.path.join(ROOT_DIR, 'sample_teste_250.db')
PESOS_PATH = os.path.join(V7_DIR, 'artifacts', 'pesos.pt')
OUT_DIR    = os.path.join(_HERE, 'features_extraidas')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)


# -- Hook para captura multi-escala -----------------------------------------------

class BackboneHook:
    """Registra hooks nos blocos intermediários do EfficientNet para capturar
    features em múltiplas escalas (texturas, padrões médios, semântica)."""

    def __init__(self, backbone):
        self.features = {}
        self._hooks = []

        # Registrar nos blocos 2 (texturas ~40ch), 4 (médio ~112ch), 6 (alto ~320ch)
        for block_idx in [2, 4, 6]:
            block = backbone.blocks[block_idx]
            h = block.register_forward_hook(self._make_hook(f'block_{block_idx}'))
            self._hooks.append(h)

    def _make_hook(self, name):
        def fn(module, inp, out):
            # out é feature map espacial (B*T, C, H, W) → pool → (B*T, C)
            self.features[name] = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return fn

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# -- Forward customizado para extrair todos os níveis ----------------------------

@torch.no_grad()
def extract_multilevel(modelo, images, dias, mes, mask, hook):
    """
    Roda o forward do V7 manualmente, capturando representações intermediárias.

    Retorna dict com todos os níveis de features (já em numpy, CPU).
    """
    B, T = images.shape[0], images.shape[1]

    # ---------- Backbone (dispara os hooks) ----------
    imgs_flat = images.reshape(B * T, *images.shape[2:])     # (B*T, C, H, W)
    hook.clear()
    backbone_flat = modelo.backbone(imgs_flat)                 # (B*T, 1280)
    backbone_per_t = backbone_flat.reshape(B, T, -1)           # (B, T, 1280)

    # Capturar features intermediárias dos hooks
    multi_scale = {}
    for name, feat in hook.features.items():
        multi_scale[name] = feat.reshape(B, T, -1)            # (B, T, C)

    # ---------- FiLM conditioning (dia + mês → modula features) ----------
    mes_emb = modelo.mes_embedding(mes)                        # (B, 8)
    mes_emb = mes_emb.unsqueeze(1).expand(-1, T, -1)           # (B, T, 8)
    dia_exp = dias.unsqueeze(-1)                                # (B, T, 1)
    context = torch.cat([dia_exp, mes_emb], dim=-1)             # (B, T, 9)

    film_h = F.relu(modelo.film_hidden(context))                # (B, T, 64)
    gamma = modelo.film_gamma(film_h)                           # (B, T, 1280)
    beta = modelo.film_beta(film_h)                             # (B, T, 1280)

    post_film = backbone_per_t * (1.0 + gamma) + beta           # (B, T, 1280)

    # ---------- Temporal self-attention (2 camadas) ----------
    key_pad_mask = (mask == 0)

    attn_out1, _ = modelo.attn1(post_film, post_film, post_film,
                                 key_padding_mask=key_pad_mask)
    post_attn1 = modelo.norm1(post_film + attn_out1)            # (B, T, 1280)

    attn_out2, _ = modelo.attn2(post_attn1, post_attn1, post_attn1,
                                 key_padding_mask=key_pad_mask)
    post_attn2 = modelo.norm2(post_attn1 + attn_out2)           # (B, T, 1280)

    # ---------- Pooled (o que o classificador vê) ----------
    mask_exp = mask.unsqueeze(-1)                                # (B, T, 1)
    valid_count = mask_exp.sum(dim=1).clamp(min=1.0)             # (B, 1)
    pooled = (post_attn2 * mask_exp).sum(dim=1) / valid_count    # (B, 1280)

    # ---------- Mover tudo para CPU/numpy ----------
    def to_np(t):
        return t.cpu().float().numpy()

    return {
        'backbone_per_t': to_np(backbone_per_t),    # (B, T, 1280)
        'post_film':      to_np(post_film),          # (B, T, 1280)
        'post_attn':      to_np(post_attn2),         # (B, T, 1280)
        'pooled':         to_np(pooled),              # (B, 1280)
        'multi_scale':    {k: to_np(v) for k, v in multi_scale.items()},
        'mask':           to_np(mask),                # (B, T)
    }


# -- Engenharia de features temporais -------------------------------------------

def build_temporal_features(backbone_per_t, mask):
    """
    A partir das features por timestep (B, T, D), computa estatísticas
    temporais que capturam padrões que a attention pode perder.

    Retorna arrays (B, D) para cada estatística.
    """
    B, T, D = backbone_per_t.shape
    mask_exp = mask[:, :, np.newaxis]        # (B, T, 1)
    valid_count = mask_exp.sum(axis=1).clip(min=1.0)  # (B, 1)

    # Mascarar timesteps inválidos
    masked = backbone_per_t * mask_exp       # (B, T, D)

    # Estatísticas ao longo do eixo temporal
    temporal_mean = masked.sum(axis=1) / valid_count                           # (B, D)

    # Std com máscara — variância = E[x²] - E[x]²
    sq_mean = (masked ** 2).sum(axis=1) / valid_count
    temporal_std = np.sqrt(np.maximum(sq_mean - temporal_mean ** 2, 0.0))      # (B, D)

    # Max ao longo de T (substituir padding por -inf para não afetar)
    masked_for_max = np.where(mask_exp > 0, backbone_per_t, -1e9)
    temporal_max = masked_for_max.max(axis=1)                                  # (B, D)

    # Diferenças entre timesteps consecutivos (capturam mudança ao longo do tempo)
    # Se timestep 2 ou 3 é padding, diff será ~0 (features são 0)
    diff_1_2 = backbone_per_t[:, 1, :] - backbone_per_t[:, 0, :]              # (B, D)
    diff_2_3 = backbone_per_t[:, 2, :] - backbone_per_t[:, 1, :]              # (B, D)

    # Mascarar diffs de timesteps inválidos
    diff_1_2 *= (mask[:, 1:2])  # Válido só se timestep 1 existe
    diff_2_3 *= (mask[:, 2:3])  # Válido só se timestep 2 existe

    return temporal_mean, temporal_std, temporal_max, diff_1_2, diff_2_3


def assemble_tabular(features_dict, dias_np, mes_np, total_imgs):
    """
    Monta a matriz tabular final concatenando todos os níveis de features.

    Layout das colunas:
      [0    : 1280]   Pooled (pós-attention, pós-pooling)          — 1280
      [1280 : 2560]   Post-FiLM pooled                             — 1280
      [2560 : 3840]   Temporal mean (backbone)                     — 1280
      [3840 : 5120]   Temporal std (backbone)                      — 1280
      [5120 : 6400]   Temporal max (backbone)                      — 1280
      [6400 : 7680]   Temporal diff step1→step2                    — 1280
      [7680 : 8960]   Temporal diff step2→step3                    — 1280
      [8960 : 9000]   Multi-scale block_2 (texturas)               — 40
      [9000 : 9112]   Multi-scale block_4 (padrões)                — 112
      [9112 : 9432]   Multi-scale block_6 (semântica)              — 320
      [9432 : 9435]   Dias normalizados                            — 3
      [9435 : 9436]   Mês                                          — 1
      [9436 : 9437]   Contagem de imagens válidas                  — 1

    Total: ~9437 colunas
    """
    B = features_dict['pooled'].shape[0]
    mask = features_dict['mask']

    # 1. Pooled (a representação mais rica — pós-attention)
    pooled = features_dict['pooled']                                   # (B, 1280)

    # 2. Post-FiLM pooled (features condicionadas por dia/mês)
    mask_exp = mask[:, :, np.newaxis]
    valid_count = mask_exp.sum(axis=1).clip(min=1.0)
    film_pooled = (features_dict['post_film'] * mask_exp).sum(axis=1) / valid_count  # (B, 1280)

    # 3-7. Temporal engineering no backbone
    t_mean, t_std, t_max, diff12, diff23 = build_temporal_features(
        features_dict['backbone_per_t'], mask
    )

    # 8-10. Multi-scale intermediárias (pooled across T)
    ms_parts = []
    for key in ['block_2', 'block_4', 'block_6']:
        ms_feat = features_dict['multi_scale'][key]   # (B, T, C)
        ms_pooled = (ms_feat * mask_exp[:, :, :1]).sum(axis=1) / valid_count  # (B, C)
        ms_parts.append(ms_pooled)

    # 11-13. Metadados
    parts = [
        pooled,           # 1280
        film_pooled,      # 1280
        t_mean,           # 1280
        t_std,            # 1280
        t_max,            # 1280
        diff12,           # 1280
        diff23,           # 1280
        *ms_parts,        # 40 + 112 + 320
        dias_np,          # 3
        mes_np,           # 1
        total_imgs,       # 1
    ]

    X = np.concatenate(parts, axis=1).astype(np.float32)

    # Sanity check
    assert np.all(np.isfinite(X)), "NaN/Inf detectados nas features extraídas!"
    return X


# -- Extração principal -----------------------------------------------------------

def extrair_salvar_features(db_path, prefixo, modelo, hook):
    """Carrega dados, extrai features multi-nível, salva numpy."""
    log.info(f"[{prefixo}] Carregando banco: {db_path}")

    if not os.path.exists(db_path):
        log.error(f"[{prefixo}] Banco não encontrado: {db_path}")
        return

    registros, labels, meses = carregar_dados(db_path)
    if not registros:
        log.error(f"[{prefixo}] Nenhum dado encontrado.")
        return

    log.info(f"[{prefixo}] {len(registros)} registros carregados")

    dataset = TemporalCulturaDataset(registros, labels, meses)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS,
                         pin_memory=(DEVICE.type == 'cuda'))

    modelo.eval()
    X_batches = []
    y_batches = []
    total = 0
    t0 = time.time()

    with autocast(device_type=DEVICE.type, enabled=USE_AMP):
        for imagens, dias, mes, mask, batch_labels in loader:
            B = imagens.shape[0]

            # Mover inputs para device
            imagens = imagens.to(DEVICE, non_blocking=True)
            dias    = dias.to(DEVICE, non_blocking=True)
            mes     = mes.to(DEVICE, non_blocking=True)
            mask    = mask.to(DEVICE, non_blocking=True)

            # Extrair todos os níveis de features
            features = extract_multilevel(modelo, imagens, dias, mes, mask, hook)

            # Preparar metadados
            dias_np = dias.cpu().numpy()                                  # (B, T)
            mes_np  = (mes.cpu().numpy() + 1).reshape(B, 1)              # (B, 1)
            count   = mask.sum(dim=1).cpu().numpy().reshape(B, 1)        # (B, 1)

            # Montar linha tabular
            batch_X = assemble_tabular(features, dias_np, mes_np, count)

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

    # Salvar
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f'X_{prefixo}.npy'), X)
    np.save(os.path.join(OUT_DIR, f'y_{prefixo}.npy'), y)
    log.info(f"[{prefixo}] Salvo em {OUT_DIR}/")

    # Estatísticas rápidas
    log.info(f"[{prefixo}] Feature stats — mean: {X.mean():.4f}, "
             f"std: {X.std():.4f}, min: {X.min():.4f}, max: {X.max():.4f}")

    # Distribuição de classes
    unique, counts = np.unique(y, return_counts=True)
    for cid, cnt in zip(unique, counts):
        name = CLASSES[cid] if cid < len(CLASSES) else f"cls_{cid}"
        log.info(f"[{prefixo}]   {name}: {cnt} ({100*cnt/len(y):.1f}%)")


def main():
    log.info("="*60)
    log.info("Extrator Multi-Nível de Features (V7 → XGBoost)")
    log.info("="*60)

    # 1. Carregar modelo com pesos treinados
    log.info(f"Carregando modelo V7 no device: {DEVICE}")
    modelo = EfficientNetTemporalV6(len(CLASSES)).to(DEVICE)

    if os.path.exists(PESOS_PATH):
        try:
            modelo.load_state_dict(torch.load(PESOS_PATH, map_location=DEVICE))
            log.info(f"Pesos V7 carregados de: {PESOS_PATH}")
        except Exception as e:
            log.warning(f"Falha ao carregar pesos: {e}. Usando ImageNet base.")
    else:
        log.warning("Pesos V7 não encontrados. Usando ImageNet base.")

    # 2. Registrar hooks multi-escala
    hook = BackboneHook(modelo.backbone)
    log.info("Hooks multi-escala registrados nos blocos [2, 4, 6]")

    # 3. Extrair
    extrair_salvar_features(DB_TREINO, "treino", modelo, hook)
    extrair_salvar_features(DB_TESTE,  "teste",  modelo, hook)

    hook.remove()
    log.info("Extração multi-nível concluída. Pronto para XGBoost.")


if __name__ == '__main__':
    main()
