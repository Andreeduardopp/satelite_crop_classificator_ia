"""
Classificador de culturas V9 — EfficientNetB0 multispectral (6 bandas) +
FiLM(dia, mês) + Temporal Attention + cabeça hierárquica do V8.

Diferenças vs V8:
  - Entrada: TIFF 6 bandas FLOAT32 (B02, B03, B04, B08, B11, B12).
  - conv_stem do EfficientNetB0 adaptado de 3 → 6 canais (inflate trick:
    copia pesos RGB, inicializa NIR/SWIR com média dos pesos RGB).
  - Normalização por banda do dataset (não ImageNet).
  - MAX_SEQ_LEN = 5 (vs 3 do V7/V8) — aproveita os DAPs adicionais.
  - Dataset/DB do V9 (`dados_v9.db`, tabela `culturas_tiff`).

A cabeça hierárquica do V8 (group → cereal/grão) é mantida porque resolve a
confusão semântica entre culturas similares — independente de bandas. O ganho
esperado do V9 é que aveia × trigo (ambas gramíneas) fiquem mais separáveis
graças ao SWIR (B11/B12), aumentando especificamente a perna `head_cereal`.

Uso:
    python src/models/efficientnet_v9_multispectral/train.py
"""

import json
import os
import time
import logging
from datetime import datetime
from collections import Counter

import numpy as np
import cv2
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

from dataset import (
    TemporalCulturaTiffDataset,
    carregar_dados,
    CLASSES,
    NUM_BANDS,
    MAX_SEQ_LEN,
    DB_NOME,
)


# ---------------------------------------------------------------------------
# Paths e logging
# ---------------------------------------------------------------------------

_HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.normpath(os.path.join(_HERE, '..', '..'))
ROOT_DIR  = os.path.normpath(os.path.join(SRC_DIR, '..'))

LOG_DIR = os.path.join(_HERE, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(
    LOG_DIR, f'treino_v9_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)


# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------

DB_PATH         = os.path.join(SRC_DIR, DB_NOME)
BATCH_SIZE      = 32             # ↓ vs V7 (64) por causa de 6 canais + MAX_SEQ_LEN=5
EPOCHS_FASE1    = 10
EPOCHS_FASE2    = 15
LR_FASE1        = 1e-3
LR_FASE2        = 1e-5
GRAD_CLIP       = 1.0
LABEL_SMOOTHING = 0.1
MODELO_SAIDA    = os.path.join(_HERE, 'artifacts')

SEED             = 42
MES_EMBED_DIM    = 8
FINE_TUNE_LAYERS = 20
NUM_WORKERS      = 4

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Cabeça hierárquica do V8
# ---------------------------------------------------------------------------

# Grupo 0 = "cereal de inverno" (aveia, trigo); grupo 1 = "grão de verão"
# (feijão, milho, soja). Mapping consistente com CLASSES = [soja, milho, trigo,
# aveia, feijão].
GRUPO_DA_CLASSE = {
    0: 1,  # soja   -> grão
    1: 1,  # milho  -> grão
    2: 0,  # trigo  -> cereal
    3: 0,  # aveia  -> cereal
    4: 1,  # feijão -> grão
}
CLASSES_GRAO    = [0, 1, 4]   # soja, milho, feijão
CLASSES_CEREAL  = [2, 3]      # trigo, aveia
IDX_NO_GRUPO    = {c: i for i, c in enumerate(CLASSES_GRAO)}
IDX_NO_GRUPO.update({c: i for i, c in enumerate(CLASSES_CEREAL)})


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def adaptar_conv_stem_para_n_bandas(backbone, n_bandas: int):
    """
    Substitui o conv_stem da EfficientNet por um Conv2d que aceita `n_bandas`
    canais. Inicializa:
      - canais 0..2 com os pesos RGB pretrained (intactos);
      - canais 3..N-1 com a média dos pesos RGB (prior razoável: assume que
        bandas extras se comportam, em primeira aproximação, como uma
        combinação de RGB).
    """
    old = backbone.conv_stem
    new = nn.Conv2d(
        in_channels=n_bandas,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        new.weight.zero_()
        # Copia os 3 canais RGB
        new.weight[:, :3] = old.weight
        if n_bandas > 3:
            media_rgb = old.weight.mean(dim=1, keepdim=True)
            for c in range(3, n_bandas):
                new.weight[:, c:c+1] = media_rgb
        if old.bias is not None:
            new.bias.copy_(old.bias)
    backbone.conv_stem = new
    return backbone


class EfficientNetTemporalV9(nn.Module):
    """
    EfficientNetB0 (6-canais) + FiLM(dia, mes_emb) + 2x MultiHeadAttention +
    cabeça hierárquica (grupo, cereal, grão).
    """

    def __init__(self, n_bandas: int = NUM_BANDS):
        super().__init__()

        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.backbone = adaptar_conv_stem_para_n_bandas(self.backbone, n_bandas)
        self.feature_dim = self.backbone.num_features  # 1280

        # Congela tudo no início; descongelamos as últimas N camadas na Fase 2.
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Mas o conv_stem novo (3 canais extras inicializados com média) precisa
        # treinar desde a Fase 1 para se ajustar às bandas reais.
        for p in self.backbone.conv_stem.parameters():
            p.requires_grad = True

        self.mes_embedding = nn.Embedding(12, MES_EMBED_DIM)

        self.film_hidden = nn.Linear(1 + MES_EMBED_DIM, 64)
        self.film_gamma  = nn.Linear(64, self.feature_dim)
        self.film_beta   = nn.Linear(64, self.feature_dim)
        nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight);  nn.init.zeros_(self.film_beta.bias)

        self.attn1 = nn.MultiheadAttention(self.feature_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.attn2 = nn.MultiheadAttention(self.feature_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(self.feature_dim)

        # Tronco compartilhado antes das 3 cabeças
        self.tronco = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head_grupo  = nn.Linear(256, 2)
        self.head_cereal = nn.Linear(256, len(CLASSES_CEREAL))
        self.head_grao   = nn.Linear(256, len(CLASSES_GRAO))

    def forward(self, images, dias, mes, mask):
        B, T = images.shape[0], images.shape[1]

        imgs_flat = images.reshape(B * T, *images.shape[2:])
        feats_flat = self.backbone(imgs_flat)
        features = feats_flat.reshape(B, T, -1)

        mes_emb = self.mes_embedding(mes).unsqueeze(1).expand(-1, T, -1)
        dia_exp = dias.unsqueeze(-1)
        ctx = torch.cat([dia_exp, mes_emb], dim=-1)
        film_h = F.relu(self.film_hidden(ctx))
        gamma  = self.film_gamma(film_h)
        beta   = self.film_beta(film_h)
        tokens = features * (1.0 + gamma) + beta

        kpm = (mask == 0)
        a1, _ = self.attn1(tokens, tokens, tokens, key_padding_mask=kpm)
        tokens = self.norm1(tokens + a1)
        a2, _ = self.attn2(tokens, tokens, tokens, key_padding_mask=kpm)
        tokens = self.norm2(tokens + a2)

        m_exp  = mask.unsqueeze(-1)
        pooled = (tokens * m_exp).sum(dim=1) / m_exp.sum(dim=1).clamp(min=1.0)

        feat = self.tronco(pooled)
        logits_grupo  = self.head_grupo(feat)
        logits_cereal = self.head_cereal(feat)
        logits_grao   = self.head_grao(feat)
        return logits_grupo, logits_cereal, logits_grao

    def descongelar_ultimas_camadas(self, n: int):
        for p in self.backbone.parameters():
            p.requires_grad = False
        params = list(self.backbone.named_parameters())
        for _, p in params[-n:]:
            p.requires_grad = True
        # Mantém conv_stem treinável sempre
        for p in self.backbone.conv_stem.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Loss hierárquica
# ---------------------------------------------------------------------------

def loss_hierarquica(logits_g, logits_c, logits_gr, labels, smoothing=LABEL_SMOOTHING):
    """
    Loss = CE(grupo) + CE(subclasse condicional ao grupo verdadeiro).

    Para cada amostra, só calculamos CE em UMA das duas sub-cabeças (a que
    corresponde ao grupo verdadeiro). A outra sub-cabeça não recebe gradiente
    desta amostra, evitando "dragar" a representação para uma classe
    irrelevante.
    """
    grupos = torch.tensor(
        [GRUPO_DA_CLASSE[int(l.item())] for l in labels],
        device=labels.device, dtype=torch.long,
    )
    loss_g = F.cross_entropy(logits_g, grupos, label_smoothing=smoothing)

    # Máscaras de cada subgrupo
    is_cereal = grupos == 0
    is_grao   = grupos == 1

    losses = [loss_g]

    if is_cereal.any():
        sub_idx = torch.tensor(
            [IDX_NO_GRUPO[int(l.item())] for l in labels[is_cereal]],
            device=labels.device, dtype=torch.long,
        )
        losses.append(F.cross_entropy(
            logits_c[is_cereal], sub_idx, label_smoothing=smoothing
        ))

    if is_grao.any():
        sub_idx = torch.tensor(
            [IDX_NO_GRUPO[int(l.item())] for l in labels[is_grao]],
            device=labels.device, dtype=torch.long,
        )
        losses.append(F.cross_entropy(
            logits_gr[is_grao], sub_idx, label_smoothing=smoothing
        ))

    return sum(losses) / len(losses)


def predicoes_hierarquicas(logits_g, logits_c, logits_gr):
    """
    Combina as 3 cabeças via probabilidade conjunta para retornar a classe
    final em CLASSES (5-way).
    """
    p_g  = F.softmax(logits_g, dim=-1)        # (B, 2)   [P(cereal), P(grão)]
    p_c  = F.softmax(logits_c, dim=-1)        # (B, 2)   [P(trigo|c), P(aveia|c)]
    p_gr = F.softmax(logits_gr, dim=-1)       # (B, 3)   [P(soja|g), P(milho|g), P(feijão|g)]

    B = p_g.shape[0]
    full = torch.zeros(B, len(CLASSES), device=p_g.device)
    # cereal: trigo (idx 2 em CLASSES = idx 0 em CLASSES_CEREAL)
    full[:, 2] = p_g[:, 0] * p_c[:, 0]
    full[:, 3] = p_g[:, 0] * p_c[:, 1]
    # grão: soja, milho, feijão
    full[:, 0] = p_g[:, 1] * p_gr[:, 0]
    full[:, 1] = p_g[:, 1] * p_gr[:, 1]
    full[:, 4] = p_g[:, 1] * p_gr[:, 2]

    return full.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar_fase(modelo, loader_tr, loader_val, optimizer, epochs, fase, patience, scaler=None):
    best_val = float('inf')
    best_state = None
    sem_melhora = 0

    for ep in range(epochs):
        modelo.train()
        tot_loss = corretos = total = 0
        for images, dias, mes, mask, labels in loader_tr:
            images = images.to(DEVICE, non_blocking=True)
            dias   = dias.to(DEVICE, non_blocking=True)
            mes    = mes.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                lg, lc, lr_ = modelo(images, dias, mes, mask)
                loss = loss_hierarquica(lg, lc, lr_, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in modelo.parameters() if p.requires_grad), GRAD_CLIP,
                )
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in modelo.parameters() if p.requires_grad), GRAD_CLIP,
                )
                optimizer.step()

            preds = predicoes_hierarquicas(lg, lc, lr_)
            tot_loss += loss.item() * images.size(0)
            corretos += (preds == labels).sum().item()
            total    += images.size(0)

        train_loss, train_acc = tot_loss / total, corretos / total

        # Validação
        modelo.eval()
        v_loss = v_corr = v_tot = 0
        with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=USE_AMP):
            for images, dias, mes, mask, labels in loader_val:
                images = images.to(DEVICE, non_blocking=True)
                dias   = dias.to(DEVICE, non_blocking=True)
                mes    = mes.to(DEVICE, non_blocking=True)
                mask   = mask.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                lg, lc, lr_ = modelo(images, dias, mes, mask)
                loss = loss_hierarquica(lg, lc, lr_, labels)
                preds = predicoes_hierarquicas(lg, lc, lr_)
                v_loss += loss.item() * images.size(0)
                v_corr += (preds == labels).sum().item()
                v_tot  += images.size(0)
        val_loss, val_acc = v_loss / v_tot, v_corr / v_tot

        logging.info(
            f"[{fase}] ep {ep+1}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in modelo.state_dict().items()}
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= patience:
                logging.info(f"Early stop após {patience} epochs sem melhora.")
                break

    if best_state:
        modelo.load_state_dict(best_state)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.info("=== V9: EfficientNetB0 6-bandas + FiLM + Temporal Attention + cabeça hierárquica ===")
    logging.info(f"Dispositivo: {DEVICE} | AMP: {USE_AMP}")

    registros, labels, meses = carregar_dados(DB_PATH, src_dir=SRC_DIR)
    if not registros:
        logging.error("Nenhum talhão com TIFFs disponíveis. Rode pipeline_tiff.py primeiro.")
        return

    logging.info(f"Total: {len(registros)} talhões")
    for i, c in enumerate(CLASSES):
        logging.info(f"  {c}: {labels.count(i)}")

    reg_tr, reg_val, lab_tr, lab_val, mes_tr, mes_val = train_test_split(
        registros, labels, meses, test_size=0.2, stratify=labels, random_state=SEED,
    )

    ds_tr  = TemporalCulturaTiffDataset(reg_tr,  lab_tr,  mes_tr,  augment=True)
    ds_val = TemporalCulturaTiffDataset(reg_val, lab_val, mes_val, augment=False)
    use_pin = DEVICE.type == 'cuda'

    lab_tr_np = np.array(lab_tr)
    counts = np.bincount(lab_tr_np, minlength=len(CLASSES))
    sample_w = (1.0 / np.maximum(counts, 1))[lab_tr_np]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(lab_tr_np),
        replacement=True,
    )

    def _wi(_):
        cv2.setNumThreads(0)
        try: cv2.ocl.setUseOpenCL(False)
        except AttributeError: pass

    loader_tr = DataLoader(
        ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=use_pin,
        persistent_workers=NUM_WORKERS > 0, worker_init_fn=_wi,
    )
    loader_val = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_pin,
        persistent_workers=NUM_WORKERS > 0, worker_init_fn=_wi,
    )

    modelo = EfficientNetTemporalV9(n_bandas=NUM_BANDS).to(DEVICE)
    total_p = sum(p.numel() for p in modelo.parameters())
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Params: {total_p:,} total | {train_p:,} treináveis (Fase 1)")

    scaler = GradScaler(device=DEVICE.type) if USE_AMP else None

    # Fase 1: treina cabeça + FiLM + atenção + conv_stem (resto do backbone congelado)
    logging.info("=== Fase 1: cabeça + FiLM + attn + conv_stem ===")
    t0 = time.perf_counter()
    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE1,
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, EPOCHS_FASE1, "Fase1", patience=3, scaler=scaler)

    # Fase 2: descongela últimas N camadas
    logging.info(f"=== Fase 2: fine-tuning (últimas {FINE_TUNE_LAYERS} camadas) ===")
    modelo.descongelar_ultimas_camadas(FINE_TUNE_LAYERS)
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Treináveis (Fase 2): {train_p:,}")

    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE2,
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, EPOCHS_FASE2, "Fase2", patience=4, scaler=scaler)

    tempo_total = time.perf_counter() - t0
    logging.info(f"Tempo total de treino: {tempo_total:.1f}s")

    # Salvar
    os.makedirs(MODELO_SAIDA, exist_ok=True)
    peso_path = os.path.join(MODELO_SAIDA, 'pesos.pt')
    torch.save(modelo.state_dict(), peso_path)
    logging.info(f"Pesos: {peso_path}")

    # Avaliação final
    modelo.eval()
    y_true, y_pred = [], []
    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=USE_AMP):
        for images, dias, mes, mask, labels in loader_val:
            images = images.to(DEVICE, non_blocking=True)
            dias   = dias.to(DEVICE, non_blocking=True)
            mes    = mes.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)
            lg, lc, lr_ = modelo(images, dias, mes, mask)
            preds = predicoes_hierarquicas(lg, lc, lr_)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0)

    logging.info(f"Validação final - Acc: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    for c, f in zip(CLASSES, f1_per):
        logging.info(f"  F1 {c}: {f:.4f}")
    logging.info("Classification Report:\n" + classification_report(
        y_true, y_pred, target_names=CLASSES, zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    logging.info("Confusion Matrix (linhas=real, colunas=predito):")
    logging.info("           " + " ".join(f"{c:>8}" for c in CLASSES))
    for c, row in zip(CLASSES, cm):
        logging.info(f"  {c:>8}: " + " ".join(f"{v:>8}" for v in row))

    # Métricas em JSON
    metrics_path = os.path.join(
        MODELO_SAIDA, f"metrics_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'f1_macro': float(f1_macro),
            'accuracy': float(acc),
            'f1_per_class': {c: float(v) for c, v in zip(CLASSES, f1_per)},
            'tempo_treino_s': float(tempo_total),
        }, f, indent=2)
    logging.info(f"Métricas: {metrics_path}")


if __name__ == '__main__':
    main()
