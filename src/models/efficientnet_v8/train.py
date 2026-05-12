"""
Classificador de culturas v8 - EfficientNetB0 + FiLM(dia, mes) + Temporal Attention
                              + HEAD HIERARQUICA.

Diferenca para v7:
    A confusao na avaliacao de FocusNet/v7 mostrou dois clusters bem definidos:
        - Cereais de inverno: {aveia, trigo}    -> separaveis com facilidade
        - Graos de verao   : {feijao, milho, soja} -> confusos entre si
    Em vez de uma unica cabeca de 5 classes, v8 usa tres cabecas:
        head_group  : 2-way   {cereal, grao}                (decisao "facil")
        head_cereal : 2-way   {aveia, trigo}                (so faz sentido p/ cereais)
        head_grao   : 3-way   {feijao, milho, soja}         (decisao "dificil")
    Loss = loss(group) + loss(cereal | sample e cereal) + loss(grao | sample e grao).
    Inferencia: combina via probabilidade conjunta:
        P(classe) = P(grupo da classe) * P(classe | grupo)

Otimizacoes GPU:
    - AMP (mixed precision FP16)
    - cudnn.benchmark
    - non_blocking transfers
    - persistent DataLoader workers
    - WeightedRandomSampler + augmentation geometrica

Uso:
    python src/models/efficientnet_v8/train.py
"""

import json
import os
import re
import ast
import time
import logging
import sqlite3
from datetime import datetime
from collections import Counter

import numpy as np
import cv2
# Evita segfault em DataLoader workers: cv2 tem threadpool interno nao-fork-safe
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, accuracy_score,
)

# -- Paths relativos ao arquivo ------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(_HERE, '..', '..'))  # -> src/
ROOT_DIR = os.path.normpath(os.path.join(SRC_DIR, '..'))      # -> root/

# -- Logging -------------------------------------------------------------------
LOG_DIR = os.path.join(_HERE, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(
    LOG_DIR, f'treino_efficientnet_v8_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)

# -- Configuracoes -------------------------------------------------------------
DB_PATH          = os.path.join(SRC_DIR, 'dados_v2.db')
TABELA           = 'culturas'
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 64
EPOCHS_FASE1     = 10
EPOCHS_FASE2     = 15
LR_FASE1         = 5e-4
LR_FASE2         = 1e-5
GRAD_CLIP        = 1.0
LABEL_SMOOTHING  = 0.1
MODELO_SAIDA     = os.path.join(_HERE, 'artifacts')

# Ordem das 5 classes (mantida igual a v7 para compatibilidade de checkpoints/logs).
CLASSES = ['soja', 'milho', 'trigo', 'aveia', 'feijão']

# Hierarquia
# Grupo: 0=cereal, 1=grao
# Mapeamentos por idx em CLASSES:
#   soja(0)=grao, milho(1)=grao, trigo(2)=cereal, aveia(3)=cereal, feijao(4)=grao
GROUP_OF_CLASS    = [1, 1, 0, 0, 1]
# Idx dentro do head_cereal (aveia=0, trigo=1). -1 = nao se aplica.
CEREAL_IDX_OF_CLS = [-1, -1, 1, 0, -1]
# Idx dentro do head_grao (feijao=0, milho=1, soja=2). -1 = nao se aplica.
GRAO_IDX_OF_CLS   = [2, 1, -1, -1, 0]
NUM_CEREAL = 2
NUM_GRAO   = 3

SEED             = 42
MAX_SEQ_LEN      = 3
MAX_DIA          = 100.0
MES_EMBED_DIM    = 8
FINE_TUNE_LAYERS = 20
NUM_WORKERS      = 4

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'

# BF16 (Ampere+) tem o mesmo range de FP32 -> elimina overflow no Swish do EfficientNet.
# Em GPUs anteriores, cai de volta para FP16 + GradScaler.
if USE_AMP and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
    USE_FP16  = False
else:
    AMP_DTYPE = torch.float16
    USE_FP16  = USE_AMP

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# -- Dados ---------------------------------------------------------------------

def extrair_dia(caminho: str) -> int:
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_dados(db_path: str) -> tuple[list[list[tuple[str, int]]], list[int], list[int]]:
    classe_para_id = {c: i for i, c in enumerate(CLASSES)}
    classe_para_id['feijao'] = classe_para_id['feijão']

    registros = []
    labels = []
    meses = []

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


# -- Dataset -------------------------------------------------------------------

class TemporalCulturaDataset(Dataset):
    def __init__(self, registros, labels, meses, augment: bool = False):
        self.registros = registros
        self.labels = labels
        self.meses = meses
        self.augment = augment

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

        if self.augment:
            flip_h = np.random.rand() < 0.5
            flip_v = np.random.rand() < 0.5
            rot_k = np.random.randint(0, 4)
        else:
            flip_h = flip_v = False
            rot_k = 0

        for i in range(seq_len):
            caminho, dia = items[i]
            img = preprocessar_imagem(caminho)
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


# -- Modelo --------------------------------------------------------------------

class EfficientNetTemporalV8(nn.Module):
    """
    EfficientNetB0 + FiLM(dia, mes_embedding) + 2x MultiHeadAttention temporal
    + cabecas hierarquicas:
        head_group  (2 logits): cereal vs grao
        head_cereal (2 logits): aveia vs trigo
        head_grao   (3 logits): feijao vs milho vs soja
    """

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 1280

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.mes_embedding = nn.Embedding(12, MES_EMBED_DIM)

        self.film_hidden = nn.Linear(1 + MES_EMBED_DIM, 64)
        self.film_gamma = nn.Linear(64, self.feature_dim)
        self.film_beta = nn.Linear(64, self.feature_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.attn1 = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=8, dropout=0.1, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=8, dropout=0.1, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.feature_dim)

        # Cabecas hierarquicas (3). Cada uma com seu MLP independente para nao
        # forcar compartilhar capacidade entre tarefas com fronteiras distintas.
        def _mlp(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, out_dim),
            )

        self.head_group  = _mlp(2)
        self.head_cereal = _mlp(NUM_CEREAL)
        self.head_grao   = _mlp(NUM_GRAO)

    def encode(self, images, dias, mes, mask):
        """Pipeline backbone+FiLM+attention -> features pooled (B, feature_dim)."""
        B, T = images.shape[0], images.shape[1]

        imgs_flat = images.reshape(B * T, *images.shape[2:])
        feats_flat = self.backbone(imgs_flat)
        features = feats_flat.reshape(B, T, -1)

        mes_emb = self.mes_embedding(mes)
        mes_emb = mes_emb.unsqueeze(1).expand(-1, T, -1)
        dia_exp = dias.unsqueeze(-1)
        context = torch.cat([dia_exp, mes_emb], dim=-1)

        film_h = F.relu(self.film_hidden(context))
        gamma = self.film_gamma(film_h)
        beta = self.film_beta(film_h)
        tokens = features * (1.0 + gamma) + beta

        key_pad_mask = (mask == 0)
        attn_out1, _ = self.attn1(tokens, tokens, tokens, key_padding_mask=key_pad_mask)
        tokens = self.norm1(tokens + attn_out1)
        attn_out2, _ = self.attn2(tokens, tokens, tokens, key_padding_mask=key_pad_mask)
        tokens = self.norm2(tokens + attn_out2)

        mask_exp = mask.unsqueeze(-1)
        pooled = (tokens * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(self, images, dias, mes, mask):
        pooled = self.encode(images, dias, mes, mask)
        return (
            self.head_group(pooled),    # (B, 2)
            self.head_cereal(pooled),   # (B, 2)
            self.head_grao(pooled),     # (B, 3)
        )

    def descongelar_ultimas_camadas(self, n_camadas: int):
        for param in self.backbone.parameters():
            param.requires_grad = False
        all_layers = list(self.backbone.named_parameters())
        for _, param in all_layers[-n_camadas:]:
            param.requires_grad = True


# -- Hierarquia: utilitarios ---------------------------------------------------

def _make_hierarchy_tensors(device):
    group_of    = torch.tensor(GROUP_OF_CLASS,    dtype=torch.long, device=device)
    cereal_idx  = torch.tensor(CEREAL_IDX_OF_CLS, dtype=torch.long, device=device)
    grao_idx    = torch.tensor(GRAO_IDX_OF_CLS,   dtype=torch.long, device=device)
    return group_of, cereal_idx, grao_idx


def combinar_5way_logprobs(group_l, cereal_l, grao_l):
    """
    Combina os 3 heads em log-probabilidades 5-way usando probabilidade conjunta:
        P(c) = P(grupo de c) * P(c | grupo)
    Ordem das 5 saidas: ['soja','milho','trigo','aveia','feijão'].
    """
    log_g = F.log_softmax(group_l,  dim=-1)  # [:,0]=cereal, [:,1]=grao
    log_c = F.log_softmax(cereal_l, dim=-1)  # [:,0]=aveia,  [:,1]=trigo
    log_r = F.log_softmax(grao_l,   dim=-1)  # [:,0]=feijão, [:,1]=milho, [:,2]=soja

    out = torch.empty(group_l.shape[0], 5, device=group_l.device, dtype=log_g.dtype)
    out[:, 0] = log_g[:, 1] + log_r[:, 2]   # soja
    out[:, 1] = log_g[:, 1] + log_r[:, 1]   # milho
    out[:, 2] = log_g[:, 0] + log_c[:, 1]   # trigo
    out[:, 3] = log_g[:, 0] + log_c[:, 0]   # aveia
    out[:, 4] = log_g[:, 1] + log_r[:, 0]   # feijão
    return out


def calcular_loss_hierarquica(group_l, cereal_l, grao_l, labels, criterion,
                              group_of, cereal_idx, grao_idx):
    """
    loss = loss_group + loss_cereal (so amostras cereais) + loss_grao (so amostras graos).
    """
    group_lab = group_of[labels]   # (B,)
    loss_g = criterion(group_l, group_lab).mean()

    is_cereal = group_lab == 0
    is_grao   = group_lab == 1

    if is_cereal.any():
        cereal_lab = cereal_idx[labels[is_cereal]]
        loss_c = criterion(cereal_l[is_cereal], cereal_lab).mean()
    else:
        loss_c = torch.zeros((), device=group_l.device)

    if is_grao.any():
        grao_lab = grao_idx[labels[is_grao]]
        loss_r = criterion(grao_l[is_grao], grao_lab).mean()
    else:
        loss_r = torch.zeros((), device=group_l.device)

    return loss_g + loss_c + loss_r, (loss_g.detach(), loss_c.detach(), loss_r.detach())


# -- Treino --------------------------------------------------------------------

def treinar_fase(
    modelo, loader_treino, loader_val, optimizer, criterion,
    epochs, fase_nome, patience,
    group_of, cereal_idx, grao_idx,
    scaler=None,
):
    best_val_loss = float('inf')
    best_state = None
    epochs_sem_melhora = 0

    for epoch in range(epochs):
        modelo.train()
        total_loss = corretos = total = 0
        sum_lg = sum_lc = sum_lr = 0.0
        nan_streak = 0
        nan_total = 0

        for images, dias, mes, mask, labels in loader_treino:
            images = images.to(DEVICE, non_blocking=True)
            dias   = dias.to(DEVICE, non_blocking=True)
            mes    = mes.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                group_l, cereal_l, grao_l = modelo(images, dias, mes, mask)
                loss, (lg, lc, lr) = calcular_loss_hierarquica(
                    group_l, cereal_l, grao_l, labels, criterion,
                    group_of, cereal_idx, grao_idx,
                )

            # Guarda contra NaN/Inf (overflow FP16 no Swish do EfficientNet):
            # se o loss nao for finito, NAO chamamos backward — os pesos ficam intactos.
            if not torch.isfinite(loss):
                nan_streak += 1
                nan_total += 1
                logging.warning(
                    f"[{fase_nome}] Loss nao-finito (g={lg.item()} c={lc.item()} r={lr.item()}) "
                    f"- batch ignorado ({nan_streak} consecutivos, {nan_total} no epoch)"
                )
                if nan_streak >= 20:
                    logging.error(
                        f"[{fase_nome}] 20 batches NaN consecutivos — possivel corrupcao "
                        "dos pesos. Abortando epoch."
                    )
                    break
                continue
            nan_streak = 0

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in modelo.parameters() if p.requires_grad),
                    max_norm=GRAD_CLIP,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in modelo.parameters() if p.requires_grad),
                    max_norm=GRAD_CLIP,
                )
                optimizer.step()

            with torch.no_grad():
                logits5 = combinar_5way_logprobs(group_l, cereal_l, grao_l)
                corretos += (logits5.argmax(1) == labels).sum().item()

            bs = images.size(0)
            total_loss += loss.item() * bs
            sum_lg += lg.item() * bs
            sum_lc += lc.item() * bs
            sum_lr += lr.item() * bs
            total += bs

        train_loss = total_loss / total
        train_acc = corretos / total

        # Validacao
        modelo.eval()
        vl_loss = vl_corr = vl_tot = 0
        with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            for images, dias, mes, mask, labels in loader_val:
                images = images.to(DEVICE, non_blocking=True)
                dias   = dias.to(DEVICE, non_blocking=True)
                mes    = mes.to(DEVICE, non_blocking=True)
                mask   = mask.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                group_l, cereal_l, grao_l = modelo(images, dias, mes, mask)
                loss, _ = calcular_loss_hierarquica(
                    group_l, cereal_l, grao_l, labels, criterion,
                    group_of, cereal_idx, grao_idx,
                )
                logits5 = combinar_5way_logprobs(group_l, cereal_l, grao_l)
                vl_loss += loss.item() * images.size(0)
                vl_corr += (logits5.argmax(1) == labels).sum().item()
                vl_tot += images.size(0)

        val_loss = vl_loss / vl_tot
        val_acc = vl_corr / vl_tot

        logging.info(
            f"[{fase_nome}] Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_loss:.4f} (g={sum_lg/total:.3f} c={sum_lc/total:.3f} r={sum_lr/total:.3f}) | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in modelo.state_dict().items()}
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= patience:
                logging.info(f"Early stopping apos {patience} epochs sem melhora.")
                break

    if best_state:
        modelo.load_state_dict(best_state)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=DB_PATH)
    parser.add_argument('--out_dir', type=str, default=MODELO_SAIDA)
    args = parser.parse_args()

    db_path_atual = args.db_path
    modelo_saida_atual = args.out_dir

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.info(
        "=== V8: EfficientNetB0 + FiLM(dia, mes_embedding) + Temporal Attention + HEAD HIERARQUICA ==="
    )
    logging.info(
        f"Dispositivo: {DEVICE} | AMP: {USE_AMP} | dtype: {AMP_DTYPE} | "
        f"GradScaler: {USE_FP16} | cuDNN benchmark: {torch.backends.cudnn.benchmark}"
    )

    # 1. Carregar dados
    logging.info(f"Carregando dados de: {db_path_atual}")
    registros, labels, meses = carregar_dados(db_path_atual)
    total_imgs = sum(len(r) for r in registros)
    logging.info(f"Total: {len(registros)} talhoes, {total_imgs} imagens")

    if not registros:
        logging.error("Nenhum talhao com imagens disponiveis.")
        return

    seq_lens = Counter(min(len(r), MAX_SEQ_LEN) for r in registros)
    for n, count in sorted(seq_lens.items()):
        logging.info(f"  Talhoes com {n} imagem(ns): {count}")
    for i, c in enumerate(CLASSES):
        logging.info(f"  {c}: {labels.count(i)} talhoes")

    mes_dist = Counter(meses)
    logging.info("Distribuicao de meses:")
    for m in sorted(mes_dist):
        logging.info(f"  mes {m:>2}: {mes_dist[m]} talhoes")

    # 2. Split
    reg_treino, reg_val, lab_treino, lab_val, mes_treino, mes_val = train_test_split(
        registros, labels, meses, test_size=0.2, stratify=labels, random_state=SEED
    )
    logging.info(f"Treino: {len(reg_treino)} talhoes | Validacao: {len(reg_val)} talhoes")

    # 3. DataLoaders + WeightedRandomSampler
    ds_tr  = TemporalCulturaDataset(reg_treino, lab_treino, mes_treino, augment=True)
    ds_val = TemporalCulturaDataset(reg_val, lab_val, mes_val, augment=False)
    use_pin = DEVICE.type == 'cuda'

    lab_tr_np = np.array(lab_treino)
    class_counts = np.bincount(lab_tr_np, minlength=len(CLASSES))
    sample_w = (1.0 / np.maximum(class_counts, 1))[lab_tr_np]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(lab_tr_np),
        replacement=True,
    )
    logging.info(f"Counts treino por classe: {dict(zip(CLASSES, class_counts.tolist()))}")
    logging.info("Usando WeightedRandomSampler (batches balanceados por classe) + augmentation geometrica")

    def _worker_init(_):
        cv2.setNumThreads(0)
        try:
            cv2.ocl.setUseOpenCL(False)
        except AttributeError:
            pass

    loader_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, sampler=sampler,
                            num_workers=NUM_WORKERS, pin_memory=use_pin,
                            persistent_workers=NUM_WORKERS > 0,
                            worker_init_fn=_worker_init)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=use_pin,
                            persistent_workers=NUM_WORKERS > 0,
                            worker_init_fn=_worker_init)
    logging.info(f"Treino batches: {len(loader_tr)} | Val batches: {len(loader_val)}")

    # 4. Hierarquia
    group_of, cereal_idx, grao_idx = _make_hierarchy_tensors(DEVICE)

    # 5. Modelo
    logging.info("Criando EfficientNetB0 V8 + FiLM(dia,mes) + Temporal Attention + Head Hierarquica...")
    modelo = EfficientNetTemporalV8().to(DEVICE)

    total_p = sum(p.numel() for p in modelo.parameters())
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Params: {total_p:,} total | {train_p:,} treinaveis")

    scaler = GradScaler(device=DEVICE.type) if USE_FP16 else None
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=LABEL_SMOOTHING)

    # -- Fase 1
    logging.info("=== Fase 1: Treinando heads + FiLM + temporal attention (base congelada) ===")
    train_start_time = time.perf_counter()
    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE1,
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, criterion,
                 EPOCHS_FASE1, "Fase1", patience=3,
                 group_of=group_of, cereal_idx=cereal_idx, grao_idx=grao_idx,
                 scaler=scaler)

    # -- Fase 2
    logging.info(f"=== Fase 2: Fine-tuning (ultimas {FINE_TUNE_LAYERS} camadas) ===")
    modelo.descongelar_ultimas_camadas(FINE_TUNE_LAYERS)
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Parametros treinaveis: {train_p:,}")

    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE2,
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, criterion,
                 EPOCHS_FASE2, "Fase2", patience=4,
                 group_of=group_of, cereal_idx=cereal_idx, grao_idx=grao_idx,
                 scaler=scaler)

    train_total_time = time.perf_counter() - train_start_time
    logging.info(f"Tempo total de treinamento: {train_total_time:.2f} segundos")

    # -- Salvar
    os.makedirs(modelo_saida_atual, exist_ok=True)
    peso_path = os.path.join(modelo_saida_atual, 'pesos.pt')
    torch.save(modelo.state_dict(), peso_path)
    logging.info(f"Pesos salvos em: {peso_path}")

    # -- Avaliacao final
    modelo.eval()
    y_true, y_pred = [], []
    y_pred_group_only = []   # rotulo previsto so pelo head_group, p/ medir o gargalo
    n_samples = 0
    t_start = time.perf_counter()

    with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
        for images, dias, mes, mask, labels in loader_val:
            images = images.to(DEVICE, non_blocking=True)
            dias   = dias.to(DEVICE, non_blocking=True)
            mes    = mes.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)

            group_l, cereal_l, grao_l = modelo(images, dias, mes, mask)
            logits5 = combinar_5way_logprobs(group_l, cereal_l, grao_l)
            y_pred.extend(logits5.argmax(1).cpu().numpy())
            y_pred_group_only.extend(group_l.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
            n_samples += labels.size(0)

    t_total = time.perf_counter() - t_start

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Acuracia do head de grupo (cereal vs grao) isolado
    y_true_group = [GROUP_OF_CLASS[t] for t in y_true]
    acc_group = accuracy_score(y_true_group, y_pred_group_only)

    logging.info(f"Validacao final - Acc 5-way: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    logging.info(f"Acuracia do head_group (cereal vs grao): {acc_group:.4f}")
    for cls_name, f1_cls in zip(CLASSES, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_true, y_pred, target_names=CLASSES, zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    logging.info("Confusion Matrix (linhas=real, colunas=predito):")
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    logging.info(header)
    for cls, row in zip(CLASSES, cm):
        logging.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    tempo_medio = (t_total / n_samples) * 1000
    logging.info(f"Tempo medio de inferencia: {tempo_medio:.2f} ms/talhao ({n_samples} talhoes)")

    # -- Salvar metricas
    try:
        os.makedirs(modelo_saida_atual, exist_ok=True)
        metrics_path = os.path.join(
            modelo_saida_atual, f"metrics_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        metrics_dict = {
            "f1_macro": float(f1_macro),
            "accuracy": float(acc),
            "accuracy_group_head": float(acc_group),
            "f1_per_class": {c: float(f) for c, f in zip(CLASSES, f1_per_class)},
            "tempo_medio_ms": float(tempo_medio),
            "tempo_treino_segundos": float(train_total_time),
        }
        with open(metrics_path, "w", encoding="utf-8") as m_f:
            json.dump(metrics_dict, m_f, indent=4)
        logging.info(f"Metricas salvas em {metrics_path}")
    except Exception as e:
        logging.warning(f"Erro ao salvar metricas: {e}")


if __name__ == '__main__':
    main()
