"""
Classificador de culturas v5 — MobileNetV3-Small → embeddings → XGBoost.

Pipeline em 2 estágios:
    Estágio 1 — Treina MobileNetV3-Small (2.5M params) como classificador
                de imagens individuais (por-imagem, não por-talhão).
                Duas fases: backbone congelado → fine-tuning últimos blocos.

    Estágio 2 — Extrai embedding 576-dim de cada imagem com o backbone treinado,
                agrega por talhão (mean/min/max/last) + metadados
                (mês, dia, n_timesteps) e treina XGBoost.

Motivação:
    O melhor resultado anterior usava 35 CNNs separadas → 35 sigmoids por
    imagem → XGBoost. Este pipeline substitui as 35 CNNs por 1 modelo leve,
    gerando embeddings mais ricos (576-dim vs 35-dim) com 1 forward pass
    por imagem em vez de 35.

Features para XGBoost (por talhão):
    mean(576) + min(576) + max(576) + last(576) + [mes, n_tps, last_dia]
    = 2307 features totais

Uso:
    python src/treinamento/treinar_classificador_v5.py
"""

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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, accuracy_score,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(
    LOG_DIR, f'treino_v5_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)

# ── Configurações ─────────────────────────────────────────────────────────────
DB_PATH          = './sample_treino_6k.db'
TABELA           = 'culturas'
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32           # conservador para CPU — aumentar se tiver GPU
BATCH_SIZE_EMBED = 64           # batch p/ extração de embeddings (sem gradientes)
EPOCHS_FASE1     = 15
EPOCHS_FASE2     = 20
LR_FASE1         = 1e-4          # 1e-3 causava overfitting imediato
LR_FASE2         = 5e-5          # 1e-5 mal movia o backbone
GRAD_CLIP        = 1.0
LABEL_SMOOTHING  = 0.05
MODELO_SAIDA     = './modelos/classificador_cultura_v5'
CLASSES          = ['milho', 'soja', 'trigo']
SEED             = 42
MAX_DIA          = 100.0
MAX_SEQ_LEN      = 3
BACKBONE_NAME    = 'mobilenetv3_small_100'
FINE_TUNE_BLOCOS = 3             # 2 era pouco — backbone precisa adaptar ao domínio

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# ── Dados ─────────────────────────────────────────────────────────────────────

def extrair_dia(caminho: str) -> int:
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_dados(
    db_path: str,
) -> tuple[list[list[tuple[str, int]]], list[int], list[int | None]]:
    """
    Retorna (registros, labels, meses).
      - registros: lista de listas de (caminho_png, dia) por talhão
      - labels: classe por talhão
      - meses: mês de plantio por talhão (None se coluna não existir)
    """
    classe_para_id = {c: i for i, c in enumerate(CLASSES)}
    registros: list[list[tuple[str, int]]] = []
    labels: list[int] = []
    meses: list[int | None] = []

    with sqlite3.connect(db_path) as conn:
        colunas = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({TABELA})").fetchall()
        }
    tem_mes = 'mes' in colunas

    select = f"SELECT cultura, imagens_processadas{', mes' if tem_mes else ''} FROM {TABELA}"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(select).fetchall()

    descartados = 0
    for row in rows:
        cultura, imgs_str = row[0], row[1]
        mes = row[2] if tem_mes else None

        if cultura not in classe_para_id:
            continue
        try:
            paths = ast.literal_eval(imgs_str)
        except (ValueError, SyntaxError):
            continue

        validos: list[tuple[str, int]] = []
        for p in paths:
            if os.path.exists(p):
                validos.append((p, extrair_dia(p)))

        if not validos:
            descartados += 1
            continue

        validos.sort(key=lambda x: x[1])
        registros.append(validos)
        labels.append(classe_para_id[cultura])
        meses.append(mes)

    logging.info(f"Registros carregados: {len(registros)} | descartados: {descartados}")
    return registros, labels, meses


def preprocessar_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))  # HWC → CHW


# ── Dataset Stage 1: imagens individuais ─────────────────────────────────────

class ImagemIndividualDataset(Dataset):
    """
    Cada sample é UMA imagem com o label do talhão.
    Flat — sem agrupamento temporal.
    """

    def __init__(self, registros: list[list[tuple[str, int]]], labels: list[int]):
        self.samples: list[tuple[str, int]] = []  # (path, label)
        for items, label in zip(registros, labels):
            for caminho, _dia in items[:MAX_SEQ_LEN]:
                self.samples.append((caminho, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        caminho, label = self.samples[idx]
        img = torch.tensor(preprocessar_imagem(caminho), dtype=torch.float32)
        return img, torch.tensor(label, dtype=torch.long)


# ── Modelo Stage 1 ──────────────────────────────────────────────────────────

class MobileNetV3Classificador(nn.Module):
    """
    MobileNetV3-Small: classificador por imagem (Stage 1)
    e extrator de embeddings (Stage 2).

    embed_dim = 1024 para mobilenetv3_small_100 (conv_head expande 576→1024).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME, pretrained=True, num_classes=0,
        )
        # num_features pode divergir da saída real (conv_head expande)
        with torch.no_grad():
            dummy = torch.randn(1, 3, *IMG_SIZE)
            self.embed_dim = self.backbone(dummy).shape[-1]

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def extrair_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def descongelar_ultimos_blocos(self, n_blocos: int):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.blocks[-n_blocos:]:
            for param in block.parameters():
                param.requires_grad = True
        # conv_head faz parte da representação final
        if hasattr(self.backbone, 'conv_head'):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True


# ── Stage 1: treino do backbone ─────────────────────────────────────────────

def treinar_fase(
    modelo, loader_treino, loader_val, optimizer, criterion,
    epochs, class_weight_tensor, fase_nome, patience,
):
    best_val_loss = float('inf')
    best_state = None
    epochs_sem_melhora = 0

    for epoch in range(epochs):
        modelo.train()
        total_loss = corretos = total = 0

        for images, lab in loader_treino:
            images, lab = images.to(DEVICE), lab.to(DEVICE)

            optimizer.zero_grad()
            logits = modelo(images)
            loss = (criterion(logits, lab) * class_weight_tensor[lab]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in modelo.parameters() if p.requires_grad),
                max_norm=GRAD_CLIP,
            )
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            corretos += (logits.argmax(1) == lab).sum().item()
            total += images.size(0)

        train_loss = total_loss / total
        train_acc = corretos / total

        # Validação
        modelo.eval()
        vl_loss = vl_corr = vl_tot = 0
        with torch.no_grad():
            for images, lab in loader_val:
                images, lab = images.to(DEVICE), lab.to(DEVICE)
                logits = modelo(images)
                vl_loss += criterion(logits, lab).mean().item() * images.size(0)
                vl_corr += (logits.argmax(1) == lab).sum().item()
                vl_tot += images.size(0)

        val_loss = vl_loss / vl_tot
        val_acc = vl_corr / vl_tot

        logging.info(
            f"[{fase_nome}] Epoch {epoch+1}/{epochs} — "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in modelo.state_dict().items()}
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= patience:
                logging.info(f"Early stopping após {patience} épocas sem melhora.")
                break

    if best_state:
        modelo.load_state_dict(best_state)


# ── Stage 2: extração de embeddings + XGBoost ───────────────────────────────

def extrair_embeddings_por_talhao(
    modelo: MobileNetV3Classificador,
    registros: list[list[tuple[str, int]]],
    labels: list[int],
    meses: list[int | None],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrai embeddings por talhão de forma memory-efficient.

    Carrega imagens em chunks de BATCH_SIZE_EMBED, extrai embeddings e
    descarta os pixels imediatamente — nunca mantém todas as imagens em RAM.

    Features por talhão:
        mean(D) + min(D) + max(D) + last(D) + [mes, n_tps, last_dia]
    """
    modelo.eval()
    embed_dim = modelo.embed_dim

    # Passa 1: montar lista flat de (caminho, dia, talhao_idx) sem carregar pixels
    flat: list[tuple[str, int, int]] = []
    for t_idx, items in enumerate(registros):
        for caminho, dia in items[:MAX_SEQ_LEN]:
            flat.append((caminho, dia, t_idx))

    # Passa 2: extrair embeddings em chunks (carrega pixels só do chunk atual)
    all_embeddings = np.empty((len(flat), embed_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(flat), BATCH_SIZE_EMBED):
            chunk = flat[i : i + BATCH_SIZE_EMBED]
            imgs = np.stack([preprocessar_imagem(c) for c, _, _ in chunk])
            batch = torch.tensor(imgs, dtype=torch.float32).to(DEVICE)
            all_embeddings[i : i + len(chunk)] = (
                modelo.extrair_embedding(batch).cpu().numpy()
            )
            del imgs, batch  # libera imediatamente

    # Passa 3: agrupar por talhão e agregar
    # Montar boundaries a partir dos talhao_idx
    boundaries: list[tuple[int, int]] = []
    start = 0
    for t_idx in range(len(registros)):
        n = min(len(registros[t_idx]), MAX_SEQ_LEN)
        boundaries.append((start, start + n))
        start += n

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for (s, e), label, mes in zip(boundaries, labels, meses):
        embs = all_embeddings[s:e]
        dias_t = [flat[j][1] for j in range(s, e)]

        mean = embs.mean(axis=0)
        mn   = embs.min(axis=0)
        mx   = embs.max(axis=0)
        last = embs[-1]

        mes_val  = float(mes) if mes is not None else 0.0
        n_tps    = float(e - s)
        last_dia = float(dias_t[-1])

        vetor = np.concatenate([mean, mn, mx, last, [mes_val, n_tps, last_dia]])
        X_list.append(vetor)
        y_list.append(label)

    return np.stack(X_list), np.array(y_list, dtype=np.int64)


def treinar_xgboost(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.info("=== V5: MobileNetV3-Small → embeddings → XGBoost ===")

    # ── Carregar dados ────────────────────────────────────────────────
    logging.info("Carregando dados...")
    registros, labels, meses = carregar_dados(DB_PATH)
    total_imgs = sum(min(len(r), MAX_SEQ_LEN) for r in registros)
    logging.info(f"Total: {len(registros)} talhões, {total_imgs} imagens")

    if not registros:
        logging.error("Nenhum talhão com imagens disponíveis.")
        return

    seq_lens = Counter(min(len(r), MAX_SEQ_LEN) for r in registros)
    for n, count in sorted(seq_lens.items()):
        logging.info(f"  Talhões com {n} timestep(s): {count}")
    for i, c in enumerate(CLASSES):
        logging.info(f"  {c}: {labels.count(i)} talhões")

    # ── Split por talhão (evita data leakage) ─────────────────────────
    indices = list(range(len(registros)))
    idx_tr, idx_val = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=SEED,
    )
    reg_tr  = [registros[i] for i in idx_tr]
    lab_tr  = [labels[i]    for i in idx_tr]
    mes_tr  = [meses[i]     for i in idx_tr]
    reg_val = [registros[i] for i in idx_val]
    lab_val = [labels[i]    for i in idx_val]
    mes_val = [meses[i]     for i in idx_val]

    logging.info(f"Treino: {len(reg_tr)} talhões | Validação: {len(reg_val)} talhões")

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 — Treinar MobileNetV3-Small (classificador por-imagem)
    # ══════════════════════════════════════════════════════════════════
    logging.info("══ STAGE 1: Treinando MobileNetV3-Small ══")

    ds_tr   = ImagemIndividualDataset(reg_tr, lab_tr)
    ds_val1 = ImagemIndividualDataset(reg_val, lab_val)
    use_pin = DEVICE.type == 'cuda'
    n_workers = 2 if use_pin else 0
    loader_tr  = DataLoader(ds_tr,   batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=n_workers, pin_memory=use_pin)
    loader_val = DataLoader(ds_val1, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=n_workers, pin_memory=use_pin)
    logging.info(f"Stage 1 — Treino: {len(ds_tr)} imgs | Val: {len(ds_val1)} imgs")

    # Class weights (por imagem, não por talhão)
    flat_labels = [lb for items, lb in zip(reg_tr, lab_tr) for _ in items[:MAX_SEQ_LEN]]
    pesos = compute_class_weight(
        'balanced', classes=np.arange(len(CLASSES)), y=np.array(flat_labels),
    )
    cw_tensor = torch.tensor(pesos, dtype=torch.float32).to(DEVICE)
    logging.info(f"Class weights: {dict(zip(CLASSES, pesos))}")

    modelo = MobileNetV3Classificador(len(CLASSES)).to(DEVICE)
    total_p = sum(p.numel() for p in modelo.parameters())
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(
        f"Backbone: {BACKBONE_NAME} | embed_dim: {modelo.embed_dim} | "
        f"params: {total_p:,} total, {train_p:,} treináveis"
    )

    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=LABEL_SMOOTHING)

    # Fase 1: backbone congelado
    logging.info("── Fase 1: cabeça (backbone congelado) ──")
    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE1,
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, criterion,
                 EPOCHS_FASE1, cw_tensor, "Fase1", patience=5)

    # Fase 2: fine-tuning
    logging.info(f"── Fase 2: fine-tuning (últimos {FINE_TUNE_BLOCOS} blocos) ──")
    modelo.descongelar_ultimos_blocos(FINE_TUNE_BLOCOS)
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Parâmetros treináveis: {train_p:,}")

    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE2,
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, criterion,
                 EPOCHS_FASE2, cw_tensor, "Fase2", patience=5)

    # Salvar backbone
    os.makedirs(MODELO_SAIDA, exist_ok=True)
    backbone_path = os.path.join(MODELO_SAIDA, 'backbone.pt')
    torch.save(modelo.state_dict(), backbone_path)
    logging.info(f"Backbone salvo em: {backbone_path}")

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2 — Extrair embeddings + treinar XGBoost
    # ══════════════════════════════════════════════════════════════════
    logging.info("══ STAGE 2: Extraindo embeddings + treinando XGBoost ══")

    t0 = time.perf_counter()
    X_tr_xgb, y_tr_xgb = extrair_embeddings_por_talhao(modelo, reg_tr, lab_tr, mes_tr)
    X_val_xgb, y_val_xgb = extrair_embeddings_por_talhao(modelo, reg_val, lab_val, mes_val)
    t_extract = time.perf_counter() - t0

    n_feat = X_tr_xgb.shape[1]
    logging.info(
        f"Embeddings extraídos em {t_extract:.1f}s — "
        f"X_tr: {X_tr_xgb.shape}, X_val: {X_val_xgb.shape} ({n_feat} features)"
    )

    logging.info("Treinando XGBoost...")
    clf = treinar_xgboost(X_tr_xgb, y_tr_xgb, X_val_xgb, y_val_xgb, len(CLASSES))

    xgb_path = os.path.join(MODELO_SAIDA, 'xgboost.json')
    clf.save_model(xgb_path)
    logging.info(f"XGBoost salvo em: {xgb_path}")

    # ══════════════════════════════════════════════════════════════════
    #  AVALIAÇÃO
    # ══════════════════════════════════════════════════════════════════
    logging.info("══ AVALIAÇÃO FINAL ══")

    # CNN Stage 1 (per-image)
    modelo.eval()
    y_true_img, y_pred_img = [], []
    with torch.no_grad():
        for imgs, lab in loader_val:
            logits = modelo(imgs.to(DEVICE))
            y_pred_img.extend(logits.argmax(1).cpu().numpy())
            y_true_img.extend(lab.numpy())

    logging.info(
        f"CNN per-image — Acc: {accuracy_score(y_true_img, y_pred_img):.4f} | "
        f"F1-macro: {f1_score(y_true_img, y_pred_img, average='macro'):.4f}"
    )

    # XGBoost Stage 2 (per-talhão)
    y_pred_xgb = clf.predict(X_val_xgb)

    acc_xgb  = accuracy_score(y_val_xgb, y_pred_xgb)
    f1_macro = f1_score(y_val_xgb, y_pred_xgb, average='macro')
    f1_per   = f1_score(y_val_xgb, y_pred_xgb, average=None)

    logging.info(f"XGBoost per-talhão — Acc: {acc_xgb:.4f} | F1-macro: {f1_macro:.4f}")
    for cls_name, f1_cls in zip(CLASSES, f1_per):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_val_xgb, y_pred_xgb, target_names=CLASSES,
    ))

    cm = confusion_matrix(y_val_xgb, y_pred_xgb, labels=list(range(len(CLASSES))))
    logging.info("Confusion Matrix (linhas=real, colunas=predito):")
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    logging.info(header)
    for cls, row in zip(CLASSES, cm):
        logging.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    # Top-10 features
    embed_dim = modelo.embed_dim
    nomes = (
        [f'mean_{i}' for i in range(embed_dim)]
        + [f'min_{i}'  for i in range(embed_dim)]
        + [f'max_{i}'  for i in range(embed_dim)]
        + [f'last_{i}' for i in range(embed_dim)]
        + ['mes', 'n_tps', 'last_dia']
    )
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    logging.info("Top-10 feature importances:")
    for rank, fi in enumerate(top_idx, 1):
        logging.info(f"  {rank:>2}. {nomes[fi]:<12} {importances[fi]:.4f}")

    # Tempo de inferência
    t0 = time.perf_counter()
    X_bench, _ = extrair_embeddings_por_talhao(modelo, reg_val, lab_val, mes_val)
    clf.predict(X_bench)
    t_inf = time.perf_counter() - t0
    ms_per = (t_inf / len(reg_val)) * 1000
    logging.info(f"Tempo médio de inferência: {ms_per:.2f} ms/talhão ({len(reg_val)} talhões)")


if __name__ == '__main__':
    main()
