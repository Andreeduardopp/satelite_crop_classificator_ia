"""
Classificador rápido de culturas via XGBoost usando apenas os vetores
`sigmoides_iv` já persistidos no banco principal.

Contexto:
    O banco `./src/bkp/26-04-06.dados.db` guarda, para cada talhão, um
    dicionário JSON `sigmoides_iv` com uma entrada por timestep:

        {
            "mascara_{uuid}_v_d{dia}": [35 floats em [0, 1]],
            ...
        }

    Cada vetor de 35 posições vem de outro modelo que já processa a
    imagem. Este script é um TESTE RÁPIDO para descobrir quanta
    informação sobre a cultura está nesses vetores sozinhos — sem tocar
    em imagens/TIFFs/bandas espectrais.

Feature engineering:
    Para cada talhão, agregamos os N timesteps (variável, 1 a ~4) em
    estatísticas por-posição, gerando um vetor fixo de 35*4 + extras:
        - mean  (35)
        - min   (35)
        - max   (35)
        - last  (35)  ← último timestep (dia mais alto)
        - mes   (1)
        - n_tps (1)   ← nº de timesteps observados
        - last_dia (1)

    Total: 143 features. Invariantes ao nº de observações.

Uso:
    python src/treinamento/treinar_xgboost_sigmoides.py
"""

import os
import json
import sqlite3
import logging
from datetime import datetime

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, accuracy_score,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(
    LOG_DIR, f'treino_xgb_sigmoides_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
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
DB_PATH         = './src/bkp/26-04-06.dados.db'
TABELA          = 'culturas'
MODELO_SAIDA    = './modelos/xgb_sigmoides.json'
SEED            = 42
SIGMOID_LEN     = 35
TEST_SIZE       = 0.2

# Teste rápido: só milho/soja/trigo, até 2000 talhões por cultura (~6k total)
CLASSES_ALVO    = ['milho', 'soja', 'trigo']
MAX_POR_CLASSE  = 2000


# ── Extração ──────────────────────────────────────────────────────────────────

def _dia_da_chave(chave: str) -> int:
    """Extrai o dia da chave 'mascara_{uuid}_v_d{dia}'."""
    try:
        return int(chave.rsplit('_d', 1)[-1])
    except ValueError:
        return 0


def vetor_features(sigmoides: dict[str, list[float]], mes: int | None) -> np.ndarray | None:
    """
    Converte o dicionário de sigmoides em um vetor de 143 features.

    Retorna None se nenhum timestep válido.
    """
    # Ordena timesteps por dia
    itens = sorted(sigmoides.items(), key=lambda kv: _dia_da_chave(kv[0]))
    vetores = []
    dias = []
    for chave, vec in itens:
        if not isinstance(vec, list) or len(vec) != SIGMOID_LEN:
            continue
        vetores.append(vec)
        dias.append(_dia_da_chave(chave))

    if not vetores:
        return None

    matriz = np.asarray(vetores, dtype=np.float32)  # (T, 35)

    mean = matriz.mean(axis=0)
    mn = matriz.min(axis=0)
    mx = matriz.max(axis=0)
    last = matriz[-1]

    mes_val = float(mes) if mes is not None else 0.0
    n_tps = float(len(vetores))
    last_dia = float(dias[-1]) if dias else 0.0

    return np.concatenate([mean, mn, mx, last, [mes_val, n_tps, last_dia]]).astype(np.float32)


def carregar_dataset(db_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Lê o banco e retorna (X, y, nomes_classes).

    Restringe a `CLASSES_ALVO` e limita a `MAX_POR_CLASSE` talhões por
    cultura (para ficar num teste rápido de ~6k linhas totais).
    """
    cultura_para_id = {c: i for i, c in enumerate(CLASSES_ALVO)}
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    contagem = {c: 0 for c in CLASSES_ALVO}
    descartados = 0

    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join("?" * len(CLASSES_ALVO))
        rows = conn.execute(
            f"""
            SELECT cultura, mes, sigmoides_iv
            FROM   {TABELA}
            WHERE  sigmoides_iv IS NOT NULL
              AND  sigmoides_iv != ''
              AND  sigmoides_iv != '{{}}'
              AND  cultura IN ({placeholders})
            ORDER BY RANDOM()
            """,
            CLASSES_ALVO,
        ).fetchall()

    for cultura, mes, sig_str in rows:
        if cultura not in cultura_para_id:
            continue
        if contagem[cultura] >= MAX_POR_CLASSE:
            continue
        try:
            sigmoides = json.loads(sig_str)
        except (json.JSONDecodeError, TypeError):
            descartados += 1
            continue
        if not isinstance(sigmoides, dict) or not sigmoides:
            descartados += 1
            continue

        vetor = vetor_features(sigmoides, mes)
        if vetor is None:
            descartados += 1
            continue

        X_list.append(vetor)
        y_list.append(cultura_para_id[cultura])
        contagem[cultura] += 1

        # Corte antecipado quando todas as classes estão cheias
        if all(contagem[c] >= MAX_POR_CLASSE for c in CLASSES_ALVO):
            break

    logging.info(
        f"Lidos: {len(rows)} | usados: {len(X_list)} | descartados: {descartados}"
    )
    for c in CLASSES_ALVO:
        logging.info(f"  {c}: {contagem[c]}")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, CLASSES_ALVO


# ── Treino ────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED)

    logging.info("=== XGBoost sobre sigmoides_iv (teste rápido) ===")
    logging.info(f"Lendo {DB_PATH}...")

    X, y, classes = carregar_dataset(DB_PATH)
    logging.info(f"Dataset final: X={X.shape}, y={y.shape}")
    logging.info(f"Classes ({len(classes)}): {classes}")
    for i, c in enumerate(classes):
        n = int((y == i).sum())
        logging.info(f"  {c}: {n}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    logging.info(f"Treino: {len(X_tr)} | Validação: {len(X_val)}")

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        objective='multi:softprob',
        num_class=len(classes),
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=SEED,
        n_jobs=-1,
    )

    logging.info("Treinando XGBoost...")
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Avaliação ─────────────────────────────────────────────────────
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_per_class = f1_score(y_val, y_pred, average=None)

    logging.info(f"Accuracy:  {acc:.4f}")
    logging.info(f"F1-macro:  {f1_macro:.4f}")
    for cls_name, f1_cls in zip(classes, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_val, y_pred, target_names=classes
    ))

    logging.info("Confusion Matrix (linhas=real, colunas=predito):")
    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(classes))))
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in classes)
    logging.info(header)
    for cls, row in zip(classes, cm):
        logging.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    # Top-10 features mais importantes
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    logging.info("Top-10 feature importances:")
    nomes_base = (
        [f'mean_{i}' for i in range(SIGMOID_LEN)] +
        [f'min_{i}'  for i in range(SIGMOID_LEN)] +
        [f'max_{i}'  for i in range(SIGMOID_LEN)] +
        [f'last_{i}' for i in range(SIGMOID_LEN)] +
        ['mes', 'n_tps', 'last_dia']
    )
    for rank, idx in enumerate(top_idx, start=1):
        logging.info(f"  {rank:>2}. {nomes_base[idx]:<12} {importances[idx]:.4f}")

    # Salva modelo
    os.makedirs(os.path.dirname(MODELO_SAIDA), exist_ok=True)
    clf.save_model(MODELO_SAIDA)
    logging.info(f"Modelo salvo em: {MODELO_SAIDA}")


if __name__ == '__main__':
    main()
