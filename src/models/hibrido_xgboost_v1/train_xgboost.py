"""
XGBoost sobre features multi-nível extraídas do V7.

Adaptado para o novo extrator que fornece ~9400 colunas com informação
de múltiplos níveis do modelo (pooled, FiLM, temporal stats, multi-escala).

Inclui regularização adequada para alta dimensionalidade
(colsample_bytree, subsample, early stopping).
"""

import os
import time
import json
import logging
from datetime import datetime
import numpy as np

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# -- Constantes -------------------------------------------------------------------
CLASSES = ['soja', 'milho', 'trigo', 'aveia', 'feijão']

_HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(_HERE, 'features_extraidas')
MODEL_OUT_DIR = os.path.join(_HERE, 'artifacts')
METRICS_OUT_DIR = os.path.join(_HERE, 'metrics')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)


def main():
    log.info("="*60)
    log.info("XGBoost Multi-Level: Treinando sobre Features Profundas")
    log.info("="*60)

    # 1. Carregar features
    f_X_tr = os.path.join(FEATURES_DIR, 'X_treino.npy')
    f_y_tr = os.path.join(FEATURES_DIR, 'y_treino.npy')
    f_X_te = os.path.join(FEATURES_DIR, 'X_teste.npy')
    f_y_te = os.path.join(FEATURES_DIR, 'y_teste.npy')

    if not all(os.path.exists(p) for p in [f_X_tr, f_y_tr, f_X_te, f_y_te]):
        log.error("Features não encontradas. Execute extrator.py primeiro!")
        return

    log.info("Carregando features na RAM...")
    t0 = time.time()
    X_train = np.load(f_X_tr)
    y_train = np.load(f_y_tr)
    X_test  = np.load(f_X_te)
    y_test  = np.load(f_y_te)
    log.info(f"Carregado em {time.time()-t0:.2f}s")
    log.info(f"  X_train: {X_train.shape}  ({X_train.nbytes/1e6:.1f} MB)")
    log.info(f"  X_test:  {X_test.shape}")

    # Validar
    assert np.all(np.isfinite(X_train)), "NaN/Inf em X_train!"
    assert np.all(np.isfinite(X_test)),  "NaN/Inf em X_test!"
    log.info("  Dados validados (sem NaN/Inf)")

    # 2. Distribuição de classes
    log.info("Distribuição:")
    for cid, name in enumerate(CLASSES):
        n_tr = (y_train == cid).sum()
        n_te = (y_test == cid).sum()
        log.info(f"  {name:8s} — treino: {n_tr:5d} ({100*n_tr/len(y_train):5.1f}%) | "
                 f"teste: {n_te:4d} ({100*n_te/len(y_test):5.1f}%)")

    # 3. Class weights
    pesos = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=y_train)
    sample_weights = np.zeros_like(y_train, dtype=float)
    for c_id in range(len(CLASSES)):
        sample_weights[y_train == c_id] = pesos[c_id]

    # 4. XGBoost com regularização para alta dimensionalidade
    #    - colsample_bytree=0.3: cada árvore vê 30% das features → evita overfitting
    #    - subsample=0.8: cada árvore vê 80% das amostras → mais diversidade
    #    - early_stopping_rounds=20: para se não melhorar
    n_features = X_train.shape[1]
    log.info(f"Features totais: {n_features}")

    xgb_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(CLASSES),
        'max_depth': 6,
        'learning_rate': 0.08,
        'n_estimators': 500,
        'colsample_bytree': 0.3,      # Cada árvore vê 30% das features
        'subsample': 0.8,              # Cada árvore vê 80% das amostras
        'reg_alpha': 0.1,              # L1 regularization
        'reg_lambda': 1.0,             # L2 regularization
        'n_jobs': -1,
        'use_label_encoder': False,
        'tree_method': 'hist',
        'random_state': 42,
        'early_stopping_rounds': 20,
    }

    log.info("Hiperparâmetros XGBoost:")
    for k, v in xgb_params.items():
        log.info(f"  {k}: {v}")

    modelo = xgb.XGBClassifier(**xgb_params)

    # 5. Treinar
    log.info("Treinando...")
    t_train = time.time()
    modelo.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=25
    )
    tempo_treino = time.time() - t_train
    log.info(f"Treinamento concluído em {tempo_treino:.2f}s "
             f"(parou em {modelo.best_iteration + 1} árvores)")

    # 6. Salvar modelo
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    modelo_path = os.path.join(MODEL_OUT_DIR, 'xgboost_modelo.json')
    modelo.save_model(modelo_path)
    log.info(f"Modelo salvo: {modelo_path}")

    # 7. Feature importance — quais níveis importam mais?
    importances = modelo.feature_importances_
    top_idx = np.argsort(importances)[-15:][::-1]

    # Mapear índice → nível de feature
    level_names = {
        (0,    1280): "pooled (pós-attention)",
        (1280, 2560): "post-FiLM pooled",
        (2560, 3840): "temporal mean",
        (3840, 5120): "temporal std",
        (5120, 6400): "temporal max",
        (6400, 7680): "temporal diff 1→2",
        (7680, 8960): "temporal diff 2→3",
        (8960, 9000): "multi-scale block_2",
        (9000, 9112): "multi-scale block_4",
        (9112, 9432): "multi-scale block_6",
    }

    def get_level_name(idx):
        for (lo, hi), name in level_names.items():
            if lo <= idx < hi:
                return name
        return "metadata"

    log.info("Top 15 features mais importantes:")
    for rank, idx in enumerate(top_idx, 1):
        level = get_level_name(idx)
        log.info(f"  {rank:2d}. Feature {idx:5d} ({level}): {importances[idx]:.6f}")

    # Importância agregada por nível
    log.info("Importância agregada por nível:")
    for (lo, hi), name in level_names.items():
        if hi <= n_features:
            agg = importances[lo:hi].sum()
            log.info(f"  {name:30s}: {agg:.4f} ({100*agg/importances.sum():.1f}%)")

    # 8. Inferência e métricas
    log.info("Inferência no conjunto de teste...")
    t_inf = time.time()
    y_pred = modelo.predict(X_test)
    tempo_inf = time.time() - t_inf
    tempo_medio = (tempo_inf / len(X_test)) * 1000

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    log.info("="*60)
    log.info(f"RESULTADO: Acurácia={acc:.4f} | F1-macro={f1_macro:.4f}")
    log.info("="*60)

    for cls_name, f1_cls in zip(CLASSES, f1_per_class):
        log.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    log.info("Classification Report:")
    log.info("\n" + classification_report(
        y_test, y_pred, target_names=CLASSES, zero_division=0, digits=4
    ))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    log.info("Confusion Matrix:")
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    log.info(header)
    for cls, row in zip(CLASSES, cm):
        log.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    log.info(f"Tempo inferência: {tempo_medio:.3f} ms/amostra")

    # 9. Salvar métricas
    os.makedirs(METRICS_OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    m_path = os.path.join(METRICS_OUT_DIR, f'xgb_metrics_{ts}.json')
    with open(m_path, 'w') as f:
        json.dump({
            'f1_macro': float(f1_macro),
            'accuracy': float(acc),
            'f1_per_class': {c: float(v) for c, v in zip(CLASSES, f1_per_class)},
            'n_features': n_features,
            'best_iteration': int(modelo.best_iteration),
            'training_time_s': tempo_treino,
            'inference_ms_per_sample': tempo_medio,
        }, f, indent=4)
    log.info(f"Métricas salvas: {m_path}")


if __name__ == '__main__':
    main()
