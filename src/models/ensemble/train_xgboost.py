"""
XGBoost Stacking: aprende a combinar as predições de 3 modelos (ensemble).

Recebe features extraídas dos 3 backbones (pooled + probabilidades) e treina
um XGBoost que aprende:
  - Qual modelo confiar para cada classe
  - Quando os modelos discordam, qual costuma acertar
  - Combinações não-lineares de features entre os modelos

Uso:
    python train_xgboost.py
"""

import os
import time
import json
import logging
from datetime import datetime
import numpy as np

import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

from data import CLASSES
from model import BACKBONE_CONFIG

# -- Paths ---------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(_HERE, 'features')
MODEL_DIR    = os.path.join(_HERE, 'artifacts')
METRICS_DIR  = os.path.join(_HERE, 'metrics')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

# Feature dims por backbone (para mapeamento de importância)
BACKBONE_DIMS = {
    'efficientnet_b0': 1280,
    'resnet50': 2048,
    'convnext_tiny': 768,
}


def build_feature_map(n_features: int) -> dict:
    """Constrói mapa de ranges de features para cada backbone."""
    feature_map = {}
    offset = 0
    for bb_name, dim in BACKBONE_DIMS.items():
        feature_map[f'{bb_name}_pooled'] = (offset, offset + dim)
        offset += dim
        feature_map[f'{bb_name}_probs'] = (offset, offset + len(CLASSES))
        offset += len(CLASSES)

    feature_map['dias'] = (offset, offset + 3)
    offset += 3
    feature_map['mes'] = (offset, offset + 1)
    offset += 1
    feature_map['count'] = (offset, offset + 1)
    offset += 1

    return feature_map


def main():
    log.info("="*70)
    log.info("XGBoost Stacking: Combinando Ensemble de 3 Backbones")
    log.info("="*70)

    # 1. Carregar features
    f_X_tr = os.path.join(FEATURES_DIR, 'X_treino.npy')
    f_y_tr = os.path.join(FEATURES_DIR, 'y_treino.npy')
    f_X_te = os.path.join(FEATURES_DIR, 'X_teste.npy')
    f_y_te = os.path.join(FEATURES_DIR, 'y_teste.npy')

    if not all(os.path.exists(p) for p in [f_X_tr, f_y_tr, f_X_te, f_y_te]):
        log.error("Features não encontradas! Execute extrator.py primeiro.")
        return

    log.info("Carregando features...")
    t0 = time.time()
    X_train = np.load(f_X_tr)
    y_train = np.load(f_y_tr)
    X_test  = np.load(f_X_te)
    y_test  = np.load(f_y_te)
    log.info(f"Carregado em {time.time()-t0:.2f}s")
    log.info(f"  X_train: {X_train.shape} ({X_train.nbytes/1e6:.1f} MB)")
    log.info(f"  X_test:  {X_test.shape}")

    assert np.all(np.isfinite(X_train)), "NaN/Inf em X_train!"
    assert np.all(np.isfinite(X_test)), "NaN/Inf em X_test!"

    n_features = X_train.shape[1]
    feature_map = build_feature_map(n_features)
    log.info(f"Feature layout:")
    for name, (lo, hi) in feature_map.items():
        log.info(f"  [{lo:5d}:{hi:5d}] {name} ({hi-lo} features)")

    # 2. Distribuição
    log.info("Distribuição:")
    for cid, name in enumerate(CLASSES):
        n_tr = (y_train == cid).sum()
        n_te = (y_test == cid).sum()
        log.info(f"  {name:8s} — treino: {n_tr:5d} | teste: {n_te:4d}")

    # 3. Class weights (tolerante a classes ausentes)
    present_classes = np.unique(y_train)
    pesos_present = compute_class_weight('balanced', classes=present_classes, y=y_train)
    sample_weights = np.ones_like(y_train, dtype=float)
    for idx, cls_id in enumerate(present_classes):
        sample_weights[y_train == cls_id] = pesos_present[idx]

    # 4. XGBoost
    xgb_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(CLASSES),
        'max_depth': 6,
        'learning_rate': 0.08,
        'n_estimators': 500,
        'colsample_bytree': 0.3,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'use_label_encoder': False,
        'tree_method': 'hist',
        'random_state': 42,
        'early_stopping_rounds': 20,
    }

    log.info("Hiperparâmetros:")
    for k, v in xgb_params.items():
        log.info(f"  {k}: {v}")

    modelo = xgb.XGBClassifier(**xgb_params)

    # 5. Treinar
    log.info("Treinando XGBoost stacking...")
    t_train = time.time()
    modelo.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=25,
    )
    tempo_treino = time.time() - t_train
    log.info(f"Treinamento: {tempo_treino:.2f}s (parou em {modelo.best_iteration+1} árvores)")

    # 6. Salvar modelo
    os.makedirs(MODEL_DIR, exist_ok=True)
    modelo_path = os.path.join(MODEL_DIR, 'xgboost_ensemble.json')
    modelo.save_model(modelo_path)
    log.info(f"Modelo salvo: {modelo_path}")

    # 7. Importância por nível (qual backbone contribui mais?)
    importances = modelo.feature_importances_
    log.info("Importância por componente do ensemble:")
    total_imp = importances.sum()
    for name, (lo, hi) in feature_map.items():
        if hi <= n_features:
            agg = importances[lo:hi].sum()
            log.info(f"  {name:30s}: {agg:.4f} ({100*agg/total_imp:.1f}%)")

    # Top 15 features individuais
    top_idx = np.argsort(importances)[-15:][::-1]
    log.info("Top 15 features individuais:")
    for rank, idx in enumerate(top_idx, 1):
        # Identificar a qual componente pertence
        component = "metadata"
        for name, (lo, hi) in feature_map.items():
            if lo <= idx < hi:
                component = name
                break
        log.info(f"  {rank:2d}. Feature {idx:5d} ({component}): {importances[idx]:.6f}")

    # 8. Inferência
    log.info("Inferência no teste...")
    t_inf = time.time()
    y_pred = modelo.predict(X_test)
    tempo_inf = time.time() - t_inf
    tempo_medio = (tempo_inf / len(X_test)) * 1000

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    log.info("="*70)
    log.info(f"RESULTADO ENSEMBLE: Acc={acc:.4f} | F1-macro={f1_macro:.4f}")
    log.info("="*70)

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

    log.info(f"Inferência: {tempo_medio:.3f} ms/amostra")

    # 9. Salvar métricas
    os.makedirs(METRICS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    m_path = os.path.join(METRICS_DIR, f'ensemble_xgb_{ts}.json')

    # Importância por componente
    component_importance = {}
    for name, (lo, hi) in feature_map.items():
        if hi <= n_features:
            component_importance[name] = float(importances[lo:hi].sum())

    with open(m_path, 'w') as f:
        json.dump({
            'f1_macro': float(f1_macro),
            'accuracy': float(acc),
            'f1_per_class': {c: float(v) for c, v in zip(CLASSES, f1_per_class)},
            'n_features': n_features,
            'n_models': len([k for k in BACKBONE_DIMS
                             if os.path.exists(os.path.join(MODEL_DIR, f'pesos_{k}.pt'))]),
            'best_iteration': int(modelo.best_iteration),
            'training_time_s': tempo_treino,
            'inference_ms_per_sample': tempo_medio,
            'component_importance': component_importance,
        }, f, indent=4)

    log.info(f"Métricas salvas: {m_path}")
    log.info("="*70)


if __name__ == '__main__':
    main()
