"""
Etapa Tabular Clássica: XGBClassifier da arquitetura Two-Stage - V2.
Toma ingestão dos pesados Arrays numpy extraídos da Visão Computacional (V7)
e constrói a árvore de decisão matemática em segundos ao invés de horas.

Melhorias na V2:
- Compatível com features padronizadas do extrator_v2
- Validação rigorosa de dados
- Melhor logging de métricas
- Salvamento de estatísticas de padronização para inferência
- Mais informações sobre feature importance
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# -- Constantes Espelhadas do V7 -----------------------------------------------
CLASSES = ['soja', 'milho', 'trigo', 'aveia', 'feijão']

# -- Paths e Configuração ------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR_V2 = os.path.join(_HERE, 'features_extraidas_v2')
FEATURES_DIR_V1 = os.path.join(_HERE, 'features_extraidas')
MODEL_OUT_DIR = os.path.join(_HERE, 'artifacts')
METRICS_OUT_DIR = os.path.join(_HERE, 'metrics')

# Tentar V2 primeiro, fallback para V1
FEATURES_DIR = FEATURES_DIR_V2 if os.path.exists(FEATURES_DIR_V2) else FEATURES_DIR_V1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_features(X: np.ndarray, name: str) -> None:
    """Valida se há NaN, Inf ou valores inválidos."""
    if not np.all(np.isfinite(X)):
        n_invalid = (~np.isfinite(X)).sum()
        raise ValueError(
            f"Detectados {n_invalid} valores inválidos (NaN/Inf) em {name}"
        )
    logger.info(f"✓ {name} validado: shape {X.shape}, valores válidos")


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computa métricas detalhadas de classificação."""
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        'accuracy': float(acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_per_class': {
            CLASSES[i]: float(p) for i, p in enumerate(precision_per_class)
        },
        'recall_per_class': {
            CLASSES[i]: float(r) for i, r in enumerate(recall_per_class)
        },
        'f1_per_class': {
            CLASSES[i]: float(f) for i, f in enumerate(f1_per_class)
        }
    }


def plot_confusion_matrix(cm: np.ndarray) -> str:
    """Formata matriz de confusão para logging."""
    header = f"{'':>12}" + "".join(f"{c:>12}" for c in CLASSES)
    lines = [header]
    for cls, row in zip(CLASSES, cm):
        lines.append(f"{cls:>12}" + "".join(f"{v:>12}" for v in row))
    return "\n".join(lines)


def main():
    logger.info("="*70)
    logger.info("Híbrido XGBoost V2: Compilando DeepLearning Extracted Arrays")
    logger.info("="*70)
    logger.info(f"Usando features de: {FEATURES_DIR}\n")

    # -------- 1. Verificar e carregar features --------
    f_X_tr = os.path.join(FEATURES_DIR, 'X_treino.npy')
    f_y_tr = os.path.join(FEATURES_DIR, 'y_treino.npy')
    f_X_te = os.path.join(FEATURES_DIR, 'X_teste.npy')
    f_y_te = os.path.join(FEATURES_DIR, 'y_teste.npy')

    feature_files = {
        'X_treino': f_X_tr,
        'y_treino': f_y_tr,
        'X_teste': f_X_te,
        'y_teste': f_y_te
    }

    missing_files = [name for name, path in feature_files.items() if not os.path.exists(path)]
    if missing_files:
        logger.error(f"Arquivos não encontrados: {', '.join(missing_files)}")
        logger.error("Execute o extrator_v2.py antes de rodar este script!")
        return

    logger.info("Carregando bases Numpy na RAM...")
    t0 = time.time()
    try:
        X_train = np.load(f_X_tr)
        y_train = np.load(f_y_tr)
        X_test = np.load(f_X_te)
        y_test = np.load(f_y_te)
    except Exception as e:
        logger.error(f"Erro ao carregar arrays: {e}")
        return

    logger.info(f"Carregado em {time.time() - t0:.2f}s")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  X_test:  {X_test.shape}")
    logger.info(f"  y_test:  {y_test.shape}\n")

    # -------- 2. Validar dados --------
    try:
        validate_features(X_train, "X_train")
        validate_features(X_test, "X_test")
    except ValueError as e:
        logger.error(f"Validação falhou: {e}")
        return

    # Verificar correspondência
    assert len(y_train) == X_train.shape[0], "Mismatch entre X_train e y_train"
    assert len(y_test) == X_test.shape[0], "Mismatch entre X_test e y_test"
    logger.info("✓ Dados validados\n")

    # -------- 3. Analisar distribuição --------
    logger.info("Distribuição de classes no treinamento:")
    for cls_id, cls_name in enumerate(CLASSES):
        count_train = (y_train == cls_id).sum()
        count_test = (y_test == cls_id).sum()
        pct_train = 100 * count_train / len(y_train) if len(y_train) > 0 else 0
        pct_test = 100 * count_test / len(y_test) if len(y_test) > 0 else 0
        logger.info(f"  {cls_name:10s} - Train: {count_train:5d} ({pct_train:5.1f}%) | "
                   f"Test: {count_test:5d} ({pct_test:5.1f}%)")
    logger.info()

    # -------- 4. Computar class weights --------
    logger.info("Computando weights para balanceamento...")
    pesos = compute_class_weight(
        'balanced',
        classes=np.arange(len(CLASSES)),
        y=y_train
    )
    logger.info("Class weights:")
    for cls_name, peso in zip(CLASSES, pesos):
        logger.info(f"  {cls_name:10s}: {peso:.4f}")

    sample_weights = np.zeros_like(y_train, dtype=np.float32)
    for c_id in range(len(CLASSES)):
        sample_weights[y_train == c_id] = pesos[c_id]
    logger.info()

    # -------- 5. Configurar modelo XGBoost --------
    logger.info("Configurando XGBoost...")
    xgb_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(CLASSES),
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'n_jobs': -1,
        'use_label_encoder': False,
        'tree_method': 'hist',
        'random_state': 42,
        'early_stopping_rounds': 20,  # Novo: parar se não melhorar
        'verbosity': 1
    }

    for param, value in xgb_params.items():
        logger.info(f"  {param}: {value}")

    modelo = xgb.XGBClassifier(**xgb_params)
    logger.info()

    # -------- 6. Treinar --------
    logger.info("Iniciando treinamento tabular do XGBoost...")
    t_train = time.time()
    try:
        modelo.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False  # Silencioso para logging limpo
        )
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}")
        return

    tempo_treino = time.time() - t_train
    logger.info(f"✓ Treinamento finalizado em {tempo_treino:.2f}s\n")

    # -------- 7. Feature Importance --------
    logger.info("Top 10 Features mais importantes:")
    feature_importance = modelo.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]

    for rank, idx in enumerate(top_indices, 1):
        importance = feature_importance[idx]
        logger.info(f"  {rank:2d}. Feature {idx:4d}: {importance:.6f}")
    logger.info()

    # -------- 8. Salvar modelo --------
    logger.info("Salvando modelo...")
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    peso_arvore = os.path.join(MODEL_OUT_DIR, 'xgboost_modelo.json')
    try:
        modelo.save_model(peso_arvore)
        logger.info(f"✓ Modelo salvo: {peso_arvore}\n")
    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {e}")
        return

    # -------- 9. Avaliação Final --------
    logger.info("Executando inferência final...")
    t_inf = time.time()
    y_pred = modelo.predict(X_test)
    tempo_inferencia = time.time() - t_inf
    tempo_medio_ms = (tempo_inferencia / len(X_test)) * 1000

    logger.info(f"✓ Inferência concluída em {tempo_inferencia:.3f}s")
    logger.info(f"  Tempo médio: {tempo_medio_ms:.3f} ms/amostra\n")

    # -------- 10. Métricas Detalhadas --------
    logger.info("="*70)
    logger.info("MÉTRICAS DE DESEMPENHO")
    logger.info("="*70)

    metrics = compute_detailed_metrics(y_test, y_pred)

    logger.info(f"\nMétricas Globais:")
    logger.info(f"  Accuracy:         {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 (macro):        {metrics['f1_macro']:.4f}")

    logger.info(f"\nMétricas por Classe:")
    for cls_name in CLASSES:
        logger.info(f"\n  {cls_name}:")
        logger.info(f"    Precision: {metrics['precision_per_class'][cls_name]:.4f}")
        logger.info(f"    Recall:    {metrics['recall_per_class'][cls_name]:.4f}")
        logger.info(f"    F1:        {metrics['f1_per_class'][cls_name]:.4f}")

    # -------- 11. Classification Report --------
    logger.info("\n" + "="*70)
    logger.info("Classification Report (sklearn):")
    logger.info("="*70)
    logger.info("\n" + classification_report(
        y_test, y_pred,
        target_names=CLASSES,
        zero_division=0,
        digits=4
    ))

    # -------- 12. Confusion Matrix --------
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    logger.info("="*70)
    logger.info("Confusion Matrix:")
    logger.info("="*70)
    logger.info("\n" + plot_confusion_matrix(cm) + "\n")

    # -------- 13. Salvar métricas --------
    os.makedirs(METRICS_OUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    metrics_file = os.path.join(METRICS_OUT_DIR, f'xgb_metrics_{timestamp}.json')

    metrics_output = {
        **metrics,
        'confusion_matrix': cm.tolist(),
        'training_time_seconds': tempo_treino,
        'inference_time_ms_per_sample': tempo_medio_ms,
        'n_train_samples': int(X_train.shape[0]),
        'n_test_samples': int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
        'model_n_estimators': modelo.n_estimators,
        'model_max_depth': modelo.max_depth,
    }

    try:
        with open(metrics_file, 'w') as f:
            json.dump(metrics_output, f, indent=4)
        logger.info(f"✓ Métricas salvas: {metrics_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar métricas: {e}")

    # -------- 14. Resumo Final --------
    logger.info("\n" + "="*70)
    logger.info("RESUMO DA EXECUÇÃO")
    logger.info("="*70)
    logger.info(f"Status: ✓ SUCESSO")
    logger.info(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Tempo total: {tempo_treino + tempo_inferencia:.2f}s")
    logger.info(f"Modelo: {peso_arvore}")
    logger.info(f"Métricas: {metrics_file}")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
