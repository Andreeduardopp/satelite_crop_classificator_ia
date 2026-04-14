"""
Script de Comparação: V1 vs V2

Executa ambas as pipelines e compara resultados lado-a-lado.
Útil para validar se a migração para V2 é benéfica.

Uso:
    python compare_v1_v2.py [--skip-v1] [--skip-v2]

Flags:
    --skip-v1   Pular extração V1 (já feita) e ir direto pro treino
    --skip-v2   Pular extração V2 (já feita) e ir direto pro treino
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, Tuple
import argparse

# -- Configuração ---------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['soja', 'milho', 'trigo', 'aveia', 'feijão']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -- Utilities ------------------------------------------------------------------

def load_metrics_from_json(json_path: str) -> Dict:
    """Carrega última métrica gerada."""
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Erro ao carregar {json_path}: {e}")
        return None


def get_latest_metrics(metrics_dir: str) -> Dict:
    """Procura arquivo de métricas mais recente."""
    if not os.path.exists(metrics_dir):
        return None

    json_files = list(Path(metrics_dir).glob('xgb_metrics_*.json'))
    if not json_files:
        return None

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return load_metrics_from_json(str(latest))


def compare_metrics(metrics_v1: Dict, metrics_v2: Dict) -> None:
    """Compara métricas de V1 e V2."""

    if not metrics_v1 and not metrics_v2:
        logger.error("Nenhuma métrica encontrada para comparar!")
        return

    logger.info("\n" + "="*80)
    logger.info("COMPARAÇÃO V1 vs V2")
    logger.info("="*80 + "\n")

    # Header
    header = f"{'Métrica':<30} | {'V1':>12} | {'V2':>12} | {'Delta':>12} | {'Status':>10}"
    logger.info(header)
    logger.info("-" * 80)

    # Metrics a comparar
    metrics_to_compare = [
        ('accuracy', 'Accuracy'),
        ('precision_macro', 'Precision (Macro)'),
        ('recall_macro', 'Recall (Macro)'),
        ('f1_macro', 'F1 (Macro)'),
    ]

    for metric_key, metric_name in metrics_to_compare:
        v1_val = metrics_v1.get(metric_key, None) if metrics_v1 else None
        v2_val = metrics_v2.get(metric_key, None) if metrics_v2 else None

        if v1_val is None and v2_val is None:
            continue

        v1_str = f"{v1_val:.4f}" if v1_val is not None else "N/A"
        v2_str = f"{v2_val:.4f}" if v2_val is not None else "N/A"

        if v1_val is not None and v2_val is not None:
            delta = v2_val - v1_val
            delta_str = f"{delta:+.4f}"
            status = "✓ V2 melhor" if delta > 0.001 else ("✓ V1 melhor" if delta < -0.001 else "≈ Igual")
        else:
            delta_str = "N/A"
            status = "N/A"

        logger.info(f"{metric_name:<30} | {v1_str:>12} | {v2_str:>12} | {delta_str:>12} | {status:>10}")

    # Per-class comparison
    logger.info("\nMétricas por Classe (F1):")
    logger.info(f"{'Classe':<15} | {'V1':>12} | {'V2':>12} | {'Delta':>12}")
    logger.info("-" * 60)

    for cls in CLASSES:
        v1_f1 = metrics_v1.get('f1_per_class', {}).get(cls) if metrics_v1 else None
        v2_f1 = metrics_v2.get('f1_per_class', {}).get(cls) if metrics_v2 else None

        v1_str = f"{v1_f1:.4f}" if v1_f1 is not None else "N/A"
        v2_str = f"{v2_f1:.4f}" if v2_f1 is not None else "N/A"

        if v1_f1 is not None and v2_f1 is not None:
            delta_str = f"{v2_f1 - v1_f1:+.4f}"
        else:
            delta_str = "N/A"

        logger.info(f"{cls:<15} | {v1_str:>12} | {v2_str:>12} | {delta_str:>12}")


def compare_features(features_dir_v1: str, features_dir_v2: str) -> None:
    """Compara características dos arrays extraídos."""

    logger.info("\n" + "="*80)
    logger.info("COMPARAÇÃO DE FEATURES")
    logger.info("="*80 + "\n")

    # Tentar carregar V1
    try:
        X_v1 = np.load(os.path.join(features_dir_v1, 'X_treino.npy'))
        y_v1 = np.load(os.path.join(features_dir_v1, 'y_treino.npy'))
        logger.info(f"✓ V1 Features carregadas: {X_v1.shape}")
    except Exception as e:
        logger.warning(f"✗ Não foi possível carregar V1: {e}")
        X_v1 = None
        y_v1 = None

    # Tentar carregar V2
    try:
        X_v2 = np.load(os.path.join(features_dir_v2, 'X_treino.npy'))
        y_v2 = np.load(os.path.join(features_dir_v2, 'y_treino.npy'))
        logger.info(f"✓ V2 Features carregadas: {X_v2.shape}")
    except Exception as e:
        logger.warning(f"✗ Não foi possível carregar V2: {e}")
        X_v2 = None
        y_v2 = None

    if X_v1 is None and X_v2 is None:
        logger.error("Não foi possível carregar features para comparação!")
        return

    # Comparar shapes
    logger.info("\nShapes:")
    logger.info(f"  V1: {X_v1.shape if X_v1 is not None else 'N/A'}")
    logger.info(f"  V2: {X_v2.shape if X_v2 is not None else 'N/A'}")

    if X_v1 is not None and X_v2 is not None:
        if X_v1.shape == X_v2.shape:
            logger.info("  ✓ Shapes idênticos")
        else:
            logger.warning("  ✗ Shapes diferentes!")

    # Comparar distribuição
    logger.info("\nDistribuição (primeiras 100 features):")
    logger.info(f"{'Aspecto':<20} | {'V1':>20} | {'V2':>20}")
    logger.info("-" * 65)

    if X_v1 is not None:
        v1_mean = X_v1[:, :100].mean()
        v1_std = X_v1[:, :100].std()
        v1_min = X_v1[:, :100].min()
        v1_max = X_v1[:, :100].max()

        logger.info(f"{'Mean':<20} | {v1_mean:>20.6f} | ", end="")
    else:
        logger.info(f"{'Mean':<20} | {'N/A':>20} | ", end="")

    if X_v2 is not None:
        v2_mean = X_v2[:, :100].mean()
        v2_std = X_v2[:, :100].std()
        v2_min = X_v2[:, :100].min()
        v2_max = X_v2[:, :100].max()

        logger.info(f"{v2_mean:>20.6f}")
    else:
        logger.info(f"{'N/A':>20}")

    if X_v1 is not None and X_v2 is not None:
        logger.info(f"{'Std':<20} | {v1_std:>20.6f} | {v2_std:>20.6f}")
        logger.info(f"{'Min':<20} | {v1_min:>20.6f} | {v2_min:>20.6f}")
        logger.info(f"{'Max':<20} | {v1_max:>20.6f} | {v2_max:>20.6f}")

    # Verificar validade
    logger.info("\nValidade de Dados:")
    if X_v1 is not None:
        v1_valid = np.all(np.isfinite(X_v1))
        logger.info(f"  V1 (sem NaN/Inf): {'✓ Sim' if v1_valid else '✗ NÃO'}")

    if X_v2 is not None:
        v2_valid = np.all(np.isfinite(X_v2))
        logger.info(f"  V2 (sem NaN/Inf): {'✓ Sim' if v2_valid else '✗ NÃO'}")


def run_comparison(skip_v1: bool = False, skip_v2: bool = False) -> None:
    """Executa pipeline de comparação."""

    logger.info("="*80)
    logger.info("Ferramenta de Comparação: V1 vs V2")
    logger.info("="*80 + "\n")

    # Paths
    extractor_v1 = os.path.join(_HERE, 'extrator.py')
    extractor_v2 = os.path.join(_HERE, 'extrator_v2.py')
    trainer_v1 = os.path.join(_HERE, 'train_xgboost.py')
    trainer_v2 = os.path.join(_HERE, 'train_xgboost_v2.py')

    features_dir_v1 = os.path.join(_HERE, 'features_extraidas')
    features_dir_v2 = os.path.join(_HERE, 'features_extraidas_v2')
    metrics_dir_v1 = os.path.join(_HERE, 'metrics')
    metrics_dir_v2 = os.path.join(_HERE, 'metrics')  # mesmo dir, different json names

    # -------- Etapa 1: Extração --------
    if not skip_v1:
        logger.info("ETAPA 1a: Extraindo features com V1...")
        if os.path.exists(extractor_v1):
            ret = os.system(f"cd {_HERE} && python extrator.py")
            if ret == 0:
                logger.info("✓ V1 extraction completed\n")
            else:
                logger.warning("✗ V1 extraction failed\n")
        else:
            logger.warning(f"✗ {extractor_v1} não encontrado\n")

    if not skip_v2:
        logger.info("ETAPA 1b: Extraindo features com V2...")
        if os.path.exists(extractor_v2):
            ret = os.system(f"cd {_HERE} && python extrator_v2.py")
            if ret == 0:
                logger.info("✓ V2 extraction completed\n")
            else:
                logger.warning("✗ V2 extraction failed\n")
        else:
            logger.warning(f"✗ {extractor_v2} não encontrado\n")

    # -------- Etapa 2: Treino --------
    logger.info("ETAPA 2a: Treinando XGBoost com V1 features...")
    if os.path.exists(trainer_v1):
        ret = os.system(f"cd {_HERE} && python train_xgboost.py > /dev/null 2>&1")
        if ret == 0:
            logger.info("✓ V1 training completed\n")
        else:
            logger.warning("✗ V1 training failed\n")
    else:
        logger.warning(f"✗ {trainer_v1} não encontrado\n")

    logger.info("ETAPA 2b: Treinando XGBoost com V2 features...")
    if os.path.exists(trainer_v2):
        ret = os.system(f"cd {_HERE} && python train_xgboost_v2.py > /dev/null 2>&1")
        if ret == 0:
            logger.info("✓ V2 training completed\n")
        else:
            logger.warning("✗ V2 training failed\n")
    else:
        logger.warning(f"✗ {trainer_v2} não encontrado\n")

    # -------- Etapa 3: Comparação --------
    logger.info("ETAPA 3: Analisando resultados...\n")

    # Comparar features
    if os.path.exists(features_dir_v1) and os.path.exists(features_dir_v2):
        compare_features(features_dir_v1, features_dir_v2)
    else:
        logger.warning("Diretórios de features não encontrados para comparação")

    # Comparar métricas
    metrics_v1 = get_latest_metrics(metrics_dir_v1)
    metrics_v2 = get_latest_metrics(metrics_dir_v2)

    if metrics_v1 or metrics_v2:
        compare_metrics(metrics_v1, metrics_v2)
    else:
        logger.warning("Nenhuma métrica encontrada para comparação")

    # -------- Recomendação Final --------
    logger.info("\n" + "="*80)
    logger.info("RECOMENDAÇÃO")
    logger.info("="*80 + "\n")

    if metrics_v1 and metrics_v2:
        f1_v1 = metrics_v1.get('f1_macro', 0)
        f1_v2 = metrics_v2.get('f1_macro', 0)

        if f1_v2 > f1_v1 + 0.01:
            logger.info("✓ V2 tem desempenho significativamente melhor (+1%+ em F1)")
            logger.info("  Recomendação: MIGRAR para V2")
        elif f1_v2 > f1_v1 - 0.01:
            logger.info("≈ V2 tem desempenho similar a V1")
            logger.info("  Recomendação: MIGRAR por melhor robustez (mesmo F1)")
        else:
            logger.info("✗ V2 tem desempenho pior que V1")
            logger.info("  Recomendação: INVESTIGAR problema em V2")
    else:
        logger.info("⚠ Impossível fazer recomendação (métricas incompletas)")

    logger.info("\nPróximos passos:")
    logger.info("  1. Review logs completos em extrator_v2.log e train_xgboost_v2.log")
    logger.info("  2. Se V2 > V1: cp extrator_v2.py extrator.py")
    logger.info("  3. Se V2 > V1: cp train_xgboost_v2.py train_xgboost.py")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compara performance V1 vs V2 do Hybrid XGBoost'
    )
    parser.add_argument(
        '--skip-v1',
        action='store_true',
        help='Pular extração V1 (usar features já existentes)'
    )
    parser.add_argument(
        '--skip-v2',
        action='store_true',
        help='Pular extração V2 (usar features já existentes)'
    )

    args = parser.parse_args()

    run_comparison(skip_v1=args.skip_v1, skip_v2=args.skip_v2)


if __name__ == '__main__':
    main()
