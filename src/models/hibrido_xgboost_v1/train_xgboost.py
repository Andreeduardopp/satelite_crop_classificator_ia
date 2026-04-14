"""
Etapa Tabular Clássica: XGBClassifier da arquitetura Two-Stage.
Toma ingestão dos pesados Arrays numpy extraídos da Visão Computacional (V7)
e constrói a árvore de decisão matemática em segundos ao invés de horas.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
import numpy as np

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# -- Constantes Espelhadas do V7 -----------------------------------------------
CLASSES = ['soja', 'milho', 'trigo', 'aveia', 'feijão']

# -- Paths ---------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(_HERE, 'features_extraidas')
MODEL_OUT_DIR = os.path.join(_HERE, 'artifacts')
METRICS_OUT_DIR = os.path.join(_HERE, 'metrics')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    logging.info("=== Híbrido XGBoost V1: Compilando DeepLearning Extracted Arrays ===")

    # 1. Checar se as planilhas extraídas existem
    f_X_tr = os.path.join(FEATURES_DIR, 'X_treino.npy')
    f_y_tr = os.path.join(FEATURES_DIR, 'y_treino.npy')
    f_X_te = os.path.join(FEATURES_DIR, 'X_teste.npy')
    f_y_te = os.path.join(FEATURES_DIR, 'y_teste.npy')

    if not all(os.path.exists(p) for p in [f_X_tr, f_y_tr, f_X_te, f_y_te]):
        logging.error("Arquivos numpy não encontrados. Você precisa rodar `extrator.py` antes!")
        return

    # 2. Carregar na memória RAM (Muito veloz por ser estrutura pura de floats)
    logging.info("Carregando bases Numpy na RAM...")
    t0 = time.time()
    X_train = np.load(f_X_tr)
    y_train = np.load(f_y_tr)
    X_test  = np.load(f_X_te)
    y_test  = np.load(f_y_te)
    logging.info(f"Dados Carregados em {time.time()-t0:.2f}s!")
    logging.info(f" -> Matriz Treino: {X_train.shape} linhas/features")
    logging.info(f" -> Matriz Teste:  {X_test.shape} linhas/features")

    # 3. Tratar desequilíbrio (Compute Class Weights tal como o PyTorch CrossEntropyLoss)
    pesos = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=y_train)
    # XGBoost usa vetor de pesos diretos sample-to-sample nos dados (sample_weights):
    sample_weights = np.zeros_like(y_train, dtype=float)
    for c_id in range(len(CLASSES)):
        sample_weights[y_train == c_id] = pesos[c_id]

    # 4. Configurar Model Params (Base)
    # Parâmetros agressivos para devorar as ~4000 colunas extraídas pela convolucional
    xgb_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(CLASSES),
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,        # Número de árvores
        'n_jobs': -1,               # Liberar todas as threads da CPU
        'use_label_encoder': False,
        'tree_method': 'hist',      # otimizadíssimo e consome pouca ram
        'random_state': 42
    }

    modelo = xgb.XGBClassifier(**xgb_params)

    # 5. Treinar
    logging.info("Iniciando treinamento Tabular do XGBoost. Aguarde (leva segundos)...")
    t_train = time.time()
    modelo.fit(
        X_train, y_train, 
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=10  # Plota progresso a cada 10 epochs
    )
    logging.info(f"Treinamento Tabular Finalizado em {time.time() - t_train:.2f}s!")

    # 6. Salvar Molde da Árvore Instantaneamente
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    peso_arvore = os.path.join(MODEL_OUT_DIR, 'xgboost_modelo.json')
    modelo.save_model(peso_arvore)
    logging.info(f"Exportado artefato final para: {peso_arvore}")

    # 7. Avaliação Final e Inferência Limpa
    logging.info("Executando inferência final sobre Validacão/Holdout...")
    t_inf = time.time()
    y_pred = modelo.predict(X_test)
    tempo_medio = ((time.time() - t_inf) / len(X_test)) * 1000

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    logging.info(f"=== RESULTADO HÍBRIDO (ExtratorDL -> XGBoost) ===")
    logging.info(f"Acurácia: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    
    for cls_name, f1_cls in zip(CLASSES, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_test, y_pred, target_names=CLASSES, zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    logging.info("Confusion Matrix:")
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    logging.info(header)
    for cls, row in zip(CLASSES, cm):
        logging.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    logging.info(f"Tempo médio de inferência pura XGBoost: {tempo_medio:.3f} ms/talhão")

    # 8. Extrair Métricas para JSON histórico
    os.makedirs(METRICS_OUT_DIR, exist_ok=True)
    m_path = os.path.join(METRICS_OUT_DIR, f"xgb_metrics_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(m_path, "w") as f:
        json.dump({
            "f1_macro": float(f1_macro),
            "accuracy": float(acc),
            "f1_per_class": {c: float(f) for c, f in zip(CLASSES, f1_per_class)}
        }, f, indent=4)

if __name__ == "__main__":
    main()
