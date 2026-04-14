"""
Script de inferência para testar o EfficientNet V7 no banco de teste isolado (sample_teste_200.db).

Uso:
    python src/models/efficientnet_v7/test/test_model.py
"""

import os
import sys
import time
import logging
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# -- Paths ---------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
V7_DIR = os.path.normpath(os.path.join(_HERE, '..'))
ROOT_DIR = os.path.normpath(os.path.join(V7_DIR, '..', '..', '..'))

# Injetar V7_DIR no PATH para podermos importar componentes do train.py
sys.path.append(V7_DIR)

# Agora importamos o que precisamos do train.py com segurança
try:
    import train
    from train import (
        EfficientNetTemporalV6, carregar_dados, TemporalCulturaDataset,
        CLASSES, BATCH_SIZE, NUM_WORKERS, DEVICE, USE_AMP
    )
except ImportError as e:
    raise ImportError(f"Falha ao importar módulos do train.py. Certifique-se de executar da raiz. Erro: {e}")

# -- Configurações -------------------------------------------------------------
DB_TESTE = os.path.join(ROOT_DIR, 'sample_teste_250.db')
PESOS_PATH = os.path.join(V7_DIR, 'artifacts', 'pesos.pt')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ],
)

def main():
    logging.info("=== Teste Isolado EfficientNet V7 ===")
    
    if not os.path.exists(PESOS_PATH):
        logging.error(f"Arquivo de pesos não encontrado: {PESOS_PATH}")
        logging.error("Você precisa concluir o treinamento (`train.py`) primeiro para usar a rede!")
        return

    # 1. Carregar dados de teste
    logging.info(f"Carregando dados de teste de: {DB_TESTE}")
    if not os.path.exists(DB_TESTE):
        logging.error(f"Banco de teste {DB_TESTE} não encontrado!")
        return
        
    registros, labels, meses = carregar_dados(DB_TESTE)
    total_imgs = sum(len(r) for r in registros)
    logging.info(f"Total Teste: {len(registros)} talhões, {total_imgs} imagens")

    if not registros:
        logging.error("Nenhum talhão válido para teste.")
        return
        
    for i, c in enumerate(CLASSES):
        logging.info(f"  {c}: {labels.count(i)} talhões")

    # 2. Criar Dataset e DataLoader
    ds_teste = TemporalCulturaDataset(registros, labels, meses)
    use_pin = DEVICE.type == 'cuda'
    loader_teste = DataLoader(ds_teste, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=use_pin)

    # 3. Construir e inicializar modelo com pesos
    logging.info("Carregando arquitetura e pesos do modelo...")
    modelo = EfficientNetTemporalV6(len(CLASSES)).to(DEVICE)
    try:
        modelo.load_state_dict(torch.load(PESOS_PATH, map_location=DEVICE))
        logging.info("Pesos carregados com sucesso!")
    except Exception as e:
        logging.error(f"Erro ao carregar os pesos: {e}")
        return

    # 4. Avaliação (Inferência)
    modelo.eval()
    y_true, y_pred = [], []
    n_samples = 0
    t_start = time.perf_counter()

    logging.info("Iniciando inferência silenciosamente aguarde...")
    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=USE_AMP):
        for images, dias, mes, mask, b_labels in loader_teste:
            images = images.to(DEVICE, non_blocking=True)
            dias   = dias.to(DEVICE, non_blocking=True)
            mes    = mes.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)

            logits = modelo(images, dias, mes, mask)
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(b_labels.numpy())
            n_samples += b_labels.size(0)

    t_total = time.perf_counter() - t_start

    # 5. Métricas e Output final
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    logging.info(f"RESULTADO FINAL - Acurácia: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    
    logging.info("---")
    for cls_name, f1_cls in zip(CLASSES, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")
    logging.info("---")
    
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
    logging.info(f"Tempo médio de inferência puro: {tempo_medio:.2f} ms/talhão ({n_samples} talhões)")

if __name__ == "__main__":
    main()
