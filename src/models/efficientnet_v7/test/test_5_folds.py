"""
Script de inferência para testar o EfficientNet V7 nos 5 folds.

Uso:
    python src/models/efficientnet_v7/test/test_5_folds.py
"""

import os
import sys
import time
import json
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

try:
    import train
    from train import (
        EfficientNetTemporalV6, carregar_dados, TemporalCulturaDataset,
        CLASSES, BATCH_SIZE, NUM_WORKERS, DEVICE, USE_AMP
    )
except ImportError as e:
    raise ImportError(f"Falha ao importar módulos do train.py. Certifique-se de executar da raiz. Erro: {e}")

# -- Configurações -------------------------------------------------------------
N_FOLDS = 5
DIR_TESTE = os.path.join(ROOT_DIR, 'datasets', 'dataset_teste')
ARTIFACTS_DIR = os.path.join(V7_DIR, 'artifacts_5_folds')
METRICS_DIR = os.path.join(V7_DIR, 'metrics')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ],
)

def avaliar_fold(fold_id):
    db_path = os.path.join(DIR_TESTE, f'sample_teste_250_v{fold_id}.db')
    pesos_path = os.path.join(ARTIFACTS_DIR, f'fold_{fold_id}', 'pesos.pt')

    logging.info(f"=== Avaliando Fold {fold_id} ===")
    
    if not os.path.exists(pesos_path):
        logging.error(f"Arquivo de pesos não encontrado para o fold {fold_id}: {pesos_path}")
        return None
    if not os.path.exists(db_path):
        logging.error(f"Banco de teste {db_path} não encontrado para o fold {fold_id}!")
        return None

    registros, labels, meses = carregar_dados(db_path)
    if not registros:
        logging.error(f"Nenhum talhão válido para teste no fold {fold_id}.")
        return None

    ds_teste = TemporalCulturaDataset(registros, labels, meses)
    use_pin = DEVICE.type == 'cuda'
    loader_teste = DataLoader(ds_teste, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=use_pin)

    modelo = EfficientNetTemporalV6(len(CLASSES)).to(DEVICE)
    try:
        modelo.load_state_dict(torch.load(pesos_path, map_location=DEVICE))
    except Exception as e:
        logging.error(f"Erro ao carregar os pesos do fold {fold_id}: {e}")
        return None

    modelo.eval()
    y_true, y_pred = [], []
    n_samples = 0

    t_start = time.perf_counter()
    cpu_start = time.process_time()
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

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
    cpu_total = time.process_time() - cpu_start
    gpu_mem_peak = 0
    if DEVICE.type == 'cuda':
        gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

    tempo_medio_ms = (t_total / n_samples) * 1000
    cpu_medio_ms = (cpu_total / n_samples) * 1000

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    logging.info(f"Fold {fold_id} -> Acurácia: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    logging.info(f"  Perf -> Tempo/amostra: {tempo_medio_ms:.2f}ms | CPU/amostra: {cpu_medio_ms:.2f}ms | GPU Pico: {gpu_mem_peak:.2f}MB")
    
    return {
        'acc': acc,
        'f1_macro': f1_macro,
        'tempo_medio_ms': tempo_medio_ms,
        'cpu_medio_ms': cpu_medio_ms,
        'gpu_mem_peak_mb': gpu_mem_peak
    }

def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    resultados_folds = {}
    metricas = {'acc': [], 'f1': [], 'tempo': [], 'cpu': [], 'gpu': []}

    for idx in range(1, N_FOLDS + 1):
        res = avaliar_fold(idx)
        if res:
            resultados_folds[f"fold_{idx}"] = res
            metricas['acc'].append(res['acc'])
            metricas['f1'].append(res['f1_macro'])
            metricas['tempo'].append(res['tempo_medio_ms'])
            metricas['cpu'].append(res['cpu_medio_ms'])
            metricas['gpu'].append(res['gpu_mem_peak_mb'])
            
    if not metricas['acc']:
        logging.error("Nenhum fold foi avaliado com sucesso.")
        return

    acc_media = np.mean(metricas['acc'])
    acc_std = np.std(metricas['acc'])
    f1_media = np.mean(metricas['f1'])
    f1_std = np.std(metricas['f1'])
    tempo_medio_geral = np.mean(metricas['tempo'])
    cpu_medio_geral = np.mean(metricas['cpu'])
    gpu_medio_geral = np.mean(metricas['gpu'])

    logging.info("=======================================")
    logging.info("RESUMO FINAL - 5-FOLD CROSS VALIDATION")
    logging.info("=======================================")
    for fold, mets in resultados_folds.items():
        logging.info(f"  {fold.capitalize()}: Acc={mets['acc']:.4f}, F1-Macro={mets['f1_macro']:.4f}, Tempo={mets['tempo_medio_ms']:.2f}ms, GPU={mets['gpu_mem_peak_mb']:.2f}MB")
    logging.info("---------------------------------------")
    logging.info(f"Média Acurácia: {acc_media:.4f} (± {acc_std:.4f})")
    logging.info(f"Média F1-Macro: {f1_media:.4f} (± {f1_std:.4f})")
    logging.info(f"Média Tempo/Amostra: {tempo_medio_geral:.2f} ms")
    logging.info(f"Média CPU/Amostra: {cpu_medio_geral:.2f} ms")
    logging.info(f"Média GPU Pico: {gpu_medio_geral:.2f} MB")
    logging.info("=======================================")

    resultados_finais = {
        'folds': resultados_folds,
        'media_acuracia': float(acc_media),
        'std_acuracia': float(acc_std),
        'media_f1_macro': float(f1_media),
        'std_f1_macro': float(f1_std),
        'media_tempo_ms_amostra': float(tempo_medio_geral),
        'media_cpu_ms_amostra': float(cpu_medio_geral),
        'media_gpu_peak_mb': float(gpu_medio_geral)
    }

    results_file = os.path.join(METRICS_DIR, 'test_5_folds_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(resultados_finais, f, indent=4)
        
    logging.info(f"Resultados salvos em: {results_file}")

if __name__ == "__main__":
    main()
