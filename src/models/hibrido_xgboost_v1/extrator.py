"""
Extrator de Features via Backbone EfficientNet (V7).
Este script converte as imagens do banco SQLite em arrays estruturados do NumPy 
completamente desconectados para treinamento estático (Tabular) rápido em XGBoost.
"""

import os
import sys
import time
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

# -- Injeção segura do ambiente V7 ----------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(_HERE)
V7_DIR = os.path.join(MODELS_DIR, 'efficientnet_v7')
ROOT_DIR = os.path.normpath(os.path.join(MODELS_DIR, '..', '..'))

sys.path.append(V7_DIR)
try:
    from train import (
        EfficientNetTemporalV6, carregar_dados, TemporalCulturaDataset,
        CLASSES, BATCH_SIZE, NUM_WORKERS, DEVICE, USE_AMP, MAX_SEQ_LEN
    )
except ImportError as e:
    raise ImportError(f"Falha ao importar do train.py da V7: {e}")

# -- Configurações -------------------------------------------------------------
DB_TREINO  = os.path.join(ROOT_DIR, 'sample_treino_max9000.db')
DB_TESTE   = os.path.join(ROOT_DIR, 'sample_teste_250.db')
PESOS_PATH = os.path.join(V7_DIR, 'artifacts', 'pesos.pt')

# Diretório base deste estágio
OUT_DIR = os.path.join(_HERE, 'features_extraidas')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def extrair_salvar_features(db_path, prefixo, modelo):
    """Lê os dados temporalizados, passa pelo backbone e salva o tensor tabulado."""
    logging.info(f"[{prefixo}] Iniciando extração do banco: {db_path}")

    registros, labels, meses = carregar_dados(db_path)
    if not registros:
        logging.error(f"[{prefixo}] Nenhum dado encontrado em {db_path}")
        return

    dataset = TemporalCulturaDataset(registros, labels, meses)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

    modelo.eval()
    
    # Armazenamento em RAM (Como extraímos apenas arrays, caberá tranquilamente na RAM antes de salvar)
    X_todas_features = []
    y_todas_labels = []

    total_amostras = 0
    t_start = time.time()

    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=USE_AMP):
        for imagens, dias, mes, mask, batch_labels in loader:
            # imagens: [B, T=3, C=3, H, W]
            B, T = imagens.shape[0], imagens.shape[1]
            
            imagens = imagens.to(DEVICE, non_blocking=True)
            
            # Formatar T para que o backbone ignore a dimensão sequence batched.
            imgs_flat = imagens.reshape(B * T, *imagens.shape[2:])  # (B*T, C, H, W)
            
            # Passa apenas pelo EffNet pretreinado sem a cabeça classificadora
            features_flat = modelo.backbone(imgs_flat)              # (B*T, 1280)
            
            # Reshape novamente para a sequência:
            features = features_flat.reshape(B, T, -1).cpu().numpy() # (B, 3, 1280)
            
            # Precisamos transformar features(B, 3, 1280) para Features(B, 3840) estirando horizontalmente.
            features_1d = features.reshape(B, -1)                    # (B, 3840)
            
            # Formatar dias (B, 3) 
            dias_np = dias.numpy()
            
            # Formatar mes original (+1 devido ao emb do dataset que reduziu p/ 0) -> (B, 1)
            mes_np = (mes.numpy() + 1).reshape(B, 1)
            
            # Formatar count temporal baseado na máscara -> (B, 1)
            total_imagens_validas = mask.sum(dim=1).numpy().reshape(B, 1)

            # Tabular TUDO
            # Array Final esperado: [Feature1(1280), Feature2(1280), Feature3(1280), Dia1, Dia2, Dia3, Mês, Count_img]
            # Total colunas = 3840 + 3 + 1 + 1 = 3845
            batch_tabular = np.concatenate([
                features_1d, 
                dias_np,
                mes_np,
                total_imagens_validas
            ], axis=1)

            X_todas_features.append(batch_tabular)
            y_todas_labels.append(batch_labels.numpy())
            
            total_amostras += B
            if total_amostras % 1000 == 0:
                logging.info(f"[{prefixo}] Processados {total_amostras} talhões.")

    X_completo = np.vstack(X_todas_features)
    y_completo = np.concatenate(y_todas_labels)

    t_total = time.time() - t_start
    logging.info(f"[{prefixo}] Finalizado! -> {X_completo.shape} extraídas em {t_total:.2f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_x = os.path.join(OUT_DIR, f"X_{prefixo}.npy")
    out_y = os.path.join(OUT_DIR, f"y_{prefixo}.npy")
    
    np.save(out_x, X_completo)
    np.save(out_y, y_completo)
    logging.info(f"[{prefixo}] Salvo matrizes no disco: {out_x}")

def main():
    logging.info("--- Extrator de Conhecimento DeepLearning Tabular ---")
    
    # 1. Instanciar V7 e injetar a inteligência (Pesos do Finetune das Culturas)
    logging.info(f"Instanciando Backbone V7 e injetando Pesos.")
    modelo = EfficientNetTemporalV6(len(CLASSES)).to(DEVICE)
    
    if os.path.exists(PESOS_PATH):
        try:
            modelo.load_state_dict(torch.load(PESOS_PATH, map_location=DEVICE))
            logging.info("Pesos V7 carregados com sucesso! O extractor usará features vegetalizadas.")
        except Exception as e:
            logging.warning(f"Erro lendo pesos customizados, usará ImageNet base. Erro: {e}")
    else:
        logging.warning("Pesos V7 não detectados, usará os puristas do ImageNet.")

    # 2. Executar extração massiva no TREINO (demorado)
    extrair_salvar_features(DB_TREINO, "treino", modelo)

    # 3. Executar extração massiva no TESTE (rápido)
    extrair_salvar_features(DB_TESTE, "teste", modelo)
    
    logging.info("Mapeamento concluído e salvo no disco! Pronto para o XGBoost.")

if __name__ == '__main__':
    main()
