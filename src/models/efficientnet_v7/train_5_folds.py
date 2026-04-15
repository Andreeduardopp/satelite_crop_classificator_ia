"""
Script helper para executar o treinamento do modelo EfficientNet v7 5 vezes, 
utilizando um dataset gerado distinto a cada execução.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'src', 'models', 'efficientnet_v7', 'train.py')
DIR_TREINO = os.path.join(BASE_DIR, 'datasets', 'dataset_treino')
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'src', 'models', 'efficientnet_v7', 'artifacts_5_folds')

# Pode ser alterado se foram gerados mais ou menos datasets
N_FOLDS = 5

def main() -> None:
    logging.info(f"Iniciando treinamento em lote para {N_FOLDS} folds/datasets...")
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    for i in range(1, N_FOLDS + 1):
        db_nome = f"sample_treino_max9000_v{i}.db"
        db_path = os.path.join(DIR_TREINO, db_nome)
        
        # Diretório de saída único para essa iteração (guarda pesos .pt e metrics json)
        out_dir = os.path.join(OUTPUT_BASE_DIR, f"fold_{i}")
        
        if not os.path.exists(db_path):
            logging.error(f"Dataset não encontrado: {db_path}. Verifique se você rodou gerar_sample_treino.py.")
            return
            
        logging.info(f"==================================================")
        logging.info(f"=== Rodando Treinamento: Fold {i} / {N_FOLDS} ===")
        logging.info(f"   Database: {db_path}")
        logging.info(f"   Output  : {out_dir}")
        logging.info(f"==================================================")
        
        command = [
            sys.executable, TRAIN_SCRIPT,
            "--db_path", db_path,
            "--out_dir", out_dir
        ]
        
        # Chama a execução externa do train.py
        result = subprocess.run(command)
        if result.returncode != 0:
            logging.error(f"Treinamento do fold {i} falhou com código de saída {result.returncode}!")
        else:
            logging.info(f"Fold {i} finalizado com sucesso!")
        
    logging.info("Ciclo de treinamentos em lote concluído!")

if __name__ == '__main__':
    main()
