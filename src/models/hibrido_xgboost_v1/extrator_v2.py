"""
Extrator de Features via Backbone EfficientNet (V7) - Versão Otimizada.
Este script converte as imagens do banco SQLite em arrays estruturados do NumPy
completamente desconectados para treinamento estático (Tabular) rápido em XGBoost.

Melhorias na V2:
- Validação rigorosa de shapes e valores (NaN/Inf)
- Otimização de operações de reshape (uma única transferência CPU)
- Configuração centralizada de hiperparâmetros
- Logging detalhado de estatísticas de features
- Feature standardization opcional
- Tratamento robusto de erros
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import stats

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

# -- Configurações Centralizadas -----------------------------------------------
@dataclass
class ExtractorConfig:
    """Configuração unificada do extrator."""
    # Dimensões do backbone
    BACKBONE_FEATURE_DIM: int = 1280
    TEMPORAL_STEPS: int = 3

    # Cálculos derivados (não modificar manualmente)
    @property
    def FLATTENED_FEATURES(self) -> int:
        """Total de features após achatar sequência temporal."""
        return self.BACKBONE_FEATURE_DIM * self.TEMPORAL_STEPS

    @property
    def TOTAL_TABULAR_COLS(self) -> int:
        """Total de colunas na saída final (features + metadata)."""
        # 3840 (features) + 3 (dias) + 1 (mês) + 1 (count_imagens)
        return self.FLATTENED_FEATURES + self.TEMPORAL_STEPS + 2

    # Diretórios
    DB_TREINO: str = os.path.join(ROOT_DIR, 'sample_treino_max9000.db')
    DB_TESTE: str = os.path.join(ROOT_DIR, 'sample_teste_250.db')
    PESOS_PATH: str = os.path.join(V7_DIR, 'artifacts', 'pesos.pt')
    OUT_DIR: str = os.path.join(_HERE, 'features_extraidas_v2')

    # Opções de processamento
    STANDARDIZE_FEATURES: bool = True  # Normalizar features após extração
    COMPUTE_STATISTICS: bool = True    # Computar e logar estatísticas
    VALIDATE_SHAPES: bool = True       # Validar shapes em cada estágio

CONFIG = ExtractorConfig()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -- Funções de Validação e Processamento -------------------------------------

def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str
) -> None:
    """Valida shape de tensor com mensagem descritiva."""
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch em '{name}': esperado {expected_shape}, "
            f"obteve {tensor.shape}"
        )


def validate_feature_values(
    features: np.ndarray,
    name: str,
    allow_negative: bool = True
) -> None:
    """Valida se há NaN, Inf ou valores inválidos em features."""
    if not np.all(np.isfinite(features)):
        n_invalid = (~np.isfinite(features)).sum()
        raise ValueError(
            f"Detectados {n_invalid} valores inválidos (NaN/Inf) em {name}"
        )

    if not allow_negative and np.any(features < 0):
        n_negative = (features < 0).sum()
        logger.warning(
            f"Detectados {n_negative} valores negativos em {name} "
            "(backbone features não devem ser negativos)"
        )


def compute_feature_statistics(features: np.ndarray, name: str) -> None:
    """Computa e loga estatísticas detalhadas de features."""
    if not CONFIG.COMPUTE_STATISTICS:
        return

    logger.info(f"Estatísticas de {name}:")
    logger.info(f"  Shape: {features.shape}")
    logger.info(f"  Mean: {features.mean():.6f} ± {features.std():.6f}")
    logger.info(f"  Min: {features.min():.6f}")
    logger.info(f"  Max: {features.max():.6f}")
    logger.info(f"  Percentis [25%, 50%, 75%]: "
                f"{np.percentile(features, 25):.6f}, "
                f"{np.percentile(features, 50):.6f}, "
                f"{np.percentile(features, 75):.6f}")


def standardize_features(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Padroniza features para média=0 e desvio=1.
    Retorna: (features_padronizadas, mean, std)
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)

    # Evitar divisão por zero
    std = np.where(std == 0, 1.0, std)

    X_standardized = (X - mean) / std
    return X_standardized, mean, std


def assemble_tabular_batch(
    features_1d: np.ndarray,
    dias_np: np.ndarray,
    mes_np: np.ndarray,
    total_imagens_validas: np.ndarray,
    batch_size: int
) -> np.ndarray:
    """
    Monta batch tabular com validações.

    Shape esperado:
    - features_1d: (B, 3840)
    - dias_np: (B, 3)
    - mes_np: (B, 1)
    - total_imagens_validas: (B, 1)

    Output: (B, 3845)
    """
    if CONFIG.VALIDATE_SHAPES:
        expected_feature_cols = CONFIG.FLATTENED_FEATURES

        assert features_1d.shape == (batch_size, expected_feature_cols), \
            f"Features shape mismatch: {features_1d.shape} != {(batch_size, expected_feature_cols)}"
        assert dias_np.shape == (batch_size, 3), \
            f"Dias shape mismatch: {dias_np.shape}"
        assert mes_np.shape == (batch_size, 1), \
            f"Mes shape mismatch: {mes_np.shape}"
        assert total_imagens_validas.shape == (batch_size, 1), \
            f"Total imagens shape mismatch: {total_imagens_validas.shape}"

    # Concatenar ao longo do eixo 1 (features)
    batch_tabular = np.concatenate([
        features_1d,
        dias_np,
        mes_np,
        total_imagens_validas
    ], axis=1)

    # Validar output
    if CONFIG.VALIDATE_SHAPES:
        expected_cols = CONFIG.TOTAL_TABULAR_COLS
        assert batch_tabular.shape[1] == expected_cols, \
            f"Output shape mismatch: {batch_tabular.shape[1]} != {expected_cols}"

    validate_feature_values(batch_tabular, "batch_tabular")
    return batch_tabular


# -- Extração Principal -------------------------------------------------------

def extrair_salvar_features(db_path: str, prefixo: str, modelo) -> None:
    """Lê os dados temporalizados, passa pelo backbone e salva o tensor tabulado."""
    logger.info(f"[{prefixo}] ========== Iniciando extração ==========")
    logger.info(f"[{prefixo}] Banco de dados: {db_path}")

    if not os.path.exists(db_path):
        logger.error(f"[{prefixo}] Arquivo não encontrado: {db_path}")
        return

    # -------- 1. Carregar dados --------
    try:
        registros, labels, meses = carregar_dados(db_path)
    except Exception as e:
        logger.error(f"[{prefixo}] Erro ao carregar dados: {e}")
        return

    if not registros:
        logger.error(f"[{prefixo}] Nenhum dado encontrado em {db_path}")
        return

    logger.info(f"[{prefixo}] Dados carregados: {len(registros)} registros, "
                f"{len(set(labels))} classes")

    # -------- 2. Criar dataset e dataloader --------
    try:
        dataset = TemporalCulturaDataset(registros, labels, meses)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == 'cuda'),
            drop_last=False  # Manter último batch mesmo se incompleto
        )
    except Exception as e:
        logger.error(f"[{prefixo}] Erro ao criar dataloader: {e}")
        return

    # -------- 3. Extração em loop --------
    modelo.eval()
    X_todas_features = []
    y_todas_labels = []

    total_amostras = 0
    total_batches = len(loader)
    t_start = time.time()

    logger.info(f"[{prefixo}] Processando {total_batches} batches...")

    try:
        with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=USE_AMP):
            for batch_idx, (imagens, dias, mes, mask, batch_labels) in enumerate(loader):
                try:
                    # Dimensões
                    B, T = imagens.shape[0], imagens.shape[1]

                    if CONFIG.VALIDATE_SHAPES:
                        validate_tensor_shape(imagens, (B, T, 3, *imagens.shape[3:]),
                                            f"batch {batch_idx} imagens")

                    # Mover para device
                    imagens = imagens.to(DEVICE, non_blocking=True)

                    # Achatar dimensão temporal para passar pelo backbone
                    imgs_flat = imagens.reshape(B * T, *imagens.shape[2:])  # (B*T, C, H, W)

                    # Extrair features via EfficientNet backbone
                    features_flat = modelo.backbone(imgs_flat)  # (B*T, 1280)

                    if CONFIG.VALIDATE_SHAPES:
                        assert features_flat.shape == (B * T, CONFIG.BACKBONE_FEATURE_DIM), \
                            f"Backbone output shape mismatch: {features_flat.shape}"

                    # Transferir ONCE para CPU e converter para numpy (operação crítica)
                    features_flat_np = features_flat.cpu().numpy()

                    # Validar valores de features
                    validate_feature_values(features_flat_np, f"batch {batch_idx} backbone output")

                    # Reshape para sequência temporal e depois para 1D em uma única operação
                    # (B*T, 1280) -> (B, T, 1280) -> (B, 3840)
                    features_1d = features_flat_np.reshape(B, T, -1).reshape(B, -1)

                    # Processar metadados
                    dias_np = dias.numpy().astype(np.float32)
                    mes_np = (mes.numpy() + 1).reshape(B, 1).astype(np.float32)  # +1 devido embedding
                    total_imagens_validas = mask.sum(dim=1).numpy().reshape(B, 1).astype(np.float32)

                    # Montar batch tabular com validações
                    batch_tabular = assemble_tabular_batch(
                        features_1d,
                        dias_np,
                        mes_np,
                        total_imagens_validas,
                        B
                    )

                    # Salvar batch
                    X_todas_features.append(batch_tabular)
                    y_todas_labels.append(batch_labels.numpy())

                    total_amostras += B

                    # Log de progresso a cada 1000 amostras
                    if total_amostras % 1000 == 0 or batch_idx == total_batches - 1:
                        tempo_decorrido = time.time() - t_start
                        taxa_amostras = total_amostras / tempo_decorrido if tempo_decorrido > 0 else 0
                        logger.info(f"[{prefixo}] Processados {total_amostras} talhões "
                                   f"({taxa_amostras:.0f} amostras/s)")

                except Exception as e:
                    logger.error(f"[{prefixo}] Erro ao processar batch {batch_idx}: {e}")
                    raise

    except Exception as e:
        logger.error(f"[{prefixo}] Erro durante extração: {e}")
        return

    # -------- 4. Consolidar e validar --------
    if not X_todas_features:
        logger.error(f"[{prefixo}] Nenhuma feature foi extraída!")
        return

    try:
        X_completo = np.vstack(X_todas_features)
        y_completo = np.concatenate(y_todas_labels)
    except Exception as e:
        logger.error(f"[{prefixo}] Erro ao consolidar arrays: {e}")
        return

    t_total = time.time() - t_start

    # Validações finais
    if CONFIG.VALIDATE_SHAPES:
        assert X_completo.shape[0] == len(y_completo), \
            f"Mismatch entre X ({X_completo.shape[0]}) e y ({len(y_completo)})"
        assert X_completo.shape[1] == CONFIG.TOTAL_TABULAR_COLS, \
            f"Shape colunas mismatch: {X_completo.shape[1]} != {CONFIG.TOTAL_TABULAR_COLS}"

    validate_feature_values(X_completo, "X_completo")

    logger.info(f"[{prefixo}] Finalizado! {X_completo.shape} extraídas em {t_total:.2f}s")

    # -------- 5. Estatísticas --------
    if CONFIG.COMPUTE_STATISTICS:
        logger.info(f"[{prefixo}] Análise de distribuição de classes:")
        unique, counts = np.unique(y_completo, return_counts=True)
        for cls_id, count in zip(unique, counts):
            pct = 100 * count / len(y_completo)
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"classe_{cls_id}"
            logger.info(f"  {cls_name}: {count} amostras ({pct:.1f}%)")

    # -------- 6. Padronização Opcional --------
    if CONFIG.STANDARDIZE_FEATURES:
        logger.info(f"[{prefixo}] Padronizando features...")
        # Padronizar apenas as colunas de features (primeiras 3840)
        feature_cols = slice(0, CONFIG.FLATTENED_FEATURES)
        X_features = X_completo[:, feature_cols]
        X_features_std, mean_feat, std_feat = standardize_features(X_features)
        X_completo[:, feature_cols] = X_features_std

        # Salvar estatísticas para referência futura
        stats_path = os.path.join(CONFIG.OUT_DIR, f"standardization_{prefixo}.npz")
        np.savez(stats_path, mean=mean_feat, std=std_feat)
        logger.info(f"[{prefixo}] Estatísticas de padronização salvas: {stats_path}")

    compute_feature_statistics(X_completo, f"{prefixo} final")

    # -------- 7. Salvar no disco --------
    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
    out_x = os.path.join(CONFIG.OUT_DIR, f"X_{prefixo}.npy")
    out_y = os.path.join(CONFIG.OUT_DIR, f"y_{prefixo}.npy")

    try:
        np.save(out_x, X_completo)
        np.save(out_y, y_completo)
        logger.info(f"[{prefixo}] Salvo X em: {out_x} ({X_completo.nbytes / 1e6:.1f} MB)")
        logger.info(f"[{prefixo}] Salvo y em: {out_y}")
    except Exception as e:
        logger.error(f"[{prefixo}] Erro ao salvar arquivos: {e}")
        return

    logger.info(f"[{prefixo}] ========== Extração Concluída ==========\n")


# -- Main -------------------------------------------------------------------

def main():
    logger.info("="*60)
    logger.info("Extrator de Conhecimento DeepLearning Tabular - V2")
    logger.info("="*60)

    # Verificar configuração
    logger.info(f"Configurações:")
    logger.info(f"  Standardize: {CONFIG.STANDARDIZE_FEATURES}")
    logger.info(f"  Compute Stats: {CONFIG.COMPUTE_STATISTICS}")
    logger.info(f"  Validate Shapes: {CONFIG.VALIDATE_SHAPES}")
    logger.info(f"  Expected output cols: {CONFIG.TOTAL_TABULAR_COLS}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Output dir: {CONFIG.OUT_DIR}\n")

    # 1. Instanciar e carregar modelo
    logger.info("Instanciando Backbone V7 e carregando pesos...")
    try:
        modelo = EfficientNetTemporalV6(len(CLASSES)).to(DEVICE)

        if os.path.exists(CONFIG.PESOS_PATH):
            try:
                state_dict = torch.load(CONFIG.PESOS_PATH, map_location=DEVICE)
                modelo.load_state_dict(state_dict)
                logger.info("✓ Pesos V7 carregados com sucesso!")
            except Exception as e:
                logger.warning(f"Erro ao carregar pesos customizados: {e}")
                logger.warning("Utilizando pesos padrão do ImageNet")
        else:
            logger.warning(f"Pesos não encontrados em: {CONFIG.PESOS_PATH}")
            logger.warning("Utilizando pesos padrão do ImageNet")
    except Exception as e:
        logger.error(f"Erro ao instanciar modelo: {e}")
        return

    # 2. Executar extração
    try:
        logger.info("\n" + "="*60)
        logger.info("ETAPA 1: Extração no conjunto de TREINO")
        logger.info("="*60 + "\n")
        extrair_salvar_features(CONFIG.DB_TREINO, "treino", modelo)

        logger.info("\n" + "="*60)
        logger.info("ETAPA 2: Extração no conjunto de TESTE")
        logger.info("="*60 + "\n")
        extrair_salvar_features(CONFIG.DB_TESTE, "teste", modelo)
    except Exception as e:
        logger.error(f"Erro durante extração: {e}")
        return

    logger.info("\n" + "="*60)
    logger.info("✓ Extração completada e salva no disco!")
    logger.info("Pronto para o treinamento XGBoost.")
    logger.info("="*60)


if __name__ == '__main__':
    main()
