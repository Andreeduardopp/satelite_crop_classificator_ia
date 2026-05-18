"""
Pipeline de aquisição V9 — TIFF multispectral com DAPs fixos e retomada.

Equivalente ao `pipeline_daps_fixos.py` (PNG, V7/V8) adaptado para a Processing
API e TIFF FLOAT32 6 bandas. Características preservadas:

  - DAPS_FIXOS (união dos DAPs das culturas permitidas);
  - skip-if-exists por DAP (verifica se o TIFF processado já está em disco);
  - retry exponencial para erros de rede;
  - validação da imagem processada antes de gravar no DB.

Banco e pastas distintas do pipeline antigo, para os dois conviverem:
  - DB:               dados_v9.db          (tabela 'culturas_tiff')
  - TIFFs brutos:     ./imagens_tiff_v9/<tag>/<hash>/response.tiff
  - Máscaras:         ./mascaras_v2/...    (reaproveitadas — são as mesmas)
  - TIFFs prontos:    ./processadas_tiff_v9/mascara_<tag>.tiff
"""

import argparse
import ast
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from sentinelhub import SHConfig

# Permite rodar a partir de qualquer pasta:
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'dados')))

from processamento_imagens import cria_mascara, calcular_area2  # noqa: E402

from sentinel_tiff_request import (  # noqa: E402
    baixar_tiff_s2,
    localizar_tiff_existente,
)
from tiff_io import (  # noqa: E402
    aplicar_mascara_tiff,
    imagem_tiff_valida,
)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DB_NOME            = 'dados_v9.db'
DB_TABELA          = 'culturas_tiff'
PASTA_IMAGENS      = './imagens_tiff_v9'
PASTA_MASCARAS     = './mascaras_v2'           # reaproveita as do PNG
PASTA_PROCESSADAS  = './processadas_tiff_v9'
PASTA_LOGS         = './logs'
ARQUIVO_LOG_ERRO   = os.path.join(PASTA_LOGS, 'error_pipeline_v9.txt')
CSV_DEFAULT        = 'dataframe_processado_v2.csv'

CULTURAS_PERMITIDAS = {'aveia', 'trigo', 'feijão', 'feijao', 'milho', 'soja'}

# Mesmos DAPs do pipeline_daps_fixos.py — para que V9 e V7/V8 sejam comparáveis
DAPS_FIXOS = sorted({21, 26, 29, 31, 32, 44, 47, 56, 64})

LIMITE_NUVEM         = 0.30
JANELA_DIAS          = 15
RETRY_TENTATIVAS     = 3
RETRY_BACKOFF_S      = 2.0

ERROS_REDE = (
    requests.exceptions.RequestException,
    ConnectionError,
    TimeoutError,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configurar_logging():
    os.makedirs(PASTA_LOGS, exist_ok=True)
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(ARQUIVO_LOG_ERRO)
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def inicializar_db():
    with sqlite3.connect(DB_NOME) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_TABELA} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cultura TEXT,
                ref_infra_v TEXT UNIQUE,
                ref_rgb TEXT,
                data TEXT,
                mes INTEGER,
                path TEXT,
                area REAL,
                tiffs_brutos TEXT,
                tiffs_processados TEXT,
                versao_pipeline TEXT DEFAULT 'v9_tiff'
            )
        """)


def upsert_registro(row, area, brutos, processados):
    with sqlite3.connect(DB_NOME) as conn:
        conn.execute(f"""
            INSERT INTO {DB_TABELA}
                (cultura, ref_infra_v, ref_rgb, data, mes, path, area,
                 tiffs_brutos, tiffs_processados)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ref_infra_v) DO UPDATE SET
                area=excluded.area,
                tiffs_brutos=excluded.tiffs_brutos,
                tiffs_processados=excluded.tiffs_processados
        """, (
            row['cultura'], row['ref_infra_v'], row['ref_rgb'],
            str(row['data']), int(row['mes']), row['path'],
            area, str(brutos), str(processados),
        ))


# ---------------------------------------------------------------------------
# Sentinel Hub
# ---------------------------------------------------------------------------

def carregar_config_sh():
    """Carrega SHConfig do perfil 'main_test' (mesmo do V7/V8)."""
    config = SHConfig('main_test')
    if not config.sh_client_id or not config.sh_client_secret:
        raise RuntimeError(
            "SHConfig vazio. Configure com SHConfig.save() antes de rodar."
        )
    return config


def chamar_sentinel_com_retry(data_alvo, kml, tag, config):
    last_exc = None
    for tentativa in range(1, RETRY_TENTATIVAS + 1):
        try:
            return baixar_tiff_s2(
                data_alvo=data_alvo,
                kml_path=kml,
                tag=tag,
                output_dir=PASTA_IMAGENS,
                config=config,
                janela_dias=JANELA_DIAS,
                max_cloud_cover=LIMITE_NUVEM,
            )
        except ERROS_REDE as ex:
            last_exc = ex
            espera = RETRY_BACKOFF_S ** (tentativa - 1)
            logging.warning(
                f"  retry {tentativa}/{RETRY_TENTATIVAS} para {tag}: "
                f"{type(ex).__name__}: {ex}; aguardando {espera:.0f}s"
            )
            time.sleep(espera)
    raise last_exc


# ---------------------------------------------------------------------------
# Por talhão
# ---------------------------------------------------------------------------

def processar_talhao(row, config):
    ref     = row['ref_infra_v']
    cultura = str(row['cultura']).strip().lower()
    if cultura not in CULTURAS_PERMITIDAS:
        return None

    kml     = row['path']
    plantio = datetime.strptime(str(row['data']), '%Y-%m-%d')

    brutos, processados = [], []
    pulados, baixados, falhos = [], [], []

    for dap in DAPS_FIXOS:
        tag        = f"{ref}_d{dap}"
        mascara    = os.path.join(PASTA_MASCARAS,    f"mascara_{tag}.png")
        processada = os.path.join(PASTA_PROCESSADAS, f"mascara_{tag}.tiff")

        # ---- skip-if-exists ----
        if os.path.exists(processada):
            bruto = localizar_tiff_existente(tag, PASTA_IMAGENS)
            if bruto:
                brutos.append(bruto)
            processados.append(processada)
            pulados.append(dap)
            continue

        # ---- baixar ----
        data_dap = plantio + timedelta(days=dap)
        try:
            tiff_path, _meta = chamar_sentinel_com_retry(
                data_dap, kml, tag, config
            )
        except ERROS_REDE as ex:
            logging.error(f"  {tag}: rede falhou após retries: {ex}")
            falhos.append(dap)
            continue
        except Exception as ex:
            logging.error(f"  {tag}: erro inesperado no download: {ex}")
            falhos.append(dap)
            continue

        if tiff_path is None:
            logging.warning(f"  {tag}: nada disponível na janela (leastCC vazio)")
            falhos.append(dap)
            continue

        # ---- garantir máscara ----
        if not os.path.exists(mascara):
            try:
                cria_mascara(kml, mascara)
            except Exception as ex:
                logging.error(f"  {tag}: erro criando máscara: {ex}")
                falhos.append(dap)
                continue

        # ---- aplicar máscara e validar ----
        try:
            ok = aplicar_mascara_tiff(tiff_path, mascara, processada)
        except Exception as ex:
            logging.error(f"  {tag}: erro aplicando máscara: {ex}")
            falhos.append(dap)
            continue

        if not ok:
            logging.warning(f"  {tag}: aplica_mascara_tiff retornou False")
            falhos.append(dap)
            continue

        if not imagem_tiff_valida(processada):
            logging.warning(
                f"  {tag}: TIFF processado inválido (vazio/uniforme); descartando"
            )
            try:
                os.remove(processada)
            except OSError:
                pass
            falhos.append(dap)
            continue

        logging.info(f"  {tag}: ok")
        brutos.append(tiff_path)
        processados.append(processada)
        baixados.append(dap)

    if not (brutos or processados):
        return {'ref': ref, 'pulados': [], 'baixados': [], 'falhos': falhos}

    area = calcular_area2(kml)
    upsert_registro(row, area, brutos, processados)
    return {
        'ref': ref,
        'pulados': pulados,
        'baixados': baixados,
        'falhos': falhos,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline V9 — TIFF multispectral com DAPs fixos"
    )
    parser.add_argument('--csv',   default=CSV_DEFAULT)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    configurar_logging()
    inicializar_db()
    os.makedirs(PASTA_IMAGENS,     exist_ok=True)
    os.makedirs(PASTA_PROCESSADAS, exist_ok=True)
    os.makedirs(PASTA_MASCARAS,    exist_ok=True)

    config = carregar_config_sh()

    logging.info(f"DAPs fixos: {DAPS_FIXOS} ({len(DAPS_FIXOS)} janelas/talhão)")
    logging.info(f"Janela: {JANELA_DIAS} dias | maxcc server-side: {LIMITE_NUVEM}")
    logging.info(f"Lendo {args.csv}...")
    df = pd.read_csv(args.csv)
    if args.start:
        df = df.iloc[args.start:]
    if args.limit:
        df = df.head(args.limit)
    total = len(df)

    resumo = {
        'pulados': 0, 'baixados': 0, 'falhos': 0,
        'completos': 0, 'processados': 0, 'filtrados': 0,
    }

    for i, (_, row) in enumerate(df.iterrows(), 1):
        ref = row.get('ref_infra_v', '?')
        cultura = row.get('cultura', '?')
        logging.info(f"[{i}/{total}] {ref} ({cultura})")

        try:
            res = processar_talhao(row, config)
        except Exception as ex:
            logging.error(f"Erro inesperado em {ref}: {ex}")
            continue

        if res is None:
            resumo['filtrados'] += 1
            continue

        n_p, n_b, n_f = len(res['pulados']), len(res['baixados']), len(res['falhos'])
        resumo['pulados']     += n_p
        resumo['baixados']    += n_b
        resumo['falhos']      += n_f
        resumo['processados'] += 1
        if (n_p + n_b) == len(DAPS_FIXOS):
            resumo['completos'] += 1

        logging.info(
            f"  -> cache: {n_p} | novos: {n_b} | falhos: {n_f} "
            f"({n_p + n_b}/{len(DAPS_FIXOS)})"
        )

    logging.info("=" * 60)
    logging.info("Resumo final V9 TIFF:")
    for k, v in resumo.items():
        logging.info(f"  {k}: {v}")


if __name__ == '__main__':
    main()
