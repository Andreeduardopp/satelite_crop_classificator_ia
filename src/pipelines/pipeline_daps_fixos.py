"""
Pipeline V2 - DAPs fixos com retomada idempotente.

Diferenças vs pipeline.py original:

1. **DAPs fixos para todas as culturas** (DAPS_FIXOS = união dos DAPs por
   cultura). Cada talhão recebe imagens nas mesmas janelas, independente da
   cultura declarada. Resolve o failure mode adversarial em que mentir a
   cultura mudava as datas pedidas ao Sentinel Hub.

2. **Skip-if-exists por DAP**: antes de chamar Sentinel Hub, verifica se
   `./processadas_v2/mascara_<ref>_d<dap>.png` já existe. Se sim, registra
   o caminho e pula para o próximo DAP. Permite estender execuções
   anteriores sem redownload.

3. **DB atualizado com a lista completa de imagens** (existentes em disco +
   novas baixadas), reconstruída a cada rodada — garantindo consistência
   entre disco e DB mesmo após múltiplas execuções.

Banco e pastas são os mesmos do pipeline antigo:
  - DB:       dados_v2.db
  - Imagens:  ./imagens_v2/
  - Máscaras: ./mascaras_v2/
  - Saída:    ./processadas_v2/
"""

import argparse
import ast
import glob
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import requests

from processamento_sentinel_Hub import request_sentinel_hub
from processamento_imagens import aplica_mascara, calcular_area2
from utils import analisar_cobertura_de_nuvens

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DB_NOME            = 'dados_v2.db'
DB_TABELA          = 'culturas'
PASTA_IMAGENS      = './imagens_v2'
PASTA_MASCARAS     = './mascaras_v2'
PASTA_PROCESSADAS  = './processadas_v2'
PASTA_LOGS         = './logs'
ARQUIVO_LOG_ERRO   = os.path.join(PASTA_LOGS, 'error_pipeline.txt')
CSV_DEFAULT        = 'dataframe_processado_v2.csv'

CULTURAS_PERMITIDAS = {'aveia', 'trigo', 'feijão', 'feijao', 'milho', 'soja'}

# União dos DAPs antigos por cultura:
#   soja/milho/feijão -> [21, 31, 56]
#   trigo             -> [26, 32, 47]
#   aveia             -> [29, 44, 64]
# Resultado: 9 DAPs cobrindo o ciclo das 5 culturas permitidas.
DAPS_FIXOS = sorted({21, 26, 29, 31, 32, 44, 47, 56, 64})

# ---- Qualidade de imagem ---------------------------------------------------
# request_sentinel_hub baixa todas as imagens disponíveis na janela de 5 dias
# (data-5 .. data) na pasta ./imagens_v2/<tag>/. Antes confiávamos em [:1]
# (primeira por mtime). Agora avaliamos a cobertura de nuvens de TODAS e
# escolhemos a melhor; se nenhuma estiver dentro do limite, ampliamos a janela.

LIMITE_NUVEM         = 0.30   # cobertura máxima aceita (0..1)
LIMITE_PIXELS_NAO_ZERO = 0.05 # mín. fração de pixels não-zero na processada
LIMITE_STD_INTERIOR  = 5.0    # std mín. dos pixels dentro do talhão (uint8)
JANELA_EXTENSAO_DIAS = 7      # extensão forward se a janela inicial for nublada

# ---- Retry de rede ---------------------------------------------------------
RETRY_TENTATIVAS = 3
RETRY_BACKOFF_S  = 2.0  # 2^0=1s, 2^1=2s, 2^2=4s


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configurar_logging():
    os.makedirs(PASTA_LOGS, exist_ok=True)
    logger = logging.getLogger()
    if logger.handlers:
        return  # idempotente em re-imports
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
# Banco de dados
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
                imagens_baixadas TEXT,
                imagens_processadas TEXT
            )
        """)


def upsert_registro(row, area, baixadas, processadas):
    with sqlite3.connect(DB_NOME) as conn:
        conn.execute(f"""
            INSERT INTO {DB_TABELA}
                (cultura, ref_infra_v, ref_rgb, data, mes, path, area,
                 imagens_baixadas, imagens_processadas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ref_infra_v) DO UPDATE SET
                area=excluded.area,
                imagens_baixadas=excluded.imagens_baixadas,
                imagens_processadas=excluded.imagens_processadas
        """, (
            row['cultura'], row['ref_infra_v'], row['ref_rgb'],
            str(row['data']), int(row['mes']), row['path'],
            area, str(baixadas), str(processadas),
        ))


# ---------------------------------------------------------------------------
# Helpers de disco
# ---------------------------------------------------------------------------

def localizar_path_bruto(tag: str):
    """
    O diretório intermediário entre PASTA_IMAGENS/<tag>/ e response.png é um
    hash gerado pelo Sentinel Hub e não é determinístico, então achamos via glob.
    Retorna o primeiro match ou None.
    """
    pattern = os.path.join(PASTA_IMAGENS, tag, '*', 'response.png')
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Robustez de download
# ---------------------------------------------------------------------------

# Tipos de exceção que justificam retry. Erros de KML, configuração ou
# parsing NÃO entram aqui — esses queremos que falhem rápido.
ERROS_REDE = (
    requests.exceptions.RequestException,
    ConnectionError,
    TimeoutError,
)


def chamar_sentinel_com_retry(data_dap, kml, tag):
    """
    Wrapper de request_sentinel_hub com retry exponencial em erros de rede.
    Retorna o que request_sentinel_hub retornar, ou levanta a última exceção.
    """
    last_exc = None
    for tentativa in range(1, RETRY_TENTATIVAS + 1):
        try:
            return request_sentinel_hub(
                data_dap, kml, tag, pasta_imagens=PASTA_IMAGENS,
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


def listar_candidatos(tag: str):
    """Todos os response.png já gravados em PASTA_IMAGENS/<tag>/*/."""
    base = os.path.join(PASTA_IMAGENS, tag)
    if not os.path.isdir(base):
        return []
    candidatos = []
    for hash_dir in os.listdir(base):
        full = os.path.join(base, hash_dir, 'response.png')
        if os.path.exists(full):
            candidatos.append(full)
    return candidatos


def selecionar_melhor_por_nuvem(candidatos):
    """
    Recebe lista de paths e retorna (path, nuvem_frac) com menor cobertura,
    ou (None, None) se a lista for vazia.

    `analisar_cobertura_de_nuvens` (src/utils.py) retorna
    `(caminhos_ordenados_asc, porcentagens_asc)` — já ordenado por nuvem
    crescente, com porcentagens em escala 0..100. Convertemos para fração
    [0..1] para casar com LIMITE_NUVEM.
    """
    if not candidatos:
        return None, None
    try:
        caminhos_ord, pcts_ord = analisar_cobertura_de_nuvens(candidatos)
    except ValueError:
        # Pode acontecer se cv2.imread falhar em todos os candidatos
        # (zip(*[]) levanta ValueError dentro da função).
        return None, None
    if not caminhos_ord:
        return None, None
    return caminhos_ord[0], float(pcts_ord[0]) / 100.0


def baixar_com_qualidade(data_dap, kml, tag):
    """
    Garante uma imagem de boa qualidade para o (talhão, DAP):

    1. Chama o Sentinel Hub na janela [data-5d, data].
    2. Avalia TODAS as imagens da pasta e escolhe a de menor cobertura de nuvem.
    3. Se a melhor ainda passar de LIMITE_NUVEM, faz uma 2ª chamada com a janela
       deslocada para frente em JANELA_EXTENSAO_DIAS dias e reavalia o conjunto.
    4. Retorna (path, nuvem_pct) ou (None, None) se nada acima do limite.
    """
    chamar_sentinel_com_retry(data_dap, kml, tag)
    melhor, nuvem = selecionar_melhor_por_nuvem(listar_candidatos(tag))

    if melhor is None:
        return None, None
    if nuvem <= LIMITE_NUVEM:
        return melhor, nuvem

    # Janela inicial nublada: estende +N dias adiante e tenta de novo.
    logging.info(
        f"  {tag}: melhor inicial nublada ({nuvem:.1%}); "
        f"estendendo +{JANELA_EXTENSAO_DIAS}d"
    )
    try:
        chamar_sentinel_com_retry(
            data_dap + timedelta(days=JANELA_EXTENSAO_DIAS), kml, tag,
        )
    except ERROS_REDE as ex:
        logging.warning(f"  {tag}: rede falhou na extensão: {ex}")

    melhor, nuvem = selecionar_melhor_por_nuvem(listar_candidatos(tag))
    if melhor is None:
        return None, None
    if nuvem > LIMITE_NUVEM:
        logging.warning(
            f"  {tag}: mesmo após extensão a melhor ficou nublada ({nuvem:.1%})"
        )
        return None, nuvem
    return melhor, nuvem


def imagem_processada_valida(path: str) -> bool:
    """
    Valida a imagem após aplica_mascara. Rejeita:
      - leitura falhou (None);
      - menos de LIMITE_PIXELS_NAO_ZERO de pixels não-zero (talhão muito pequeno
        no recorte, bbox fora do tile, ou imagem totalmente preta);
      - desvio padrão dos pixels não-zero abaixo de LIMITE_STD_INTERIOR
        (imagem solidamente uniforme = nuvem branca, artefato, ou tile inválido).
    """
    img = cv2.imread(path)
    if img is None:
        return False
    nao_zero = img[img > 0]
    if nao_zero.size < img.size * LIMITE_PIXELS_NAO_ZERO:
        return False
    if float(nao_zero.std()) < LIMITE_STD_INTERIOR:
        return False
    return True


# ---------------------------------------------------------------------------
# Núcleo: processar um talhão
# ---------------------------------------------------------------------------

def processar_talhao(row):
    """
    Para o talhão `row`, garante que cada DAP em DAPS_FIXOS tenha imagem
    processada em disco. Retorna estatísticas (pulados/baixados/falhos) ou
    None se o talhão foi ignorado por filtro de cultura.
    """
    ref     = row['ref_infra_v']
    cultura = str(row['cultura']).strip().lower()

    if cultura not in CULTURAS_PERMITIDAS:
        return None

    kml     = row['path']
    plantio = datetime.strptime(str(row['data']), '%Y-%m-%d')

    baixadas, processadas = [], []
    daps_pulados, daps_baixados, daps_falhos = [], [], []

    for dap in DAPS_FIXOS:
        tag        = f"{ref}_d{dap}"
        mask       = os.path.join(PASTA_MASCARAS,    f"mascara_{tag}.png")
        processada = os.path.join(PASTA_PROCESSADAS, f"mascara_{tag}.png")

        # ---- skip se já temos a imagem processada ----
        if os.path.exists(processada):
            raw = localizar_path_bruto(tag)
            if raw:
                baixadas.append(raw)
            processadas.append(processada)
            daps_pulados.append(dap)
            continue

        # ---- baixar e processar ----
        data_dap = plantio + timedelta(days=dap)
        try:
            raw, nuvem = baixar_com_qualidade(data_dap, kml, tag)
            if raw is None:
                logging.warning(f"  {tag}: nenhuma imagem dentro do limite de nuvem")
                daps_falhos.append(dap)
                continue

            aplica_mascara(mask, raw, processada, kml)

            if not imagem_processada_valida(processada):
                logging.warning(
                    f"  {tag}: imagem processada inválida (vazia/uniforme); descartando"
                )
                try:
                    os.remove(processada)
                except OSError:
                    pass
                daps_falhos.append(dap)
                continue

            logging.info(f"  {tag}: ok (nuvem={nuvem:.1%})")
            baixadas.append(raw)
            processadas.append(processada)
            daps_baixados.append(dap)

        except FileNotFoundError:
            logging.warning(f"FileNotFoundError: {tag} (DAP {dap})")
            daps_falhos.append(dap)
        except ERROS_REDE as ex:
            logging.error(f"  {tag}: rede falhou após retries: {ex}")
            daps_falhos.append(dap)
        except Exception as ex:
            logging.error(f"Erro DAP {dap} para {ref}: {ex}")
            daps_falhos.append(dap)

    if not (baixadas or processadas):
        logging.info(f"  -> nenhuma imagem para {ref}; pulando upsert.")
        return {'ref': ref, 'pulados': [], 'baixados': [], 'falhos': daps_falhos}

    area = calcular_area2(kml)
    upsert_registro(row, area, baixadas, processadas)

    return {
        'ref': ref,
        'pulados':  daps_pulados,
        'baixados': daps_baixados,
        'falhos':   daps_falhos,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline V2 — DAPs fixos com retomada idempotente"
    )
    parser.add_argument('--csv',   default=CSV_DEFAULT,
                        help=f"CSV de entrada (default: {CSV_DEFAULT})")
    parser.add_argument('--limit', type=int, default=None,
                        help="Máx. de talhões a processar (debug)")
    parser.add_argument('--start', type=int, default=0,
                        help="Pular as primeiras N linhas do CSV (retomada manual)")
    args = parser.parse_args()

    configurar_logging()
    inicializar_db()
    os.makedirs(PASTA_IMAGENS,     exist_ok=True)
    os.makedirs(PASTA_MASCARAS,    exist_ok=True)
    os.makedirs(PASTA_PROCESSADAS, exist_ok=True)

    logging.info(f"DAPs fixos: {DAPS_FIXOS} ({len(DAPS_FIXOS)} janelas/talhão)")
    logging.info(f"Lendo {args.csv}...")
    df = pd.read_csv(args.csv)
    if args.start:
        df = df.iloc[args.start:]
    if args.limit:
        df = df.head(args.limit)
    total = len(df)
    logging.info(f"Total de linhas a processar: {total}")

    resumo = {
        'pulados_total':       0,
        'baixados_total':      0,
        'falhos_total':        0,
        'talhoes_completos':   0,
        'talhoes_processados': 0,
        'talhoes_filtrados':   0,
    }

    for i, (_, row) in enumerate(df.iterrows(), 1):
        ref = row.get('ref_infra_v', '?')
        cultura = row.get('cultura', '?')
        logging.info(f"[{i}/{total}] {ref} ({cultura})")

        try:
            res = processar_talhao(row)
        except Exception as ex:
            logging.error(f"Erro inesperado em {ref}: {ex}")
            continue

        if res is None:
            resumo['talhoes_filtrados'] += 1
            continue

        n_p, n_b, n_f = len(res['pulados']), len(res['baixados']), len(res['falhos'])
        resumo['pulados_total']       += n_p
        resumo['baixados_total']      += n_b
        resumo['falhos_total']        += n_f
        resumo['talhoes_processados'] += 1
        if (n_p + n_b) == len(DAPS_FIXOS):
            resumo['talhoes_completos'] += 1

        logging.info(
            f"  -> cache: {n_p} | novos: {n_b} | falhos: {n_f}  "
            f"(total disco: {n_p + n_b}/{len(DAPS_FIXOS)})"
        )

    logging.info("=" * 60)
    logging.info("Resumo final:")
    logging.info(f"  Talhões processados:      {resumo['talhoes_processados']}")
    logging.info(f"  Talhões filtrados:        {resumo['talhoes_filtrados']}")
    logging.info(f"  Talhões 100% completos:   {resumo['talhoes_completos']}")
    logging.info(f"  DAPs pulados (cache):     {resumo['pulados_total']}")
    logging.info(f"  DAPs baixados (novos):    {resumo['baixados_total']}")
    logging.info(f"  DAPs falhos:              {resumo['falhos_total']}")


if __name__ == '__main__':
    main()
