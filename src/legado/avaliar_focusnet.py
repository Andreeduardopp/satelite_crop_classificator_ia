"""
Avalia a acurácia e F1-score do serviço Focusnet para classificação de culturas.

Fluxo:
  1. Lê KMLs das pastas locais arquivos_kml_*_sample200 (50 por cultura)
  2. Extrai cultura e data de plantio a partir do nome do arquivo
  3. Envia cada KML para /classificar/ e obtém task_id
  4. Aguarda resultado via polling em /classificacao/{task_id}
  5. Extrai predição pelo argmax do vetor_softmax
  6. Calcula acurácia e F1-score (macro) ao final

Estrutura das pastas KML:
  arquivos_kml_/
    arquivos_kml_AVEIA_sample200/arquivos_kml_AVEIA/
      AVEIA_<id>_plantio_DD-MM-YY_colheita_DD-MM-YY.kml
    arquivos_kml_FEIJAO_sample200/arquivos_kml_FEIJAO/
      FEIJÃO_<id>_plantio_DD-MM-YY_colheita_DD-MM-YY.kml
    ...

Uso:
  python legado/avaliar_focusnet.py [--por-cultura N] [--kml-base DIR] [--saida CSV] [--seed N]
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

URL_BASE = "https://focusnet.softfocus.com.br"
TOKEN = "a21ce3672b10131a55f4bb0b2d56581614f1080d"

URL_CLASSIFICAR = f"{URL_BASE}/classificar/"
URL_CLASSIFICACAO = f"{URL_BASE}/classificacao/"

# Ordem usada pelo serviço no vetor_softmax
CULTURAS = ["AVEIA", "FEIJÃO", "MILHO", "SOJA", "TRIGO"]

# Mapa: nome da pasta (sem acento) → culture_key da API (com acento)
PASTA_PARA_CULTURA = {
    "AVEIA":  "AVEIA",
    "FEIJAO": "FEIJÃO",
    "MILHO":  "MILHO",
    "SOJA":   "SOJA",
    "TRIGO":  "TRIGO",
}

POLL_INTERVALO = 10       # segundos entre cada poll
POLL_MAX_TENTATIVAS = 100  # máx ~10 min por amostra
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Carregamento dos KMLs locais
# ---------------------------------------------------------------------------

def _extrair_data_plantio(nome_arquivo: str) -> str:
    """
    Extrai a data de plantio do nome do arquivo KML e converte para YYYY-MM-DD.
    Padrão esperado: ..._plantio_DD-MM-YY_...
    Exemplo: AVEIA_517258447-1_plantio_01-06-24_colheita_10-10-24.kml
             → 2024-06-01
    """
    match = re.search(r"_plantio_(\d{2})-(\d{2})-(\d{2})_", nome_arquivo)
    if not match:
        raise ValueError(f"Data de plantio não encontrada em: {nome_arquivo}")
    dia, mes, ano_curto = match.groups()
    ano = f"20{ano_curto}"
    return f"{ano}-{mes}-{dia}"


def carregar_amostras_kml(kml_base: str, por_cultura: int, semente: int = 42):
    """
    Lê arquivos KML das pastas locais e retorna lista de dicts com:
      path, culture_key, data_plantio
    Amostra `por_cultura` arquivos de cada cultura suportada.
    """
    amostras = []
    random.seed(semente)

    for pasta_nome, culture_key in PASTA_PARA_CULTURA.items():
        pasta = Path(kml_base) / f"arquivos_kml_{pasta_nome}_sample200" / f"arquivos_kml_{pasta_nome}"
        if not pasta.exists():
            print(f"AVISO: pasta não encontrada — {pasta}")
            continue

        kmls = sorted(pasta.glob("*.kml"))
        if not kmls:
            print(f"AVISO: nenhum KML em {pasta}")
            continue

        selecionados = random.sample(kmls, min(por_cultura, len(kmls)))
        print(f"  {culture_key:<8}: {len(selecionados)} KMLs selecionados de {len(kmls)} disponíveis")

        for kml_path in selecionados:
            try:
                data_plantio = _extrair_data_plantio(kml_path.name)
            except ValueError as e:
                print(f"  AVISO: {e}")
                continue
            amostras.append({
                "path": str(kml_path),
                "culture_key": culture_key,
                "data": data_plantio,
            })

    random.shuffle(amostras)
    return amostras


# ---------------------------------------------------------------------------
# Helpers de API
# ---------------------------------------------------------------------------

def enviar_classificacao(kml_path: str, culture_key: str, data: str):
    """POST /classificar/ — retorna o JSON da resposta ou levanta exceção."""
    # Lê o conteúdo e fecha o arquivo antes de montar a requisição,
    # garantindo que apenas 1 KML esteja em memória por vez.
    kml_bytes = Path(kml_path).read_bytes()
    files = {"kml_file": (os.path.basename(kml_path), kml_bytes, "application/vnd.google-earth.kml+xml")}
    dados = {"culture_key": culture_key, "date": data, "token": TOKEN}
    resp = requests.post(URL_CLASSIFICAR, data=dados, files=files, timeout=REQUEST_TIMEOUT)
    del kml_bytes  # libera o buffer após o envio

    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code == 400:
        raise ValueError(f"400 Bad Request: {resp.text}")
    else:
        raise RuntimeError(f"Erro {resp.status_code}: {resp.text}")


def consultar_fila(task_id: str):
    """GET /classificacao/{task_id} — retorna JSON da fila."""
    url = f"{URL_CLASSIFICACAO}{task_id}"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code == 400:
        raise ValueError(f"Task ID inválido: {task_id}")
    elif resp.status_code in (500, 240):
        raise RuntimeError(f"Erro de servidor na fila ({resp.status_code})")
    else:
        raise RuntimeError(f"Erro inesperado na fila {resp.status_code}: {resp.text}")


def aguardar_resultado(task_id: str):
    """
    Faz polling até o vetor_softmax estar disponível.
    Segue a lógica de SalvarDadosCulturaSensoriamentoRemotoUseCase:
      vetor = dados['vetor_softmax'][-1]
    Retorna o vetor (lista de 5 floats) ou None em timeout.
    """
    for tentativa in range(1, POLL_MAX_TENTATIVAS + 1):
        dados = consultar_fila(task_id)
        status = dados.get("status", "?")

        # Se o status é erro, cancela e retorna None (será ignorado)
        if status and "erro" in status.lower():
            print(f"    Status de erro: {status!r}")
            return None

        vetor_list = dados.get("vetor_softmax")
        # Enquanto processa, elementos são 0 (int) como placeholder.
        # O vetor final é uma lista de 5 floats, ex: [0.05, 0.12, 0.68, 0.10, 0.05]
        ultimo = vetor_list[-1] if isinstance(vetor_list, list) and len(vetor_list) > 0 else None
        if isinstance(ultimo, list) and len(ultimo) == len(CULTURAS):
            return ultimo

        print(f"    [{tentativa}/{POLL_MAX_TENTATIVAS}] status={status!r} vetor={vetor_list} — aguardando {POLL_INTERVALO}s...")
        time.sleep(POLL_INTERVALO)

    print("    Timeout: resultado não chegou.")
    return None


# ---------------------------------------------------------------------------
# Processamento de cada amostra
# ---------------------------------------------------------------------------

def processar_amostra(amostra: dict, idx: int, total: int):
    """
    Envia um KML ao serviço, aguarda o vetor_softmax e retorna dict de resultado.
    Retorna None em caso de erro.
    """
    kml_path = amostra["path"]
    culture_key = amostra["culture_key"]
    data = amostra["data"]

    print(f"\n[{idx}/{total}] {culture_key} | {data} | {os.path.basename(kml_path)}")

    t0 = time.time()
    try:
        resposta_inicial = enviar_classificacao(kml_path, culture_key, data)
        print(f"  Enviado. Resposta: {json.dumps(resposta_inicial)[:150]}")

        task_id = resposta_inicial.get("task_id") or resposta_inicial.get("id")
        if not task_id:
            print(f"  ERRO: task_id ausente na resposta: {resposta_inicial}")
            return None

        vetor = aguardar_resultado(str(task_id))
        tempo_seg = round(time.time() - t0, 1)

        if vetor is None:
            print(f"  ERRO: vetor_softmax nulo após espera. ({tempo_seg}s)")
            return None

        pred_idx = vetor.index(max(vetor))
        pred_cultura = CULTURAS[pred_idx]
        confianca = round(max(vetor), 4)
        prob_true = round(vetor[CULTURAS.index(culture_key)], 4) if culture_key in CULTURAS else None

        acerto = culture_key == pred_cultura
        print(f"  True={culture_key} | Pred={pred_cultura} | conf={confianca} | prob_true={prob_true} | {'✓' if acerto else '✗'} | {tempo_seg}s")
        print(f"  Vetor: { {c: round(v, 4) for c, v in zip(CULTURAS, vetor)} }")

        return {
            "kml": kml_path,
            "data": data,
            "true": culture_key,
            "pred": pred_cultura,
            "confianca": confianca,
            "prob_true": prob_true,
            "acerto": acerto,
            "tempo_seg": tempo_seg,
        }

    except (ValueError, RuntimeError) as exc:
        print(f"  ERRO na requisição: {exc} ({round(time.time() - t0, 1)}s)")
        return None
    except Exception as exc:
        print(f"  ERRO inesperado: {exc} ({round(time.time() - t0, 1)}s)")
        return None


# ---------------------------------------------------------------------------
# Métricas e relatório
# ---------------------------------------------------------------------------

def calcular_metricas(resultados):
    classes = CULTURAS  # garante ordem e inclui classes sem amostras
    total = len(resultados)
    corretos = sum(1 for r in resultados if r["true"] == r["pred"])
    acuracia = corretos / total if total else 0.0

    f1_por_classe = {}
    for cls in classes:
        tp = sum(1 for r in resultados if r["true"] == cls and r["pred"] == cls)
        fp = sum(1 for r in resultados if r["true"] != cls and r["pred"] == cls)
        fn = sum(1 for r in resultados if r["true"] == cls and r["pred"] != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_por_classe[cls] = {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}

    classes_com_amostras = [c for c in classes if f1_por_classe[c]["tp"] + f1_por_classe[c]["fn"] > 0]
    f1_macro = (
        sum(f1_por_classe[c]["f1"] for c in classes_com_amostras) / len(classes_com_amostras)
        if classes_com_amostras else 0.0
    )
    return acuracia, f1_macro, f1_por_classe


def imprimir_relatorio(resultados, acuracia, f1_macro, f1_por_classe):
    print("\n" + "=" * 65)
    print("RELATÓRIO DE AVALIAÇÃO — FOCUSNET CLASSIFICAÇÃO DE CULTURAS")
    print("=" * 65)
    print(f"Total de amostras avaliadas : {len(resultados)}")
    tempos = [r["tempo_seg"] for r in resultados]
    print(f"Acurácia                    : {acuracia:.4f} ({acuracia * 100:.2f}%)")
    print(f"F1-score macro              : {f1_macro:.4f}")
    print(f"Tempo médio por request     : {sum(tempos)/len(tempos):.1f}s")
    print(f"Tempo min/max               : {min(tempos):.1f}s / {max(tempos):.1f}s")
    print(f"Tempo total                 : {sum(tempos):.0f}s ({sum(tempos)/60:.1f} min)")
    print()
    print(f"{'Cultura':<12} {'F1':>8} {'Precision':>10} {'Recall':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 65)
    for cls in CULTURAS:
        m = f1_por_classe[cls]
        print(f"{cls:<12} {m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    erros = [r for r in resultados if not r["acerto"]]
    if erros:
        print(f"\nErros ({len(erros)} de {len(resultados)}):")
        print(f"  {'True':<12} {'Pred':<12} KML")
        for e in erros[:20]:
            print(f"  {e['true']:<12} {e['pred']:<12} {os.path.basename(e['kml'])}")
        if len(erros) > 20:
            print(f"  ... e mais {len(erros) - 20} erros.")
    print("=" * 65)


CSV_CAMPOS = ["kml", "data", "true", "pred", "confianca", "prob_true", "acerto", "tempo_seg"]
_csv_lock = threading.Lock()


def salvar_csv(resultados, saida_path):
    with _csv_lock:
        with open(saida_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_CAMPOS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(resultados)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Avalia o serviço Focusnet com amostras KML locais (50 por cultura)"
    )
    parser.add_argument(
        "--por-cultura", type=int, default=100,
        help="Quantidade de KMLs por cultura (default: 50)"
    )
    parser.add_argument(
        "--kml-base", default=None,
        help="Diretório raiz com as pastas arquivos_kml_*_sample200"
    )
    parser.add_argument(
        "--saida", default=None,
        help="Caminho do CSV de saída com resultados individuais"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente aleatória para amostragem (default: 42)"
    )
    parser.add_argument(
        "--paralelo", type=int, default=5,
        help="Quantidade de requisições em paralelo (default: 5)"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent  # data_retreino/
    kml_base = args.kml_base or str(repo_root / "arquivos_kml_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida_path = args.saida or str(Path(__file__).resolve().parent / f"resultados_focusnet_{ts}.csv")

    print(f"Pasta KML base : {kml_base}")
    print(f"Por cultura    : {args.por_cultura}")
    print(f"Paralelo       : {args.paralelo}")
    print(f"Seed           : {args.seed}")
    print(f"CSV de saída   : {saida_path}")
    print()

    print("Carregando amostras...")
    amostras = carregar_amostras_kml(kml_base, args.por_cultura, args.seed)
    total = len(amostras)
    print(f"Total de amostras: {total}\n")

    if total == 0:
        print("Nenhuma amostra encontrada. Verifique --kml-base.")
        sys.exit(1)

    resultados = []
    concluidos = 0

    def _worker(idx_amostra):
        nonlocal concluidos
        idx, amostra = idx_amostra
        resultado = processar_amostra(amostra, idx, total)
        if resultado:
            with _csv_lock:
                resultados.append(resultado)
                concluidos += 1
                print(f"  >> Concluídos: {concluidos}/{total}")
            salvar_csv(resultados, saida_path)
        return resultado

    with ThreadPoolExecutor(max_workers=args.paralelo) as executor:
        futures = {executor.submit(_worker, (idx, a)): idx for idx, a in enumerate(amostras, 1)}
        for future in as_completed(futures):
            future.result()  # propaga exceções se houver

    if not resultados:
        print("\nNenhum resultado obtido. Verifique conectividade com o serviço.")
        sys.exit(1)

    acuracia, f1_macro, f1_por_classe = calcular_metricas(resultados)
    imprimir_relatorio(resultados, acuracia, f1_macro, f1_por_classe)
    print(f"\nResultados salvos em: {saida_path}")


if __name__ == "__main__":
    main()
