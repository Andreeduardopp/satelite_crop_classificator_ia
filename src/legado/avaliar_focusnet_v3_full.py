"""
Avalia a acurácia e F1-score do serviço Focusnet v3 para classificação de culturas.

Fluxo:
  1. Lê KMLs das pastas locais arquivos_kml_*_sample200
  2. Extrai cultura e data de plantio a partir do nome do arquivo
  3. Envia cada KML para /classificar/v3/ e obtém task_id
  4. Aguarda resultado via polling em /classificacao/{task_id}
  5. Extrai predição diretamente do campo "resultado"
  6. Calcula acurácia e F1-score (macro) ao final

Uso:
  python legado/avaliar_focusnet_v3_full.py [--por-cultura N] [--kml-base DIR] [--saida CSV] [--seed N] [--paralelo N]
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
TOKEN = "574cd4a215146cbb05c3d9820e5e99e28e6c4309"

URL_BASE_LOCAL = "http://localhost:8000"
TOKEN_LOCAL= "b0932531f68b9fc2a5da8b88cf21534b73468602"


URL_CLASSIFICAR = f"{URL_BASE_LOCAL}/classificar/v3/"
URL_CLASSIFICACAO = f"{URL_BASE_LOCAL}/classificacao/"

# Culturas suportadas (chaves normalizadas para comparação)
CULTURAS = ["AVEIA", "FEIJÃO", "MILHO", "SOJA", "TRIGO"]

PASTA_PARA_CULTURA = {
    "AVEIA":  "AVEIA",
    "FEIJAO": "FEIJÃO",
    "MILHO":  "MILHO",
    "SOJA":   "SOJA",
    "TRIGO":  "TRIGO",
}

POLL_INTERVALO = 10
POLL_MAX_TENTATIVAS = 100
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Carregamento dos KMLs
# ---------------------------------------------------------------------------

def _extrair_data_plantio(nome_arquivo: str) -> str:
    match = re.search(r"_plantio_(\d{2})-(\d{2})-(\d{2})_", nome_arquivo)
    if not match:
        raise ValueError(f"Data de plantio não encontrada em: {nome_arquivo}")
    dia, mes, ano_curto = match.groups()
    return f"20{ano_curto}-{mes}-{dia}"


def carregar_amostras_kml(kml_base: str, por_cultura: int, semente: int = 42):
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
# API
# ---------------------------------------------------------------------------

def enviar_classificacao(kml_path: str, culture_key: str, data: str):
    kml_bytes = Path(kml_path).read_bytes()
    files = {"kml_file": (os.path.basename(kml_path), kml_bytes, "application/vnd.google-earth.kml+xml")}
    dados = {"culture_key": culture_key, "date": data, "token": TOKEN_LOCAL, "data_plantio": data, "cultura": culture_key}
    print(dados)
    resp = requests.post(URL_CLASSIFICAR, data=dados, files=files, timeout=REQUEST_TIMEOUT)
    del kml_bytes
    resp.raise_for_status()
    return resp.json()


def aguardar_resultado(task_id: str):
    for tentativa in range(1, POLL_MAX_TENTATIVAS + 1):
        resp = requests.get(f"{URL_CLASSIFICACAO}{task_id}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        dados = resp.json()
        status = (dados.get("status") or "").upper()

        if "ERRO" in status:
            print(f"    Status de erro: {status!r}")
            return dados

        if "SUCESSO" in status:
            return dados

        print(f"    [{tentativa}/{POLL_MAX_TENTATIVAS}] status={status!r} — aguardando {POLL_INTERVALO}s...")
        time.sleep(POLL_INTERVALO)

    print("    Timeout: resultado não chegou.")
    return None


# ---------------------------------------------------------------------------
# Processamento de cada amostra
# ---------------------------------------------------------------------------

def _normalizar_cultura(s: str) -> str:
    return s.upper().replace("FEIJAO", "FEIJÃO").strip()


def processar_amostra(amostra: dict, idx: int, total: int):
    kml_path = amostra["path"]
    culture_key = amostra["culture_key"]
    data = amostra["data"]

    print(f"\n[{idx}/{total}] {culture_key} | {data} | {os.path.basename(kml_path)}")

    t0 = time.time()
    try:
        resposta_inicial = enviar_classificacao(kml_path, culture_key, data)
        task_id = resposta_inicial.get("task_id") or resposta_inicial.get("id")
        if not task_id:
            print(f"  ERRO: task_id ausente na resposta: {resposta_inicial}")
            return None

        print(f"  Enviado. task_id={task_id}")
        resultado = aguardar_resultado(str(task_id))
        tempo_seg = round(time.time() - t0, 1)

        if resultado is None:
            print(f"  ERRO: sem resultado após espera. ({tempo_seg}s)")
            return None

        # v3 retorna "resultado" e "probabilidades" diretamente
        pred_raw = resultado.get("resultado") or ""
        pred_cultura = _normalizar_cultura(pred_raw)
        confianca = resultado.get("confianca_pct")
        probabilidades = resultado.get("probabilidades", {})

        prob_true = probabilidades.get(culture_key.lower()) or probabilidades.get(culture_key)
        acerto = culture_key == pred_cultura

        print(f"  True={culture_key} | Pred={pred_cultura} | conf={confianca} | prob_true={prob_true} | {'✓' if acerto else '✗'} | {tempo_seg}s")
        print(f"  Probabilidades: {probabilidades}")

        return {
            "kml": os.path.basename(kml_path),
            "data": data,
            "true": culture_key,
            "pred": pred_cultura,
            "confianca": confianca,
            "prob_true": prob_true,
            "acerto": acerto,
            "tempo_seg": tempo_seg,
        }

    except (ValueError, RuntimeError, requests.HTTPError) as exc:
        print(f"  ERRO na requisição: {exc} ({round(time.time() - t0, 1)}s)")
        return None
    except Exception as exc:
        print(f"  ERRO inesperado: {exc} ({round(time.time() - t0, 1)}s)")
        return None


# ---------------------------------------------------------------------------
# Métricas e relatório
# ---------------------------------------------------------------------------

def calcular_metricas(resultados):
    total = len(resultados)
    corretos = sum(1 for r in resultados if r["true"] == r["pred"])
    acuracia = corretos / total if total else 0.0

    f1_por_classe = {}
    for cls in CULTURAS:
        tp = sum(1 for r in resultados if r["true"] == cls and r["pred"] == cls)
        fp = sum(1 for r in resultados if r["true"] != cls and r["pred"] == cls)
        fn = sum(1 for r in resultados if r["true"] == cls and r["pred"] != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_por_classe[cls] = {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}

    classes_com_amostras = [c for c in CULTURAS if f1_por_classe[c]["tp"] + f1_por_classe[c]["fn"] > 0]
    f1_macro = (
        sum(f1_por_classe[c]["f1"] for c in classes_com_amostras) / len(classes_com_amostras)
        if classes_com_amostras else 0.0
    )

    # Matriz de confusao: linhas = classe real, colunas = classe predita.
    # Inclui colunas extras se o modelo predisse algo fora de CULTURAS.
    extras = sorted({r["pred"] for r in resultados if r["pred"] not in CULTURAS})
    colunas = list(CULTURAS) + extras
    confusao = {t: {p: 0 for p in colunas} for t in CULTURAS}
    for r in resultados:
        if r["true"] in confusao and r["pred"] in confusao[r["true"]]:
            confusao[r["true"]][r["pred"]] += 1

    return acuracia, f1_macro, f1_por_classe, confusao, colunas


def imprimir_relatorio(resultados, acuracia, f1_macro, f1_por_classe, confusao=None, colunas=None):
    print("\n" + "=" * 65)
    print("RELATÓRIO DE AVALIAÇÃO — FOCUSNET v3 CLASSIFICAÇÃO DE CULTURAS")
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

    if confusao and colunas:
        largura = max(8, max(len(c) for c in colunas) + 2)
        print("\nMatriz de Confusão (linhas=real, colunas=predito):")
        header = f"{'':<12}" + "".join(f"{c:>{largura}}" for c in colunas) + f"{'TOTAL':>{largura}}"
        print(header)
        print("-" * len(header))
        for cls in CULTURAS:
            linha_total = sum(confusao[cls].values())
            celulas = "".join(f"{confusao[cls][p]:>{largura}}" for p in colunas)
            print(f"{cls:<12}{celulas}{linha_total:>{largura}}")

    erros = [r for r in resultados if not r["acerto"]]
    if erros:
        print(f"\nErros ({len(erros)} de {len(resultados)}):")
        print(f"  {'True':<12} {'Pred':<12} KML")
        for e in erros[:20]:
            print(f"  {e['true']:<12} {e['pred']:<12} {e['kml']}")
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
        description="Avalia o serviço Focusnet v3 com amostras KML locais"
    )
    parser.add_argument("--por-cultura", type=int, default=50,
                        help="Quantidade de KMLs por cultura (default: 50)")
    parser.add_argument("--kml-base", default=None,
                        help="Diretório raiz com as pastas arquivos_kml_*_sample200")
    parser.add_argument("--saida", default=None,
                        help="Caminho do CSV de saída")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paralelo", type=int, default=5,
                        help="Requisições em paralelo (default: 5)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    kml_base = args.kml_base or str(repo_root / "arquivos_kml_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida_path = args.saida or str(Path(__file__).resolve().parent / f"resultados_focusnet_v3_{ts}.csv")

    print(f"Endpoint       : {URL_CLASSIFICAR}")
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
            future.result()

    if not resultados:
        print("\nNenhum resultado obtido. Verifique conectividade com o serviço.")
        sys.exit(1)

    acuracia, f1_macro, f1_por_classe, confusao, colunas = calcular_metricas(resultados)
    imprimir_relatorio(resultados, acuracia, f1_macro, f1_por_classe, confusao, colunas)
    print(f"\nResultados salvos em: {saida_path}")


if __name__ == "__main__":
    main()
