"""
Avaliação do Focusnet v3 com filtro de confiança.

Igual ao avaliar_focusnet_v3_full.py, mas só conta como acerto/erro quando a
confiança da classe predita é >= LIMIAR. Caso contrário, marca como UNDEFINIDO
e essa amostra fica fora das métricas de acurácia/F1.

A ideia é simular um "modo conservador" onde o serviço se recusa a opinar
quando não está seguro — útil para entender o trade-off entre cobertura e
qualidade das predições aceitas.

Métricas reportadas:
  - Cobertura: % de amostras onde max_prob >= LIMIAR (foram "aceitas")
  - Acurácia entre aceitas: certo / aceitas
  - F1 macro entre aceitas
  - Por cultura: cobertura, acurácia, abandonos
  - Distribuição de max_prob (histograma) — para escolher LIMIAR informado

Uso:
  python legado/avaliar_focusnet_v3_confianca.py --threshold 0.8 --por-cultura 50
"""

import argparse
import csv
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

URL_BASE_LOCAL = "http://localhost:8000"
TOKEN_LOCAL    = "b0932531f68b9fc2a5da8b88cf21534b73468602"

URL_CLASSIFICAR   = f"{URL_BASE_LOCAL}/classificar/v3/"
URL_CLASSIFICACAO = f"{URL_BASE_LOCAL}/classificacao/"

CULTURAS = ["AVEIA", "FEIJÃO", "MILHO", "SOJA", "TRIGO"]

PASTA_PARA_CULTURA = {
    "AVEIA":  "AVEIA",
    "FEIJAO": "FEIJÃO",
    "MILHO":  "MILHO",
    "SOJA":   "SOJA",
    "TRIGO":  "TRIGO",
}

ROTULO_INDEFINIDO = "UNDEFINIDO"

POLL_INTERVALO       = 10
POLL_MAX_TENTATIVAS  = 100
REQUEST_TIMEOUT      = 30


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
    files = {"kml_file": (os.path.basename(kml_path), kml_bytes,
                          "application/vnd.google-earth.kml+xml")}
    dados = {
        "culture_key": culture_key, "date": data,
        "token": TOKEN_LOCAL, "data_plantio": data, "cultura": culture_key,
    }
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
# Processamento
# ---------------------------------------------------------------------------

def _normalizar_cultura(s: str) -> str:
    return s.upper().replace("FEIJAO", "FEIJÃO").strip()


def _max_probabilidade(probabilidades):
    """
    Pega (cultura, prob) com maior probabilidade no dict do serviço, com prob
    normalizada para [0, 1].

    O serviço Focusnet v3 retorna probabilidades em escala 0..100 (a soma do
    dict bate 100). Detectamos isso pela soma e normalizamos para [0, 1] —
    assim o threshold do CLI fica em fração, consistente com a notação do
    código (0.80 == 80%).
    """
    if not probabilidades:
        return None, 0.0
    soma = sum(float(v) for v in probabilidades.values())
    cultura, prob = max(probabilidades.items(), key=lambda kv: kv[1])
    prob = float(prob)
    if soma > 1.5:                # heurística: 0..100 (soma ~100) vs 0..1 (soma ~1)
        prob = prob / 100.0
    return _normalizar_cultura(cultura), prob


def processar_amostra(amostra, idx, total, threshold):
    kml_path    = amostra["path"]
    culture_key = amostra["culture_key"]
    data        = amostra["data"]

    print(f"\n[{idx}/{total}] {culture_key} | {data} | {os.path.basename(kml_path)}")

    t0 = time.time()
    try:
        resposta = enviar_classificacao(kml_path, culture_key, data)
        task_id = resposta.get("task_id") or resposta.get("id")
        if not task_id:
            print(f"  ERRO: task_id ausente: {resposta}")
            return None
        print(f"  Enviado. task_id={task_id}")

        resultado = aguardar_resultado(str(task_id))
        tempo_seg = round(time.time() - t0, 1)
        if resultado is None:
            print(f"  ERRO: sem resultado. ({tempo_seg}s)")
            return None

        pred_raw       = resultado.get("resultado") or ""
        pred_argmax    = _normalizar_cultura(pred_raw)
        confianca_pct  = resultado.get("confianca_pct")
        probabilidades = resultado.get("probabilidades", {}) or {}

        # Probabilidade da classe predita (em [0,1]).
        # Usamos o dict probabilidades como fonte de verdade — mais robusto
        # que confianca_pct, que pode vir em escala 0-100.
        _, max_prob = _max_probabilidade(probabilidades)

        # Decisão final: aplica o filtro de confiança.
        confiante = max_prob >= threshold
        decisao   = pred_argmax if confiante else ROTULO_INDEFINIDO
        acerto    = (decisao == culture_key)  # UNDEFINIDO nunca acerta

        prob_true = (probabilidades.get(culture_key.lower())
                     or probabilidades.get(culture_key))

        marcador = "OK" if acerto else ("ABSTÉM" if not confiante else "ERRO")
        print(
            f"  True={culture_key} | argmax={pred_argmax} (max_prob={max_prob:.3f}) "
            f"| limiar={threshold:.2f} → decisão={decisao} [{marcador}] | {tempo_seg}s"
        )
        print(f"  Probabilidades: {probabilidades}")

        return {
            "kml":       os.path.basename(kml_path),
            "data":      data,
            "true":      culture_key,
            "argmax":    pred_argmax,
            "max_prob":  round(max_prob, 4),
            "decisao":   decisao,
            "confiante": confiante,
            "acerto":    acerto,
            "confianca_pct": confianca_pct,
            "prob_true": prob_true,
            "tempo_seg": tempo_seg,
        }

    except (ValueError, RuntimeError, requests.HTTPError) as exc:
        print(f"  ERRO na requisição: {exc} ({round(time.time() - t0, 1)}s)")
        return None
    except Exception as exc:
        print(f"  ERRO inesperado: {exc} ({round(time.time() - t0, 1)}s)")
        return None


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def calcular_metricas(resultados, threshold):
    total = len(resultados)
    aceitas = [r for r in resultados if r["confiante"]]
    abstidas = [r for r in resultados if not r["confiante"]]

    cobertura = len(aceitas) / total if total else 0.0

    # Acurácia entre aceitas
    corretas_aceitas = sum(1 for r in aceitas if r["acerto"])
    acuracia_aceitas = corretas_aceitas / len(aceitas) if aceitas else 0.0

    # F1 por classe entre aceitas (não conta abstenções como erro nem como acerto)
    f1_por_classe = {}
    for cls in CULTURAS:
        tp = sum(1 for r in aceitas if r["true"] == cls and r["argmax"] == cls)
        fp = sum(1 for r in aceitas if r["true"] != cls and r["argmax"] == cls)
        fn = sum(1 for r in aceitas if r["true"] == cls and r["argmax"] != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_por_classe[cls] = {
            "f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn,
        }
    classes_com_amostras = [
        c for c in CULTURAS
        if f1_por_classe[c]["tp"] + f1_por_classe[c]["fn"] > 0
    ]
    f1_macro = (
        sum(f1_por_classe[c]["f1"] for c in classes_com_amostras) / len(classes_com_amostras)
        if classes_com_amostras else 0.0
    )

    # Por cultura: cobertura local + acurácia local
    por_cultura = {}
    for cls in CULTURAS:
        do_cls = [r for r in resultados if r["true"] == cls]
        n = len(do_cls)
        n_aceitas = sum(1 for r in do_cls if r["confiante"])
        n_corretas_aceitas = sum(1 for r in do_cls if r["confiante"] and r["acerto"])
        por_cultura[cls] = {
            "total":              n,
            "aceitas":            n_aceitas,
            "abstidas":           n - n_aceitas,
            "cobertura":          n_aceitas / n if n else 0.0,
            "acuracia_aceitas":   n_corretas_aceitas / n_aceitas if n_aceitas else 0.0,
        }

    # Histograma de max_prob — útil para decidir threshold
    bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001]
    histograma = Counter()
    for r in resultados:
        for i in range(len(bins) - 1):
            if bins[i] <= r["max_prob"] < bins[i + 1]:
                histograma[(bins[i], bins[i + 1])] += 1
                break

    # Matriz de confusão entre aceitas
    extras = sorted({r["argmax"] for r in aceitas if r["argmax"] not in CULTURAS})
    colunas = list(CULTURAS) + extras
    confusao = {t: {p: 0 for p in colunas} for t in CULTURAS}
    for r in aceitas:
        if r["true"] in confusao and r["argmax"] in confusao[r["true"]]:
            confusao[r["true"]][r["argmax"]] += 1

    return {
        "threshold":         threshold,
        "total":             total,
        "aceitas":           len(aceitas),
        "abstidas":          len(abstidas),
        "cobertura":         cobertura,
        "acuracia_aceitas":  acuracia_aceitas,
        "f1_macro":          f1_macro,
        "f1_por_classe":     f1_por_classe,
        "por_cultura":       por_cultura,
        "histograma":        histograma,
        "confusao":          confusao,
        "colunas_confusao":  colunas,
    }


def imprimir_relatorio(resultados, m):
    print("\n" + "=" * 72)
    print(f"RELATÓRIO COM FILTRO DE CONFIANÇA — limiar = {m['threshold']:.2f}")
    print("=" * 72)
    print(f"Total de amostras           : {m['total']}")
    print(f"  Aceitas (max_prob >= lim) : {m['aceitas']}  (cobertura: {m['cobertura']:.2%})")
    print(f"  Abstidas (max_prob < lim) : {m['abstidas']}")
    print()
    print(f"Acurácia entre aceitas      : {m['acuracia_aceitas']:.4f} "
          f"({m['acuracia_aceitas']*100:.2f}%)")
    print(f"F1 macro entre aceitas      : {m['f1_macro']:.4f}")
    if resultados:
        tempos = [r["tempo_seg"] for r in resultados]
        print(f"Tempo médio por request     : {sum(tempos)/len(tempos):.1f}s")

    print()
    print(f"{'Cultura':<10} {'F1':>8} {'Prec':>8} {'Rec':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 60)
    for cls in CULTURAS:
        s = m["f1_por_classe"][cls]
        print(f"{cls:<10} {s['f1']:>8.4f} {s['precision']:>8.4f} {s['recall']:>8.4f} "
              f"{s['tp']:>5} {s['fp']:>5} {s['fn']:>5}")

    print()
    print("Cobertura e acurácia POR cultura (true):")
    print(f"{'Cultura':<10} {'Total':>6} {'Aceitas':>8} {'Cob.':>8} {'Acc(aceitas)':>14}")
    print("-" * 50)
    for cls in CULTURAS:
        s = m["por_cultura"][cls]
        print(f"{cls:<10} {s['total']:>6} {s['aceitas']:>8} "
              f"{s['cobertura']:>7.2%} {s['acuracia_aceitas']:>13.2%}")

    print()
    print("Distribuição de max_prob (todas as amostras):")
    for (lo, hi), n in sorted(m["histograma"].items()):
        barra = "#" * min(40, n)
        print(f"  [{lo:.2f}, {hi:.2f}): {n:>4} {barra}")

    if m["confusao"]:
        largura = max(8, max(len(c) for c in m["colunas_confusao"]) + 2)
        print()
        print("Matriz de confusão entre aceitas (linhas=real, colunas=argmax):")
        header = f"{'':<10}" + "".join(f"{c:>{largura}}" for c in m["colunas_confusao"])
        print(header)
        print("-" * len(header))
        for cls in CULTURAS:
            celulas = "".join(f"{m['confusao'][cls][p]:>{largura}}"
                              for p in m["colunas_confusao"])
            print(f"{cls:<10}{celulas}")

    abstidas = [r for r in resultados if not r["confiante"]]
    if abstidas:
        print()
        print(f"Abstenções por cultura real (max_prob < {m['threshold']:.2f}):")
        cnt = Counter(r["true"] for r in abstidas)
        for cls in CULTURAS:
            print(f"  {cls:<10}: {cnt.get(cls, 0)} abstenções")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_CAMPOS = ["kml", "data", "true", "argmax", "max_prob", "decisao",
              "confiante", "acerto", "confianca_pct", "prob_true", "tempo_seg"]
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
        description="Avalia Focusnet v3 com filtro de confiança"
    )
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Limiar de confiança (probabilidade máxima) para aceitar a predição. Default: 0.8")
    parser.add_argument("--por-cultura", type=int, default=50)
    parser.add_argument("--kml-base", default=None)
    parser.add_argument("--saida", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paralelo", type=int, default=5)
    args = parser.parse_args()

    if not (0.0 < args.threshold <= 1.0):
        print(f"ERRO: threshold deve estar em (0, 1]; recebi {args.threshold}")
        sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent.parent
    kml_base = args.kml_base or str(repo_root / "arquivos_kml_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida_path = args.saida or str(
        Path(__file__).resolve().parent
        / f"resultados_focusnet_v3_confianca_{int(args.threshold*100)}_{ts}.csv"
    )

    print(f"Endpoint       : {URL_CLASSIFICAR}")
    print(f"Threshold      : {args.threshold}")
    print(f"Pasta KML base : {kml_base}")
    print(f"Por cultura    : {args.por_cultura}")
    print(f"Paralelo       : {args.paralelo}")
    print(f"Seed           : {args.seed}")
    print(f"CSV de saída   : {saida_path}\n")

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
        resultado = processar_amostra(amostra, idx, total, args.threshold)
        if resultado:
            with _csv_lock:
                resultados.append(resultado)
                concluidos += 1
                print(f"  >> Concluídos: {concluidos}/{total}")
            salvar_csv(resultados, saida_path)
        return resultado

    with ThreadPoolExecutor(max_workers=args.paralelo) as executor:
        futures = {
            executor.submit(_worker, (idx, a)): idx
            for idx, a in enumerate(amostras, 1)
        }
        for future in as_completed(futures):
            future.result()

    if not resultados:
        print("\nNenhum resultado obtido. Verifique conectividade com o serviço.")
        sys.exit(1)

    metricas = calcular_metricas(resultados, args.threshold)
    imprimir_relatorio(resultados, metricas)
    print(f"\nResultados salvos em: {saida_path}")


if __name__ == "__main__":
    main()
