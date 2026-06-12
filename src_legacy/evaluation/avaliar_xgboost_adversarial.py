"""
Teste adversarial do XGBoost — robustez a cultura declarada incorretamente.

Para cada KML cuja cultura REAL é X (extraída do nome do arquivo), envia ao
serviço declarando uma ou mais culturas. A cultura declarada (claim) afeta
o culture_key enviado ao modelo, alterando janelas fenológicas e features.

Mede:
  - Defesa: pred == true (modelo seguiu a imagem, ignorou a mentira)
  - Engano: pred == claim (modelo seguiu o contexto, ignorou a imagem)
  - Outra : pred != true e pred != claim (modelo se confundiu)

Modos:
  - controle : claim = true_culture           (sanity check)
  - mentira  : claim = uma cultura != true   (1 request por KML)
  - matriz   : claim varia entre TODAS as 7 culturas (7 requests por KML)

Uso:
  python src_legacy/evaluation/avaliar_xgboost_adversarial.py --modo matriz --por-cultura 20
"""

import argparse
import csv
import os
import random
import re
import sys
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

URL_PREDICT = "http://localhost:8008/predict/xgboost"

CULTURAS = ["ARROZ", "AVEIA", "CAFE", "FEIJAO", "MILHO", "SOJA", "TRIGO"]

PASTA_PARA_CULTURA = {
    "ARROZ":  "ARROZ",
    "AVEIA":  "AVEIA",
    "CAFE":   "CAFE",
    "FEIJAO": "FEIJAO",
    "MILHO":  "MILHO",
    "SOJA":   "SOJA",
    "TRIGO":  "TRIGO",
}

REQUEST_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Carregamento dos KMLs
# ---------------------------------------------------------------------------

def _extrair_data_plantio(nome_arquivo: str) -> str | None:
    match = re.search(r"_plantio_(\d{2})-(\d{2})-(\d{2})_", nome_arquivo)
    if not match:
        return None
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

        kmls_validos = []
        for k in kmls:
            data = _extrair_data_plantio(k.name)
            if data:
                kmls_validos.append((k, data))

        if not kmls_validos:
            print(f"AVISO: nenhum KML com data de plantio válida em {pasta}")
            continue

        selecionados = random.sample(kmls_validos, min(por_cultura, len(kmls_validos)))
        print(f"  {culture_key:<8}: {len(selecionados)} KMLs selecionados de {len(kmls_validos)} válidos ({len(kmls)} total)")

        for kml_path, data_plantio in selecionados:
            amostras.append({
                "path": str(kml_path),
                "true_culture": culture_key,
                "data": data_plantio,
            })

    random.shuffle(amostras)
    return amostras


def expandir_amostras(amostras, modo, semente):
    rnd = random.Random(semente)
    expandidas = []
    for a in amostras:
        if modo == "matriz":
            for claim in CULTURAS:
                expandidas.append({**a, "claim": claim})
        elif modo == "mentira":
            opcoes = [c for c in CULTURAS if c != a["true_culture"]]
            expandidas.append({**a, "claim": rnd.choice(opcoes)})
        elif modo == "controle":
            expandidas.append({**a, "claim": a["true_culture"]})
        else:
            raise ValueError(f"Modo desconhecido: {modo}")
    rnd.shuffle(expandidas)
    return expandidas


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def enviar_classificacao(kml_path: str, claim: str, data: str, url: str):
    with open(kml_path, "rb") as f:
        files = {"kml": (os.path.basename(kml_path), f, "application/vnd.google-earth.kml+xml")}
        dados = {"culture_key": claim, "planting_date": data}
        resp = requests.post(url, data=dados, files=files, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Processamento
# ---------------------------------------------------------------------------

def _normalizar_cultura(s: str) -> str:
    return s.upper().strip()


def processar_amostra(amostra: dict, idx: int, total: int, url: str = URL_PREDICT):
    kml_path = amostra["path"]
    true_culture = amostra["true_culture"]
    claim = amostra["claim"]
    data = amostra["data"]
    cenario = "CONTROLE" if true_culture == claim else "MENTIRA"

    print(f"\n[{idx}/{total}] [{cenario}] true={true_culture} claim={claim} | {os.path.basename(kml_path)}")

    t0 = time.time()
    try:
        resultado = enviar_classificacao(kml_path, claim, data, url)
        tempo_seg = round(time.time() - t0, 1)

        status = (resultado.get("status") or "").upper()
        if "ERRO" in status:
            print(f"  ERRO do serviço: {resultado.get('mensagem', '')} ({tempo_seg}s)")
            return None

        pred_cultura = _normalizar_cultura(resultado.get("resultado") or "")
        confianca = resultado.get("confianca_pct")
        probabilidades = resultado.get("probabilidades", {})

        prob_true = probabilidades.get(true_culture) or probabilidades.get(true_culture.lower())
        prob_claim = probabilidades.get(claim) or probabilidades.get(claim.lower())
        acerto_real = pred_cultura == true_culture
        seguiu_mentira = (pred_cultura == claim) and (true_culture != claim)

        if acerto_real:
            marcador = "DEFESA"
        elif seguiu_mentira:
            marcador = "ENGANO"
        else:
            marcador = "OUTRA"

        print(f"  pred={pred_cultura} [{marcador}] | conf={confianca} | "
              f"prob_true={prob_true} prob_claim={prob_claim} | {tempo_seg}s")

        return {
            "kml": os.path.basename(kml_path),
            "data": data,
            "true": true_culture,
            "claim": claim,
            "pred": pred_cultura,
            "confianca": confianca,
            "prob_true": prob_true,
            "prob_claim": prob_claim,
            "acerto_real": acerto_real,
            "seguiu_mentira": seguiu_mentira,
            "tempo_seg": tempo_seg,
        }

    except requests.exceptions.RequestException as exc:
        print(f"  ERRO na requisição: {exc} ({round(time.time() - t0, 1)}s)")
        return None
    except Exception as exc:
        print(f"  ERRO inesperado: {exc} ({round(time.time() - t0, 1)}s)")
        return None


# ---------------------------------------------------------------------------
# Métricas e relatório
# ---------------------------------------------------------------------------

def calcular_metricas(resultados):
    controle = [r for r in resultados if r["true"] == r["claim"]]
    mentira = [r for r in resultados if r["true"] != r["claim"]]

    metricas = {"n_total": len(resultados), "n_controle": len(controle), "n_mentira": len(mentira)}

    if controle:
        metricas["acuracia_controle"] = sum(1 for r in controle if r["acerto_real"]) / len(controle)

    if mentira:
        defesa = sum(1 for r in mentira if r["acerto_real"])
        engano = sum(1 for r in mentira if r["seguiu_mentira"])
        outra = len(mentira) - defesa - engano
        metricas["taxa_defesa"] = defesa / len(mentira)
        metricas["taxa_engano"] = engano / len(mentira)
        metricas["taxa_outra"] = outra / len(mentira)

    matriz = defaultdict(Counter)
    for r in resultados:
        matriz[(r["true"], r["claim"])][r["pred"]] += 1

    return metricas, matriz


def imprimir_relatorio(resultados, metricas, matriz):
    print("\n" + "=" * 78)
    print("RELATÓRIO ADVERSARIAL — XGBoost (true vs claim)")
    print("=" * 78)
    print(f"Total de requests : {metricas['n_total']}")
    print(f"  Controle (true==claim): {metricas['n_controle']}")
    print(f"  Mentira  (true!=claim): {metricas['n_mentira']}")

    if "acuracia_controle" in metricas:
        print(f"\nControle (sanity):")
        print(f"  Acurácia: {metricas['acuracia_controle']:.4f}")

    if "taxa_defesa" in metricas:
        print(f"\nMentira (cenário adversarial):")
        print(f"  DEFESA (pred == true)  : {metricas['taxa_defesa']:.4f}  <- modelo confia na imagem")
        print(f"  ENGANO (pred == claim) : {metricas['taxa_engano']:.4f}  <- contexto domina")
        print(f"  OUTRA  (terceira)      : {metricas['taxa_outra']:.4f}")

    print("\nMatriz (true x claim -> top predição):")
    print(f"  {'true':<10} {'claim':<10} {'top_pred':<10} {'count':>10}  outros")
    print("  " + "-" * 60)
    for (true_c, claim_c), counter in sorted(matriz.items()):
        top_pred, top_count = counter.most_common(1)[0]
        total = sum(counter.values())
        outros = {k: v for k, v in counter.items() if k != top_pred}
        outros_str = ", ".join(f"{k}={v}" for k, v in sorted(outros.items())) or "-"
        marca = "* " if top_pred == claim_c and true_c != claim_c else "  "
        print(f"{marca}{true_c:<10} {claim_c:<10} {top_pred:<10} {top_count:>3}/{total:<6} {outros_str}")
    print("  (* = predição mais frequente coincide com a mentira)")

    print("\nResumo por cultura REAL — taxa de defesa quando alguém mente:")
    for true_c in CULTURAS:
        pares = [(c, m) for (t, c), m in matriz.items() if t == true_c and c != true_c]
        if not pares:
            continue
        total = sum(sum(m.values()) for _, m in pares)
        defesa = sum(m.get(true_c, 0) for _, m in pares)
        engano = sum(m.get(claim_c, 0) for claim_c, m in pares)
        print(f"  {true_c:<8}: defesa={defesa}/{total} ({defesa/total:.2%}) | "
              f"engano={engano}/{total} ({engano/total:.2%})")
    print("=" * 78)


CSV_CAMPOS = [
    "kml", "data", "true", "claim", "pred",
    "confianca", "prob_true", "prob_claim",
    "acerto_real", "seguiu_mentira", "tempo_seg",
]
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
        description="Teste adversarial do XGBoost: envia KMLs com cultura declarada incorretamente"
    )
    parser.add_argument("--modo", choices=["matriz", "mentira", "controle"], default="matriz",
                        help="matriz: 7 claims por KML | mentira: 1 claim!=true | controle: claim=true")
    parser.add_argument("--por-cultura", type=int, default=20,
                        help="KMLs reais por cultura (default: 20)")
    parser.add_argument("--kml-base", default=None,
                        help="Diretório raiz com pastas arquivos_kml_*_sample200")
    parser.add_argument("--url", default=URL_PREDICT,
                        help=f"URL do endpoint (default: {URL_PREDICT})")
    parser.add_argument("--saida", default=None, help="Caminho do CSV de saída")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paralelo", type=int, default=3,
                        help="Requests em paralelo (default: 3)")
    args = parser.parse_args()

    url_predict = args.url

    repo_root = Path(__file__).resolve().parent.parent.parent
    kml_base = args.kml_base or str(repo_root / "arquivos_kml_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida_path = args.saida or str(
        Path(__file__).resolve().parent / f"resultados_xgboost_adversarial_{args.modo}_{ts}.csv"
    )

    print(f"Endpoint    : {url_predict}")
    print(f"KML base    : {kml_base}")
    print(f"Modo        : {args.modo}")
    print(f"Por cultura : {args.por_cultura}")
    print(f"Paralelo    : {args.paralelo}")
    print(f"Seed        : {args.seed}")
    print(f"CSV         : {saida_path}\n")

    print("Carregando amostras reais...")
    amostras_reais = carregar_amostras_kml(kml_base, args.por_cultura, args.seed)
    print(f"KMLs reais carregados: {len(amostras_reais)}")

    expandidas = expandir_amostras(amostras_reais, args.modo, args.seed)
    total = len(expandidas)
    print(f"Total de requests planejados (modo={args.modo}): {total}\n")

    if total == 0:
        print("Nenhuma amostra encontrada. Verifique --kml-base.")
        sys.exit(1)

    resultados = []
    concluidos = 0

    def _worker(idx_amostra):
        nonlocal concluidos
        idx, amostra = idx_amostra
        resultado = processar_amostra(amostra, idx, total, url=url_predict)
        if resultado:
            with _csv_lock:
                resultados.append(resultado)
                concluidos += 1
                print(f"  >> Concluídos: {concluidos}/{total}")
            salvar_csv(resultados, saida_path)
        return resultado

    with ThreadPoolExecutor(max_workers=args.paralelo) as executor:
        futures = {executor.submit(_worker, (idx, a)): idx for idx, a in enumerate(expandidas, 1)}
        for future in as_completed(futures):
            future.result()

    if not resultados:
        print("\nNenhum resultado obtido. Verifique se o serviço está rodando.")
        sys.exit(1)

    metricas, matriz = calcular_metricas(resultados)
    imprimir_relatorio(resultados, metricas, matriz)
    print(f"\nResultados salvos em: {saida_path}")


if __name__ == "__main__":
    main()
