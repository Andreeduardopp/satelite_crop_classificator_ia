"""
Teste adversarial do FocusNet v3 - robustez a cultura declarada incorretamente.

Para cada KML cuja cultura REAL e X (extraida do nome do arquivo), envia ao
servico declarando uma ou mais culturas. A cultura declarada (`claim`) afeta:
  1. Quais DAPs o pipeline usa para baixar imagens (lista_datas_cultura)
  2. O valor do tensor `dias` que alimenta a FiLM
  3. Eventualmente o `mes` de plantio dependendo de como o servico monta o input

Mede:
  - Defesa: pred == true (modelo seguiu a imagem, ignorou a mentira)
  - Engano: pred == claim (modelo seguiu o contexto, ignorou a imagem)
  - Outra : pred != true e pred != claim (modelo se confundiu)

Modos:
  - controle : claim = true_culture           (sanity, igual ao avaliar_focusnet_v3_full)
  - mentira  : claim = uma cultura != true   (1 request por KML)
  - matriz   : claim varia entre TODAS as 5 culturas (5 requests por KML)

Uso:
  python legado/avaliar_focusnet_v3_adversarial.py --modo matriz --por-cultura 20
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
# Configuracao
# ---------------------------------------------------------------------------

URL_BASE = "https://focusnet.softfocus.com.br"
TOKEN = "574cd4a215146cbb05c3d9820e5e99e28e6c4309"

URL_CLASSIFICAR = f"{URL_BASE}/classificar/v3/"
URL_CLASSIFICACAO = f"{URL_BASE}/classificacao/"

CULTURAS = ["AVEIA", "FEIJÃO", "MILHO", "SOJA", "TRIGO"]

PASTA_PARA_CULTURA = {
    "AVEIA":  "AVEIA",
    "FEIJAO": "FEIJÃO",
    "MILHO":  "MILHO",
    "SOJA":   "SOJA",
    "TRIGO":  "TRIGO",
}

# Culturas que compartilham os mesmos DAPs em lista_datas_cultura():
# soja, milho, feijao -> [21, 31, 56]
# trigo               -> [26, 32, 47]
# aveia               -> [29, 44, 64]
DAPS_POR_CULTURA = {
    "SOJA":   (21, 31, 56),
    "MILHO":  (21, 31, 56),
    "FEIJÃO": (21, 31, 56),
    "TRIGO":  (26, 32, 47),
    "AVEIA":  (29, 44, 64),
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
    """Carrega N KMLs por cultura REAL. Cada amostra retem a verdade em true_culture."""
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
        print(f"  {culture_key:<8}: {len(selecionados)} KMLs reais selecionados de {len(kmls)} disponíveis")

        for kml_path in selecionados:
            try:
                data_plantio = _extrair_data_plantio(kml_path.name)
            except ValueError as e:
                print(f"  AVISO: {e}")
                continue
            amostras.append({
                "path": str(kml_path),
                "true_culture": culture_key,
                "data": data_plantio,
            })

    random.shuffle(amostras)
    return amostras


def expandir_amostras(amostras, modo, semente):
    """Para cada amostra real, gera 1 ou 5 (amostra, claim) conforme o modo."""
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

def enviar_classificacao(kml_path: str, claim: str, data: str):
    kml_bytes = Path(kml_path).read_bytes()
    files = {"kml_file": (os.path.basename(kml_path), kml_bytes, "application/vnd.google-earth.kml+xml")}
    dados = {"culture_key": claim, "date": data, "token": TOKEN, "data_plantio": data, "cultura": claim}
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


def processar_amostra(amostra: dict, idx: int, total: int):
    kml_path = amostra["path"]
    true_culture = amostra["true_culture"]
    claim = amostra["claim"]
    data = amostra["data"]
    cenario = "CONTROLE" if true_culture == claim else "MENTIRA"
    mesmo_dap = DAPS_POR_CULTURA[true_culture] == DAPS_POR_CULTURA[claim]

    print(f"\n[{idx}/{total}] [{cenario}] true={true_culture} claim={claim} | "
          f"mesmo_DAP={mesmo_dap} | {os.path.basename(kml_path)}")

    t0 = time.time()
    try:
        resposta_inicial = enviar_classificacao(kml_path, claim, data)
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

        pred_raw = resultado.get("resultado") or ""
        pred_cultura = _normalizar_cultura(pred_raw)
        confianca = resultado.get("confianca_pct")
        probabilidades = resultado.get("probabilidades", {})

        prob_true  = probabilidades.get(true_culture.lower()) or probabilidades.get(true_culture)
        prob_claim = probabilidades.get(claim.lower())        or probabilidades.get(claim)
        acerto_real    = pred_cultura == true_culture
        seguiu_mentira = (pred_cultura == claim) and (true_culture != claim)

        if acerto_real:
            marcador = "DEFESA"
        elif seguiu_mentira:
            marcador = "ENGANO"
        else:
            marcador = "OUTRA"

        print(f"  pred={pred_cultura} [{marcador}] | conf={confianca} | "
              f"prob_true={prob_true} prob_claim={prob_claim} | {tempo_seg}s")
        print(f"  Probabilidades: {probabilidades}")

        return {
            "kml": os.path.basename(kml_path),
            "data": data,
            "true": true_culture,
            "claim": claim,
            "mesmo_dap": mesmo_dap,
            "pred": pred_cultura,
            "confianca": confianca,
            "prob_true": prob_true,
            "prob_claim": prob_claim,
            "acerto_real": acerto_real,
            "seguiu_mentira": seguiu_mentira,
            "tempo_seg": tempo_seg,
        }

    except (ValueError, RuntimeError, requests.HTTPError) as exc:
        print(f"  ERRO na requisição: {exc} ({round(time.time() - t0, 1)}s)")
        return None
    except Exception as exc:
        print(f"  ERRO inesperado: {exc} ({round(time.time() - t0, 1)}s)")
        return None


# ---------------------------------------------------------------------------
# Metricas e relatorio
# ---------------------------------------------------------------------------

def calcular_metricas(resultados):
    controle = [r for r in resultados if r["true"] == r["claim"]]
    mentira  = [r for r in resultados if r["true"] != r["claim"]]

    metricas = {"n_total": len(resultados), "n_controle": len(controle), "n_mentira": len(mentira)}

    if controle:
        metricas["acuracia_controle"] = sum(1 for r in controle if r["acerto_real"]) / len(controle)

    if mentira:
        defesa = sum(1 for r in mentira if r["acerto_real"])
        engano = sum(1 for r in mentira if r["seguiu_mentira"])
        outra  = len(mentira) - defesa - engano
        metricas["taxa_defesa"] = defesa / len(mentira)
        metricas["taxa_engano"] = engano / len(mentira)
        metricas["taxa_outra"]  = outra  / len(mentira)

        # Quebra por compartilhamento de DAPs
        mentira_mesmo  = [r for r in mentira if r["mesmo_dap"]]
        mentira_difer  = [r for r in mentira if not r["mesmo_dap"]]
        for nome, grupo in (("mesmo_dap", mentira_mesmo), ("dap_diferente", mentira_difer)):
            if not grupo:
                continue
            d = sum(1 for r in grupo if r["acerto_real"])
            e = sum(1 for r in grupo if r["seguiu_mentira"])
            metricas[f"taxa_defesa_{nome}"] = d / len(grupo)
            metricas[f"taxa_engano_{nome}"] = e / len(grupo)
            metricas[f"n_{nome}"] = len(grupo)

    matriz = defaultdict(Counter)
    for r in resultados:
        matriz[(r["true"], r["claim"])][r["pred"]] += 1

    return metricas, matriz


def imprimir_relatorio(resultados, metricas, matriz):
    print("\n" + "=" * 78)
    print("RELATÓRIO ADVERSARIAL — FOCUSNET v3 (true vs claim)")
    print("=" * 78)
    print(f"Total de requests : {metricas['n_total']}")
    print(f"  Controle (true==claim): {metricas['n_controle']}")
    print(f"  Mentira  (true!=claim): {metricas['n_mentira']}")

    if "acuracia_controle" in metricas:
        print(f"\nControle (sanity):")
        print(f"  Acurácia: {metricas['acuracia_controle']:.4f} "
              f"(deveria ser próxima do benchmark do v3)")

    if "taxa_defesa" in metricas:
        print(f"\nMentira (cenário adversarial):")
        print(f"  DEFESA (pred == true)  : {metricas['taxa_defesa']:.4f}  ← quanto maior, mais o modelo confia na imagem")
        print(f"  ENGANO (pred == claim) : {metricas['taxa_engano']:.4f}  ← quanto maior, mais a FiLM/contexto domina")
        print(f"  OUTRA  (terceira)      : {metricas['taxa_outra']:.4f}")

        print(f"\nQuebra por DAPs compartilhados:")
        if "n_mesmo_dap" in metricas:
            print(f"  mesmo DAP (soja↔milho↔feijão): "
                  f"defesa={metricas['taxa_defesa_mesmo_dap']:.4f}  "
                  f"engano={metricas['taxa_engano_mesmo_dap']:.4f}  "
                  f"(n={metricas['n_mesmo_dap']})")
            print(f"    -> aqui as imagens baixadas são as MESMAS sob qualquer claim,")
            print(f"       então engano > 0 indica viés do canal mes/FiLM.")
        if "n_dap_diferente" in metricas:
            print(f"  DAP diferente               : "
                  f"defesa={metricas['taxa_defesa_dap_diferente']:.4f}  "
                  f"engano={metricas['taxa_engano_dap_diferente']:.4f}  "
                  f"(n={metricas['n_dap_diferente']})")
            print(f"    -> claim altera as DATAS de download; viés combinado de imagem e dias/FiLM.")

    print("\nMatriz (true × claim → top predição):")
    print(f"  {'true':<10} {'claim':<10} {'top_pred':<10} {'count':>10}  outros")
    print("  " + "-" * 60)
    for (true_c, claim_c), counter in sorted(matriz.items()):
        top_pred, top_count = counter.most_common(1)[0]
        total = sum(counter.values())
        outros = {k: v for k, v in counter.items() if k != top_pred}
        outros_str = ", ".join(f"{k}={v}" for k, v in sorted(outros.items())) or "-"
        marca = "  " if true_c == claim_c else "* " if top_pred == claim_c and true_c != claim_c else "  "
        print(f"{marca}{true_c:<10} {claim_c:<10} {top_pred:<10} {top_count:>3}/{total:<6} {outros_str}")
    print("  (* = predição mais frequente coincide com a mentira → engano dominante nesse par)")

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
    "kml", "data", "true", "claim", "mesmo_dap", "pred",
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
        description="Teste adversarial do Focusnet v3: envia KMLs com cultura declarada incorretamente"
    )
    parser.add_argument("--modo", choices=["matriz", "mentira", "controle"], default="matriz",
                        help="matriz: 5 claims por KML | mentira: 1 claim≠true | controle: claim=true")
    parser.add_argument("--por-cultura", type=int, default=20,
                        help="KMLs reais por cultura (cada um expandido conforme --modo). Default: 20")
    parser.add_argument("--kml-base", default=None,
                        help="Diretório raiz com pastas arquivos_kml_*_sample200")
    parser.add_argument("--saida", default=None, help="Caminho do CSV de saída")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paralelo", type=int, default=5,
                        help="Requests em paralelo (default: 5)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    kml_base = args.kml_base or str(repo_root / "arquivos_kml_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida_path = args.saida or str(
        Path(__file__).resolve().parent / f"resultados_focusnet_v3_adversarial_{args.modo}_{ts}.csv"
    )

    print(f"Endpoint    : {URL_CLASSIFICAR}")
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
        resultado = processar_amostra(amostra, idx, total)
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
        print("\nNenhum resultado obtido. Verifique conectividade com o serviço.")
        sys.exit(1)

    metricas, matriz = calcular_metricas(resultados)
    imprimir_relatorio(resultados, metricas, matriz)
    print(f"\nResultados salvos em: {saida_path}")


if __name__ == "__main__":
    main()
