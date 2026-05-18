"""
Testa o endpoint /classificar/v3/ com um único KML e imprime o resultado.

Uso:
  python legado/avaliar_focusnet_v3.py --kml <caminho.kml> --cultura SOJA --data 2024-06-01
"""

import argparse
import json
import os
import time
import requests

URL_BASE = "https://focusnet.softfocus.com.br"
TOKEN = "574cd4a215146cbb05c3d9820e5e99e28e6c4309"

URL_CLASSIFICAR = f"{URL_BASE}/classificar/v3/"
URL_CLASSIFICACAO = f"{URL_BASE}/classificacao/"

CULTURAS = ["AVEIA", "FEIJÃO", "MILHO", "SOJA", "TRIGO"]

POLL_INTERVALO = 10
POLL_MAX_TENTATIVAS = 60
REQUEST_TIMEOUT = 30


def enviar(kml_path: str, culture_key: str, data: str):
    kml_bytes = open(kml_path, "rb").read()
    files = {"kml_file": (os.path.basename(kml_path), kml_bytes, "application/vnd.google-earth.kml+xml")}
    dados = {"culture_key": culture_key, "date": data, "token": TOKEN}
    resp = requests.post(URL_CLASSIFICAR, data=dados, files=files, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def aguardar_resultado(task_id: str):
    for tentativa in range(1, POLL_MAX_TENTATIVAS + 1):
        resp = requests.get(f"{URL_CLASSIFICACAO}{task_id}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        dados = resp.json()
        status = (dados.get("status") or "").upper()

        if "ERRO" in status:
            print(f"  Status de erro retornado: {status!r}")
            return dados

        if "SUCESSO" in status:
            return dados

        print(f"  [{tentativa}/{POLL_MAX_TENTATIVAS}] status={status!r} — aguardando {POLL_INTERVALO}s...")
        time.sleep(POLL_INTERVALO)

    print("  Timeout: resultado não chegou.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kml", required=True, help="Caminho para o arquivo KML")
    parser.add_argument("--cultura", required=True, choices=CULTURAS + ["FEIJAO"],
                        help="Cultura esperada (culture_key)")
    parser.add_argument("--data", required=True, help="Data de plantio (YYYY-MM-DD)")
    args = parser.parse_args()

    culture_key = "FEIJÃO" if args.cultura == "FEIJAO" else args.cultura

    print(f"Enviando para {URL_CLASSIFICAR}")
    print(f"  KML     : {args.kml}")
    print(f"  Cultura : {culture_key}")
    print(f"  Data    : {args.data}")

    t0 = time.time()
    resposta = enviar(args.kml, culture_key, args.data)
    print(f"\nResposta inicial:\n{json.dumps(resposta, indent=2, ensure_ascii=False)}")

    task_id = resposta.get("task_id") or resposta.get("id")
    if not task_id:
        print("Sem task_id na resposta — nada a aguardar.")
        return

    print(f"\nAguardando resultado (task_id={task_id})...")
    resultado = aguardar_resultado(str(task_id))

    if resultado is None:
        print("Sem resultado após timeout.")
        return

    print(f"\nResultado final:\n{json.dumps(resultado, indent=2, ensure_ascii=False)}")

    vetor_list = resultado.get("vetor_softmax")
    # Handle both flat list [f,f,f,f,f] and list-of-lists [[f,f,f,f,f], ...]
    if isinstance(vetor_list, list) and vetor_list:
        ultimo = vetor_list[-1] if isinstance(vetor_list[-1], list) else vetor_list
    else:
        ultimo = None
    if isinstance(ultimo, list) and len(ultimo) == len(CULTURAS):
        pred_idx = ultimo.index(max(ultimo))
        pred = CULTURAS[pred_idx]
        confianca = round(max(ultimo), 4)
        acerto = pred == culture_key
        print(f"\nTrue={culture_key} | Pred={pred} | conf={confianca} | {'CORRETO' if acerto else 'ERRADO'}")
        print(f"Vetor: { {c: round(v, 4) for c, v in zip(CULTURAS, ultimo)} }")

    print(f"\nTempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
