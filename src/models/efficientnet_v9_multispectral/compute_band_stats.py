"""
Calcula `BAND_MEAN` e `BAND_STD` por banda no dataset TIFF do V9.

Use após terminar o `pipeline_tiff.py`. A saída deste script vai ser colada em
`dataset.py` substituindo os placeholders. Sem isso, a normalização do treino
fica errada e o modelo não converge bem.

Uso:
    python compute_band_stats.py [--pasta ./processadas_tiff_v9] [--max-amostras N]

Cobertura típica: 5000 talhões × 9 DAPs × 224×224 pixels é mais que o suficiente
para estabilizar a média/desvio. Por padrão limita a 2000 arquivos amostrados
aleatoriamente para não demorar mais que ~5 minutos.
"""

import argparse
import os
import random

import numpy as np

from tiff_io import ler_tiff


def amostrar_arquivos(pasta: str, max_arquivos: int, seed: int):
    arquivos = []
    for nome in os.listdir(pasta):
        if nome.endswith('.tiff'):
            arquivos.append(os.path.join(pasta, nome))
    if not arquivos:
        raise SystemExit(f"Nenhum .tiff em {pasta}")
    random.Random(seed).shuffle(arquivos)
    return arquivos[:max_arquivos]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pasta',      default='./processadas_tiff_v9')
    parser.add_argument('--max-amostras', type=int, default=2000)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    arquivos = amostrar_arquivos(args.pasta, args.max_amostras, args.seed)
    print(f"Computando stats em {len(arquivos)} arquivos de {args.pasta}...")

    # Acumuladores online (Welford). Computamos só em pixels não-zero
    # (dentro do polígono do talhão).
    contagem = None
    soma     = None
    soma_sq  = None

    for i, path in enumerate(arquivos):
        try:
            img = ler_tiff(path)            # (H, W, C)
        except Exception as ex:
            print(f"  WARN: pulando {path}: {ex}")
            continue

        H, W, C = img.shape
        if contagem is None:
            contagem = np.zeros(C, dtype=np.float64)
            soma     = np.zeros(C, dtype=np.float64)
            soma_sq  = np.zeros(C, dtype=np.float64)

        flat = img.reshape(-1, C)
        # Pixel é "interior" se ALGUMA banda for não-zero
        interior = (flat != 0).any(axis=1)
        if interior.sum() == 0:
            continue
        validos = flat[interior]            # (N, C)

        contagem += validos.shape[0]
        soma     += validos.sum(axis=0)
        soma_sq  += (validos ** 2).sum(axis=0)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(arquivos)} processados")

    media = soma / contagem
    var   = soma_sq / contagem - media ** 2
    std   = np.sqrt(np.clip(var, 0, None))

    print()
    print("=" * 60)
    print("Cole isto em dataset.py:")
    print()
    print(f"# Computado em {len(arquivos)} amostras de {args.pasta}")
    print(f"# Total de pixels interiores: {int(contagem[0]):,}")
    print(f"BAND_MEAN = np.array({list(np.round(media, 6))}, dtype=np.float32)")
    print(f"BAND_STD  = np.array({list(np.round(std, 6))}, dtype=np.float32)")
    print("=" * 60)


if __name__ == '__main__':
    main()
