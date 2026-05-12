"""
I/O e máscaramento de TIFF multi-band para o V9.

A função `aplicar_mascara_tiff` é o equivalente do `aplica_mascara` do V7/V8,
mas opera em FLOAT32 multi-band em vez de uint8 RGB. A geração da máscara em
si reaproveita `cria_mascara` do `processamento_imagens.py`.

Dependência:
    pip install tifffile
    (a alternativa rasterio é ~10x mais pesada e só compensa se você precisar
     do georreferenciamento — para este pipeline, tifffile basta.)
"""

import os

import cv2
import numpy as np
import tifffile


def ler_tiff(path: str) -> np.ndarray:
    """
    Lê um TIFF multi-band FLOAT32 do Sentinel Hub Processing API.

    Sentinel Hub entrega no layout (H, W, C). Esta função normaliza para
    sempre retornar (H, W, C), mesmo em casos atípicos.
    """
    img = tifffile.imread(path)
    if img.ndim == 2:
        # Single-band: adiciona dim de canal
        img = img[:, :, None]
    elif img.ndim == 3 and img.shape[0] < img.shape[2] and img.shape[0] <= 12:
        # Layout (C, H, W) detectado (heurística: poucos canais e shape[0] < shape[2])
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.float32, copy=False)


def salvar_tiff(path: str, arr: np.ndarray, compression: str = 'deflate') -> None:
    """
    Salva array (H, W, C) FLOAT32 como TIFF.
    Compressão deflate dá ~3-4x redução com leitura rápida.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.ndim != 3:
        raise ValueError(f"Esperado array (H, W, C); recebi shape {arr.shape}")
    tifffile.imwrite(path, arr.astype(np.float32, copy=False), compression=compression)


def aplicar_mascara_tiff(
    tiff_path: str,
    mascara_path: str,
    saida_path: str,
    erosao_iter: int = 5,
    kernel_tam: int = 5,
) -> bool:
    """
    Aplica máscara binária a um TIFF multi-band, zerando pixels fora do polígono.

    Diferenças vs `aplica_mascara` do V7/V8:
      - Trabalha em FLOAT32 e preserva todas as bandas.
      - NÃO redimensiona a imagem para o tamanho da máscara (faz o oposto:
        redimensiona a máscara para o tamanho da imagem). Isso preserva a
        resolução original entregue pelo Sentinel Hub e mantém o aspect ratio.

    A máscara já deve existir em `mascara_path` (gerada por `cria_mascara`
    do `processamento_imagens.py`). Se não existir, retorna False sem
    levantar exceção, para o caller decidir o que fazer.

    Returns:
        True se gravou; False se a máscara não existe ou o TIFF é inválido.
    """
    if not os.path.exists(mascara_path):
        return False

    img = ler_tiff(tiff_path)              # (H, W, C) FLOAT32
    H, W, C = img.shape

    mascara = cv2.imread(mascara_path, cv2.IMREAD_GRAYSCALE)
    if mascara is None:
        return False

    # Reamostra a máscara para o tamanho da imagem (não o contrário).
    mascara = cv2.resize(mascara, (W, H), interpolation=cv2.INTER_NEAREST)

    # cria_mascara() do V7 grava o talhão como PRETO (0) sobre fundo branco (255).
    # Para usar como filtro positivo, invertemos: talhão -> 255, exterior -> 0.
    mascara_inv = cv2.bitwise_not(mascara)
    kernel = np.ones((kernel_tam, kernel_tam), np.uint8)
    mascara_erodida = cv2.erode(mascara_inv, kernel, iterations=erosao_iter)

    # Boolean mask: True onde é talhão válido
    dentro = mascara_erodida.astype(bool)
    fora = ~dentro

    # Zera todas as bandas fora do polígono (broadcast em C)
    img_out = img.copy()
    img_out[fora] = 0.0

    salvar_tiff(saida_path, img_out)
    return True


def imagem_tiff_valida(
    path: str,
    min_pixels_validos: float = 0.05,
    min_std_por_banda: float = 0.005,
) -> bool:
    """
    Validação pós-máscara para o V9 — análoga à do pipeline_daps_fixos mas
    adaptada para FLOAT32 multi-band.

    Rejeita TIFFs que:
      - falham ao ler;
      - têm menos de `min_pixels_validos` de pixels não-zero (talhão minúsculo,
        bbox vazio, máscara totalmente erodida);
      - têm desvio padrão pequeno EM TODAS as bandas (provável nuvem solidão
        ou tile vazio — uma única banda chapada não é suficiente para rejeitar
        porque algumas bandas podem ser naturalmente uniformes).
    """
    try:
        img = ler_tiff(path)
    except Exception:
        return False

    nao_zero = img > 0
    if nao_zero.mean() < min_pixels_validos:
        return False

    # std por banda, calculado SOMENTE nos pixels dentro do talhão (não-zero)
    H, W, C = img.shape
    flat = img.reshape(-1, C)
    mascara_flat = (flat > 0).any(axis=1)   # (H*W,) -> True onde alguma banda > 0
    if mascara_flat.sum() < 100:
        return False

    interior = flat[mascara_flat]            # (N, C)
    stds = interior.std(axis=0)              # (C,)
    if (stds < min_std_por_banda).all():
        return False

    return True


# ---------------------------------------------------------------------------
# Diagnóstico rápido (executar standalone)
# ---------------------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        print("Uso: python tiff_io.py <caminho.tiff>")
        sys.exit(1)
    img = ler_tiff(sys.argv[1])
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Min/Max global: {img.min():.4f} / {img.max():.4f}")
    print("Por banda:")
    for c in range(img.shape[-1]):
        b = img[..., c]
        print(
            f"  banda {c}: min={b.min():.4f} max={b.max():.4f} "
            f"mean={b.mean():.4f} std={b.std():.4f}"
        )


if __name__ == '__main__':
    main()
