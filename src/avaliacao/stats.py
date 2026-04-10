# ARROz 5791
# cafe 53093
# milho 126324
# trigo 66944
# aveia 4226
# feijao 11614
# soja 159922


import numpy as np

nome = ['arroz', 'cafe', 'milho', 'trigo', 'aveia', 'feijao', 'soja']
culturas = [5791, 53093, 126324, 66944, 4226, 11614, 159922]

media = np.mean(culturas)

desvio = np.std(culturas)

var = 2*desvio

mediana = np.median(culturas)

oitenta = media + var

print(f"desvio: {desvio}")

print(f"mediana: {mediana}")

print(f"oitenta: {oitenta}")

diferencas = []

for i in range(7):
    t = culturas[i]
    diferenca = media - t
    if diferenca < 0:
       diferenca = diferenca * -1
    diferencas.append(diferenca)

    # print(f"nome: {nome[i]}, diferenca: {diferenca}")

print(f"diferenca media: {np.mean(diferencas)}")

# O maximo de desvio que podemos chegar e a metada do menor valor da lista
menor_valor = np.min(culturas)
desvio_limite = menor_valor / 2
maior_valor = np.max(culturas)


for i in range(50000):  # limite de iterações para segurança
    std_atual = np.std(culturas)
    if std_atual <= desvio_limite:
        break

    media = np.mean(culturas)
    for j in range(len(culturas)):
        # Se o valor for maior que a média, reduz um pouco
        if culturas[j] > media:
            culturas[j] -= 10.0  # passo de redução (pode ajustar)


print(f"desvio_limite: {desvio_limite}")
print("Lista ajustada:", culturas)
print("Desvio padrão final:", std_atual)

# data augmentation
nova_lista_ini = [5791, 7243.0, 7244.0, 7244.0, 4226, 7244.0, 11712.0]

# Primeiro data augmentation forma via sintetico

# Segundo data augmentation vai ser coletar amostras em diferentes dadtas
# [5791, 53093, 126324, 66944, 4226, 11614, 159922]
# ['arroz', 'cafe', 'milho', 'trigo', 'aveia', 'feijao', 'soja']
# [5791, 7243.0, 7244.0, 7244.0, 4226, 7244.0, 11712.0]

# Arroz - usar 100% do disponivel e mais 20% sintetico
# primeira amostrar com 15 dias - 50%
# segunda com 30 dias - 50%
# terceira com 95 dias - 50%
# o total vai ser as 5791 + 1150 sinteticas + 2890 de overlap, total 9831

# Café - Não usar sintetica
# amostra igual a 2460 imagens nos primeiros 30 dias
# amostra igual a 2460 imagens nos primeiros 50 dias
# amostra igual a 2460 imagens nos primeiros 75 dias
# amostra igual a 2460 imagens nos primeiros 100 dias, com total de 9840

# Milho - Não usar sintetica
# amostra igual a 3280 imagens nos primeiros 21 dias
# amostra igual a 3280 imagens nos primeiros 31 dias
# amostra igual a 3280 imagens nos primeiros 56 dias, com total de 9840

# Trigo - Não usar sintetica
# amostra igual a 3280 imagens nos primeiros 26 dias
# amostra igual a 3280 imagens nos primeiros 32 dias
# amostra igual a 3280 imagens nos primeiros 47 dias, com total de 9840

# Aveia - usar 100% do disponivel e mais 20% sintetico
# amostrar com 29 dias - 50%
# segunda com 44 dias - 50%
# terceira com 64 dias - 50%
# o total vai ser as 4226 + 845 sinteticas + 2113 de overlap, total 7184

# Feijao - Não usar sintetica
# amostra igual a 3280 imagens nos primeiros 21 dias
# amostra igual a 3280 imagens nos primeiros 31 dias
# amostra igual a 3280 imagens nos primeiros 56 dias, com total de 9840

# Soja - Não usar sintetica
# amostra igual a 3280 imagens nos primeiros 21 dias
# amostra igual a 3280 imagens nos primeiros 31 dias
# amostra igual a 3280 imagens nos primeiros 56 dias, com total de 9840

def lista_datas(cultura):
    culturas = {
        "feijao": [21, 31, 56],
        "soja": [21, 31, 56],
        "aveia": [29, 44, 64],
        "trigo": [26, 32, 47],
        "milho": [21, 31, 56],
        "cafe": [30, 50, 75, 100],
        "arroz": [15, 30, 95]
    }
    return culturas[cultura]

