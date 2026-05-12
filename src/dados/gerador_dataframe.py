# Lê ARROZ_519339805-1_plantio_01-12-24_colheita_01-04-25

import os
import ast
import sqlite3
import cv2
import pandas as pd
import datetime
import uuid

# from processamento_sentinel_Hub import request_sentinel_hub
# from processamento_imagens import aplica_mascara, treshold_indice
# from main import prediz_sigmoide


def get_uuid():
    return str(uuid.uuid4()).split('-')[0]

# lista_arquivos = os.listdir('pasta_kmls')

# Colocar função para montar dataframe a partir do lista arquivos

# lista_arquivos['cultura'] = 'split 0' # A cultura vai ser y^: a variavel que o modelo tem que acertar
# lista_arquivos['ref_infra_v'] = 'split 1' # Add numero aleatorio para evitar duplicidade + infra
# lista_arquivos['ref_rgb'] = 'split 1' # Add numero aleatorio para evitar duplicidade + RGB
# lista_arquivos['data'] = 'split 3' # Ajustar de acordo com o ZARC por cultura
# lista_arquivos['mes'] = 'split 3.2'
# lista_arquivos['path'] = 'string toda'

LIMITE = 20000
CHUNK_SIZE = 1000
OUTPUT_CSV = 'dataframe_20k.csv'
PASTA_CULTURAS = '/home/henrique/Projetos/treinamento/kmls/culturas'


def gerar_dataframe(pasta_culturas=PASTA_CULTURAS, output_csv=OUTPUT_CSV, limite=LIMITE):
    dataframe = []
    qtd_arquivos = 0
    primeiro_chunk = True

    for root, dirs, files in os.walk(pasta_culturas):

        for file in files:

            if qtd_arquivos >= limite:
                break

            if file.endswith('.kml'):
                path_completo = os.path.join(root, file)
                nome = os.path.splitext(file)[0]
                partes = nome.split('_')

                try:
                    if len(partes) >= 6:
                        cultura = partes[0]
                        data_str = partes[3] if partes[3] != 'nan' else partes[5]
                        data = datetime.datetime.strptime(data_str, '%d-%m-%y').date()
                        mes = data.month

                        uuid_str = get_uuid()
                        dataframe.append({
                            'cultura': cultura.lower(),
                            'ref_infra_v': uuid_str + "_v",
                            'ref_rgb': uuid_str + "_rgb",
                            'data': data,
                            'mes': mes,
                            'path': path_completo
                        })

                        qtd_arquivos += 1

                        if len(dataframe) >= CHUNK_SIZE:
                            chunk_df = pd.DataFrame(dataframe)
                            chunk_df.to_csv(output_csv, mode='w' if primeiro_chunk else 'a',
                                            header=primeiro_chunk, index=False)
                            primeiro_chunk = False
                            dataframe = []

                except Exception as e:
                    print(f"Erro em arquivo {nome}:", e)
                    continue

    if dataframe:
        chunk_df = pd.DataFrame(dataframe)
        chunk_df.to_csv(output_csv, mode='w' if primeiro_chunk else 'a',
                        header=primeiro_chunk, index=False)

    print(f"Concluído: {qtd_arquivos} registros salvos em {output_csv}")


def contar_culturas(csv=OUTPUT_CSV):
    df = pd.read_csv(csv)
    contagem = df['cultura'].value_counts()
    print(f"\nContagem de culturas em '{csv}':")
    for cultura, total in contagem.items():
        print(f"  {cultura}: {total}")
    print(f"  Total: {len(df)}")
    return contagem


def inspecionar_db(db='dados.db', tabela='culturas'):
    conn = sqlite3.connect(db)

    total = conn.execute(f"SELECT COUNT(*) FROM {tabela}").fetchone()[0]

    print(f"\n=== DB: {db} | tabela: {tabela} ===")
    print(f"  Total de registros: {total}")

    print("\n  Registros por cultura:")
    rows = conn.execute(f"SELECT cultura, COUNT(*) FROM {tabela} GROUP BY cultura ORDER BY COUNT(*) DESC").fetchall()
    for cultura, count in rows:
        print(f"    {cultura}: {count}")

    print("\n  Status de processamento:")
    todos = conn.execute(f"SELECT imagens_baixadas FROM {tabela}").fetchall()
    com_imagens = sum(
        1 for (val,) in todos
        if bool(ast.literal_eval(val) if isinstance(val, str) and val else val)
    )
    print(f"    Com imagens baixadas:    {com_imagens}")
    print(f"    Sem imagens baixadas:    {total - com_imagens}")

    print("\n  Amostra (5 registros):")
    amostra = conn.execute(f"SELECT cultura, ref_infra_v, data, area, imagens_baixadas FROM {tabela} LIMIT 5").fetchall()
    for row in amostra:
        cultura, ref, data, area, imgs = row
        imgs_parsed = ast.literal_eval(imgs) if isinstance(imgs, str) and imgs else []
        print(f"    [{cultura}] {ref} | {data} | area={area} | imagens={len(imgs_parsed)}")

    conn.close()


if __name__ == '__main__':
    gerar_dataframe()
    contar_culturas()
    inspecionar_db()


# Primeiro: Para cada kml baixar a imagem e renomear com a ref

# request_sentinel_hub()

# Segundo: Aplicar mascara

# aplica_mascara()

# Terceiro: Fazer limpeza

# treshold_indice()

# Quarto: Normalizar a imagem

# cv2.resize(imagem, 224, 224)

# Quinto: Chama os modelos com o main.py
# lista_arquivos['vetor_infra_v'] = Vai ser o resultado do quinto processo
# lista_arquivos['vetor_rgb'] = Vai ser o resultado do quinto processo

# prediz_sigmoide()

# Sexto: Salvar o dataframe completo (usado para treinar o novo modelo)