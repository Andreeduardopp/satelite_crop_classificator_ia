# Pipeline de Processamento de Imagens de Satélite

Este documento descreve o fluxo completo do pipeline, desde o download das imagens do Sentinel Hub até a geração da imagem processada salva em `./processadas/`.

---

## Visão Geral

```
KML (campo agrícola)
        │
        ▼
[ 1. Download via Sentinel Hub ]
        │  ./imagens/{ref}_d{dap}/{hash}/response.png
        ▼
[ 2. Criação da Máscara ]
        │  ./mascaras/mascara_{ref}_d{dap}.png
        ▼
[ 3. Aplicação da Máscara ]
        │  ./processadas/mascara_{ref}_d{dap}.png
        ▼
[ 4. Envio ao ML Server ]
        │  vetor de sigmoides por imagem
        ▼
[ 5. Persistência no DataFrame / SQLite ]
```

---

## Entradas

| Arquivo | Descrição |
|---|---|
| `dataframe_processado.csv` | Tabela com todos os campos a processar |
| `./culturas/**/*.kml` | Polígono geográfico de cada talhão |

### Colunas relevantes do dataframe

| Coluna | Exemplo | Descrição |
|---|---|---|
| `cultura` | `arroz` | Cultura plantada |
| `ref_infra_v` | `a3f1b2c4_v` | Identificador único da imagem |
| `data` | `2024-01-01` | Data de plantio |
| `path` | `/home/.../ARROZ_517267258-1_...kml` | Caminho do KML |

---

## Etapa 1 — Download da Imagem (`processamento_sentinel_Hub.py`)

**Função:** `request_sentinel_hub(data, caminho_kml, r_imagem, sentinel_layer='BANDAS_RBN', png=True)`

### O que acontece

1. Lê o arquivo KML e extrai as coordenadas do polígono.
2. Calcula o bounding box (min/max de lat e lon).
3. Define uma janela de tempo: `[data - 5 dias, data]`.
4. Cria um `WcsRequest` para o Sentinel-2 L2A com:
   - Layer: `BANDAS_RBN` (Infravermelho próximo + Red + Blue — índices de vegetação)
   - Resolução: `10m x 10m`
   - Formato: PNG (ou TIFF)
5. Salva as imagens retornadas em `./imagens/{r_imagem}/`.
6. Ordena as imagens por data de modificação e filtra pela menor cobertura de nuvens (`analisar_cobertura_de_nuvens()`).
7. Retorna o nome da subpasta (`ASSET_ID`), o caminho da melhor imagem e a data.

### Saída

```
./imagens/{ref}_d{dap}/
    └── {hash}/
            └── response.png   ← imagem bruta do satélite (recorte do bounding box)
```

### Como o nome da imagem é formado

O pipeline em `pipeline.py` monta o identificador assim:

```python
imagem_mask = ref_infra_v + '_d' + str(dap)
# Exemplo: "a3f1b2c4_v_d30"  (campo a3f1b2c4, 30 dias após plantio)
```

Os DAPs (dias após plantio) são definidos por cultura:

| Cultura | DAPs |
|---|---|
| Arroz | 15, 30, 95 |
| Café | 30, 50, 75, 100 |
| Milho / Feijão / Soja | 21, 31, 56 |
| Trigo | 26, 32, 47 |
| Aveia | 29, 44, 64 |

---

## Etapa 2 — Criação da Máscara (`processamento_imagens.py`)

**Função:** `cria_mascara(nome_kml, caminho_destino)`

### O que acontece

1. Chama `geo_json(kml)` (de `google_engine.py`) que usa BeautifulSoup para extrair as coordenadas de todos os polígonos do KML, incluindo buracos internos (`innerBoundaryIs`).
2. Plota o polígono preenchido de **preto** sobre fundo branco usando matplotlib.
3. Remove eixos e margens para que a imagem ocupe o quadro completo.
4. Salva o resultado como PNG.

### Saída

```
./mascaras/
    └── mascara_{ref}_d{dap}.png   ← polígono do talhão em preto, resto branco
```

A máscara tem o mesmo aspect ratio do polígono KML, não da imagem do satélite — o redimensionamento acontece na etapa seguinte.

---

## Etapa 3 — Aplicação da Máscara (`processamento_imagens.py`)

**Função:** `aplica_mascara(nome_mascara, caminho_fonte, caminho_destino, caminho_kml)`

### O que acontece passo a passo

```
Imagem bruta (response.png)          Máscara (mascara_*.png)
         │                                      │
         │                         cria_mascara() → gera/atualiza a máscara
         │                                      │
         └──────── cv2.resize ─────────────────►│  (imagem redimensionada p/ dim. da máscara)
                                                │
                              cvtColor → GRAY   │  (máscara para escala de cinza)
                                                │
                              bitwise_not       │  (inverte: talhão vira branco, resto preto)
                                                │
                              erode 5x5 / 5x   │  (contrai bordas para remover artefatos)
                                                │
                              bitwise_or ───────┘  (aplica máscara na imagem: zera pixels fora do talhão)
                                                │
                                         cv2.imwrite
                                                │
                                    processadas/mascara_*.png
```

### Detalhes das operações OpenCV

| Operação | Parâmetros | Efeito |
|---|---|---|
| `cv2.resize` | dim. da máscara | Alinha imagem e máscara pixel-a-pixel |
| `cvtColor(BGR→GRAY)` | — | Converte máscara para 1 canal |
| `bitwise_not` | — | Inverte: área do talhão fica branca (255) |
| `erode` | kernel 5×5, 5 iterações | Remove pixels de borda que vazam fora do polígono |
| `bitwise_or(imagem, imagem, mask=erosao)` | — | Mantém apenas os pixels dentro do talhão; o resto vai a zero (preto) |

### Saída

```
./processadas/
    └── mascara_{ref}_d{dap}.png   ← imagem do satélite recortada ao talhão
```

---

## Etapa 4 — Envio ao ML Server (`pipeline.py`)

**Função:** `request_mlserver(ref_infra, caminho_imagem)`

A imagem processada é enviada via `multipart/form-data` para:

```
POST http://localhost:8080/sigmoides/
```

Com os campos:

| Campo | Valor |
|---|---|
| `ref_infra_v` | identificador da imagem |
| `mes` | mês normalizado [0,1] |
| `path_image` | caminho da imagem processada |

O servidor retorna um vetor de sigmoides que representa a predição de cada modelo CNN para aquele campo/data.

---

## Etapa 5 — Persistência

Os resultados são salvos de volta no `dataframe_processado_final.csv` e/ou no banco SQLite `dados.db` (tabela `culturas`) com as colunas:

| Coluna | Conteúdo |
|---|---|
| `area` | Área do talhão em m² |
| `imagens_baixadas` | Lista de caminhos das imagens brutas |
| `imagens_processadas` | Lista de caminhos das imagens em `./processadas/` |
| `sigmoides_iv` | Vetor de sigmoides retornado pelo ML server |

---

## Estrutura de Diretórios

```
src/
├── imagens/
│   └── {ref}_d{dap}/          ← criado por request_sentinel_hub()
│       └── {hash}/
│           └── response.png   ← imagem bruta
├── mascaras/
│   └── mascara_{ref}_d{dap}.png  ← criado por cria_mascara()
└── processadas/
    └── mascara_{ref}_d{dap}.png  ← saída final de aplica_mascara()
```

---

## Exemplo de Execução Completa

Para um campo de **arroz** com `ref_infra_v = "a3f1b2c4_v"` e plantio em `2024-01-01`:

| DAP | Janela de busca Sentinel | Imagem baixada | Processada |
|---|---|---|---|
| 15 dias | 2024-01-11 a 2024-01-16 | `imagens/a3f1b2c4_v_d15/{hash}/response.png` | `processadas/mascara_a3f1b2c4_v_d15.png` |
| 30 dias | 2024-01-26 a 2024-01-31 | `imagens/a3f1b2c4_v_d30/{hash}/response.png` | `processadas/mascara_a3f1b2c4_v_d30.png` |
| 95 dias | 2024-04-05 a 2024-04-10 | `imagens/a3f1b2c4_v_d95/{hash}/response.png` | `processadas/mascara_a3f1b2c4_v_d95.png` |

Cada imagem em `processadas/` contém apenas os pixels dentro do polígono do talhão, com o restante zerado (preto).
