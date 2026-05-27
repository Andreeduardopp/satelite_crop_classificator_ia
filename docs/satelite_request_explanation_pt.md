# Explicação das Requisições ao Satélite

## O Que Você Envia

Um arquivo KML contendo um polígono (o limite do seu talhão). O nome do arquivo codifica os metadados:
`CULTURA_idTalhao_plantio_DD-MM-AA_colheita_DD-MM-AA.kml`

O pipeline extrai o tipo de cultura, ID do talhão, data de plantio e data de colheita a partir dessa convenção de nomenclatura.

---

## O Que É Calculado

Para cada estágio fenológico (baseline, emergence, vegetative, flowering, grain_fill, maturity), o pipeline requisita **todos os pixels dentro do seu polígono** ao Sentinel Hub para a janela de tempo correspondente.

### Índices Ópticos (Sentinel-2) — calculados por pixel

| Índice | Fórmula | O que mede |
|--------|---------|------------|
| **NDVI** | (B08 − B04) / (B08 + B04) | Vigor vegetativo — razão entre reflectância no infravermelho próximo (NIR) e no vermelho. Vegetação saudável absorve vermelho e reflete NIR fortemente. Faixa: −1 a +1, culturas tipicamente 0.3–0.9. |
| **EVI** | 2.5 × (B08 − B04) / (B08 + 6×B04 − 7.5×B02 + 1) | Índice de vegetação aprimorado — corrige efeitos atmosféricos e de fundo do solo. Menos saturado que o NDVI em dossel denso. |
| **NDWI** | (B03 − B08) / (B03 + B08) | Índice de diferença normalizada de água — mede o conteúdo de água foliar usando reflectância no verde vs NIR. |

### Índices SAR (Sentinel-1 GRD) — calculados por pixel

| Índice | Fórmula | O que mede |
|--------|---------|------------|
| **VV** | 10 × log₁₀(VV) em dB | Retroespalhamento na polarização vertical-vertical. Sensível à umidade do solo e altura da cultura. |
| **VH** | 10 × log₁₀(VH) em dB | Retroespalhamento na polarização vertical-horizontal. Sensível ao espalhamento volumétrico (biomassa da cultura). |
| **CR** | VH / VV (linear) | Razão de polarização cruzada — indica estrutura da cultura e complexidade do dossel. |
| **RVI** | 4 × VH / (VV + VH) | Índice de vegetação por radar — proxy de biomassa que funciona através de nuvens. |

### Filtragem de Nuvens (Apenas Óptico)

Antes de calcular as estatísticas ópticas, cada pixel é verificado contra a Camada de Classificação de Cena (SCL). Apenas pixels com estas classes SCL são considerados válidos:

| Código SCL | Classe |
|------------|--------|
| 2 | Pixels de área escura |
| 4 | Vegetação |
| 5 | Solo exposto |
| 6 | Água |
| 7 | Não classificado (baixa probabilidade de nuvem) |

Nuvens, sombras de nuvens, cirrus e neve são excluídos. Isso garante que os índices de vegetação reflitam condições reais do terreno, não contaminação atmosférica.

Dados SAR (Sentinel-1) não são afetados por nuvens, o que é uma de suas principais vantagens.

### Janelas SAR Expandidas (v3)

O Sentinel-1 revisita a mesma localização a cada ~12 dias no Brasil, enquanto o Sentinel-2 (óptico) tem revisita de ~5 dias. Isso significa que as janelas curtas de estágio fenológico — especialmente para culturas de ciclo rápido como FEIJAO (10–20 dias por estágio) — frequentemente não contêm nenhuma aquisição SAR.

**O problema (pipeline v2):** Janelas ópticas e SAR eram idênticas. Resultado:

| Cultura | Janela típica por estágio | Cobertura SAR |
|---------|---------------------------|---------------|
| CAFE | 30–60 dias | ~97% |
| MILHO | 25–30 dias | ~41% |
| FEIJAO | 10–20 dias | ~6% |

**A solução (pipeline v3):** Janelas SAR são expandidas simetricamente para garantir no mínimo **24 dias** de cobertura (≥ 1 passagem do Sentinel-1). As janelas ópticas permanecem inalteradas.

```
expand_sar_window(day_start, day_end, min_days=24)

Exemplo — FEIJAO emergence (0 → 10 dias, janela de 10d):
  Janela óptica: dia 0 a dia 10   (10 dias — OK para S2 com revisita de 5d)
  Janela SAR:    dia -7 a dia 17  (24 dias — garante ≥ 1 passagem S1)

Exemplo — CAFE vegetative (30 → 90 dias, janela de 60d):
  Janela óptica: dia 30 a dia 90  (60 dias — sem mudança)
  Janela SAR:    dia 30 a dia 90  (60 dias — já >= 24d, sem expansão)
```

A expansão é simétrica em relação ao ponto médio do estágio, de forma que os dados SAR permanecem centrados no período fenológico correto. Isso aceita um pequeno trade-off — uma leve sobreposição temporal entre estágios adjacentes para SAR — em troca de uma cobertura significativamente maior.

Além disso, o pipeline v3 marca cada linha processada com uma flag `sar_backfill_done`, evitando re-requisições infinitas para campos que genuinamente não possuem cobertura SAR.

---

## O Que Você Recebe de Volta

O pipeline **não** retorna valores individuais de pixel. Em vez disso, ele agrega todos os pixels válidos em todo o polígono em **5 estatísticas resumo**:

| Estatística | Significado |
|-------------|-------------|
| **mean** | Valor médio de todos os pixels no seu talhão |
| **median** | Percentil 50 — robusto a outliers |
| **std** | Desvio padrão — variabilidade espacial dentro do talhão |
| **p10** | Percentil 10 — representa a área com pior desempenho |
| **p90** | Percentil 90 — representa a área com melhor desempenho |

**Exemplo:** `NDVI_mean_vegetative = 0.72` significa que o NDVI médio de todos os pixels no seu polígono durante o estágio vegetativo foi 0.72.

### Saída Total por Talhão

| Categoria | Quantidade |
|-----------|------------|
| Feições ópticas | 3 índices × 5 estatísticas × 6 estágios = **90** |
| Feições SAR | 4 índices × 5 estatísticas × 6 estágios = **120** |
| Metadados | field_id, crop_label, planting_date, area_hectares, latitude, longitude, stages_covered = **7** |
| **Total de colunas** | **217** |

Cada talhão produz **uma linha** no banco de dados representando seu perfil fenológico completo.

---

## Tamanho do Pixel no Mundo Real

As bandas ópticas do Sentinel-2 usadas neste pipeline (B02, B03, B04, B08) têm uma resolução nativa de **10 metros por pixel**. Isso significa que cada pixel cobre uma área de **10m × 10m = 100 m²** no terreno.

O SAR do Sentinel-1 (produto GRD, modo IW) também entrega dados na resolução de **10m × 10m** após correção de terreno.

Para um talhão típico de soja de 50 hectares (500.000 m²), você obtém aproximadamente **5.000 pixels** de dados por aquisição. As estatísticas agregam essas milhares de medições em nível de pixel em um resumo compacto em nível de talhão.

---

## Por Que Usar a Statistical API em Vez de Baixar Imagens

O pipeline suporta dois modos: a **Statistical API** (agregação no lado do servidor) e a **Process API** (download de imagens raster). A Statistical API é o padrão, e aqui está o porquê:

### 1. Sem Transferência de Imagem = Mais Rápido e Mais Barato

Com a Process API, você baixa rasters GeoTIFF completos (um por índice), então calcula as estatísticas localmente. Para um talhão de 50 ha na resolução de 10m, são ~5.000 pixels × 4 bandas × float 32-bit = **~80 KB por requisição de estágio**. Multiplique por centenas de talhões e 6 estágios cada e a largura de banda se acumula.

A Statistical API calcula mean, median, std, p10, p90 **nos servidores do Sentinel Hub** e retorna apenas um pequeno JSON (~1 KB). Isso é ordens de magnitude menos dados para transferir.

### 2. Máscara de Nuvens no Lado do Servidor

Com downloads raster, seu código precisa baixar a banda SCL, construir uma máscara e filtrar pixels localmente. A Statistical API lida com a máscara de nuvens dentro do evalscript no servidor — a saída `dataMask` diz ao servidor quais pixels incluir na agregação. Você recebe estatísticas limpas sem escrever lógica de filtragem em nível de pixel.

### 3. Sem Necessidade de Armazenamento Local

Downloads raster produzem milhares de arquivos TIFF que precisam ser armazenados, organizados e limpos. A Statistical API retorna números diretamente — o pipeline os escreve direto no banco de dados SQLite. Sem arquivos intermediários, sem preocupação com espaço em disco.

### 4. Menor Custo de Unidades de Processamento (PU)

O Sentinel Hub cobra Unidades de Processamento baseadas no tamanho da saída. A Statistical API retorna uma resposta JSON pequena, que custa menos PUs do que requisitar rasters em resolução completa. Ao processar centenas de talhões em 6 estágios, essa diferença é significativa para seu orçamento mensal de PUs.

### 5. Quando Você Ainda Quer as Imagens

A Process API (download raster) ainda está disponível via `save_tiffs=True`. Isso é útil quando você precisa:
- Inspecionar visualmente um talhão específico
- Gerar mapas espaciais (ex.: mapas de variabilidade de NDVI dentro do talhão)
- Depurar valores estatísticos inesperados
- Criar figuras para relatórios ou publicações

Para o pipeline de treinamento de ML, onde você só precisa de feições numéricas, a Statistical API é a melhor escolha.

---

## Comparação com o Código Legado de Requisição

O código original de ingestão de dados (`src_legacy/data_ingestion/processamento_sentinel_Hub.py`) utilizava uma abordagem fundamentalmente diferente para obter dados de satélite. Abaixo está uma comparação detalhada do que mudou e por quê.

### Abordagem Legada — Como Funcionava

```
Arquivo KML → extrair bounding box (min/max lat/lon)
            → requisição WCS ao Sentinel Hub (download da imagem PNG/TIFF completa)
            → salvar no disco
            → análise de nuvens pós-download na imagem baixada
            → retornar apenas a primeira imagem
```

O código antigo usava o pacote Python `sentinelhub` com o protocolo **WCS (Web Coverage Service)** — um padrão OGC legado que o Sentinel Hub ainda suporta mas não recomenda mais. Uma única chamada de função (`request_sentinel_hub`) baixava uma imagem por vez para uma data específica.

### Diferenças Principais

#### 1. Bounding Box vs Geometria Real do Polígono

| Aspecto | Legado | Pipeline Atual |
|---------|--------|----------------|
| **Geometria enviada** | Bounding box (min/max das coordenadas) | Polígono completo do KML |
| **Pixels incluídos** | Todos os pixels no retângulo, incluindo área fora do talhão | Apenas pixels dentro do limite do polígono |
| **Impacto** | Talhões vizinhos, estradas e corpos d'água contaminam as estatísticas | Dados limpos em nível de talhão — sem contaminação externa |

O código legado extraía o bounding box do KML:
```python
min_x = min(coord[0] for coord in coords_list)
max_x = max(coord[0] for coord in coords_list)
bbox = BBox(bbox=[min_x, min_y, max_x, max_y], crs=CRS.WGS84)
```

Para um talhão de formato irregular, o bounding box pode incluir 30–50% de pixels que estão fora do limite real do talhão. O pipeline atual envia a geometria completa do polígono ao Sentinel Hub, que faz o recorte no lado do servidor.

#### 2. Download de Imagem vs Estatísticas no Servidor

| Aspecto | Legado | Pipeline Atual |
|---------|--------|----------------|
| **O que retorna** | Imagem raster completa (PNG ou TIFF) | JSON com 5 estatísticas (mean, median, std, p10, p90) |
| **Volume de dados** | ~100 KB–1 MB por imagem | ~1 KB JSON por estágio |
| **Processamento local** | Necessário (análise de nuvens, cálculo de índices) | Nenhum — estatísticas calculadas no servidor |
| **Armazenamento** | Milhares de arquivos de imagem no disco | Números escritos diretamente no SQLite |

O código legado baixava imagens brutas e salvava no disco:
```python
imagens = wcs_request.get_data(save_data=True)
```

O pipeline atual recebe estatísticas pré-computadas via Statistical API — sem imagens, sem I/O de disco.

#### 3. Estratégia de Máscara de Nuvens

| Aspecto | Legado | Pipeline Atual |
|---------|--------|----------------|
| **Filtro em nível de tile** | `maxcc=0.60` — rejeita tiles com >60% de cobertura de nuvens | Não necessário (mascaramento por pixel) |
| **Mascaramento por pixel** | Pós-download via função `analisar_cobertura_de_nuvens()` | No servidor via banda SCL no evalscript (`dataMask`) |
| **Problema** | Um tile com 55% de nuvens passa no filtro; o talhão pode estar 100% encoberto | Cada pixel é verificado individualmente — apenas pixels limpos entram nas estatísticas |

A abordagem legada tinha um problema em duas etapas: primeiro, só conseguia rejeitar tiles inteiros acima de 60% de cobertura de nuvens. Um tile com 55% de cobertura geral poderia ter o talhão completamente sob nuvens. Segundo, a detecção de nuvens era feita *depois* de baixar a imagem, desperdiçando largura de banda com dados inutilizáveis.

O pipeline atual aplica mascaramento por pixel baseado no SCL dentro do evalscript no servidor. Apenas pixels classificados como limpos (classes SCL 2, 4, 5, 6, 7) contribuem para as estatísticas. Pixels nublados nunca são contados.

#### 4. Estratégia Temporal

| Aspecto | Legado | Pipeline Atual |
|---------|--------|----------------|
| **Janela temporal** | Janela fixa de 5 dias a partir de uma data única | Janelas fenológicas específicas por cultura (10–60 dias cada) |
| **Mosaicamento** | Aquisição única (uma passagem) | Mosaico `leastCC` — melhor composição livre de nuvens dentro da janela |
| **Consciência fenológica** | Nenhuma — apenas "tire uma imagem nesta data" | 6 estágios alinhados à data de plantio e ao ciclo de crescimento da cultura |

O código legado requisitava dados para uma janela fixa de 5 dias:
```python
data_inicial = data - timedelta(days=5)
```

Isso significava: se o satélite não passasse sobre o talhão nesses 5 dias, ou se estivesse nublado durante a passagem, você não recebia nada. Sem fallback, sem composição.

O pipeline atual usa janelas de 15–60 dias (dependendo do estágio da cultura), com mosaico `leastCC` — o Sentinel Hub seleciona automaticamente a aquisição com menor cobertura de nuvens dentro da janela.

#### 5. Bandas e Índices

| Aspecto | Legado | Pipeline Atual |
|---------|--------|----------------|
| **Bandas** | Layer pré-configurado (`BANDAS_RBN` — provavelmente RGB+NIR) | Evalscript customizado calculando NDVI, EVI, NDWI por pixel |
| **Cálculo de índices** | Não realizado — imagem de banda bruta retornada | Calculados no servidor dentro do evalscript |
| **Dados SAR** | Não disponível | Sentinel-1 VV, VH, CR, RVI via requisição separada |

O código legado dependia de um layer pré-configurado no dashboard do Sentinel Hub (`BANDAS_RBN`). O cálculo de índices (se houvesse) precisaria ser feito localmente após o download. O pipeline atual define evalscripts que calculam NDVI, EVI e NDWI no servidor, e busca separadamente os índices SAR do Sentinel-1.

#### 6. Protocolo de API

| Aspecto | Legado | Pipeline Atual |
|---------|--------|----------------|
| **Protocolo** | WCS (Web Coverage Service) — padrão OGC legado | Process API + Statistical API (API REST do Sentinel Hub) |
| **Autenticação** | Gerenciada pelo pacote Python `sentinelhub` | Gerenciamento direto de token OAuth2 com auto-refresh |
| **Dependência do dashboard** | Necessário layer customizado configurado no dashboard web | Evalscripts autocontidos no código |

WCS é um protocolo legado que o Sentinel Hub mantém para compatibilidade retroativa. As APIs Process e Statistical são a abordagem moderna recomendada, oferecendo mais flexibilidade (evalscripts customizados, recorte por polígono, estatísticas no servidor).

#### 7. Tratamento de Erros e Resiliência

| Aspecto | Legado | Pipeline Atual |
|---------|--------|----------------|
| **Lógica de retry** | Nenhuma | Backoff exponencial com 3 tentativas em HTTP 400/429/500/502/503 |
| **Refresh de token** | Gerenciado pelo pacote `sentinelhub` | Refresh explícito a cada 250 segundos com lock thread-safe |
| **Recuperação de crash** | Nenhuma — reinicia do zero | Resumo baseado em SQLite — pula talhões já processados |
| **Rate limiting** | Nenhum | `request_delay` configurável entre talhões + requisições de estágio escalonadas |

### Resumo: O Que Mudou e Por Quê

```
LEGADO                              PIPELINE ATUAL
─────────────────────────           ─────────────────────────
Bounding box                   →    Geometria completa do polígono
Download de imagem completa    →    Estatísticas no servidor (JSON)
Análise de nuvens pós-download →    Mascaramento SCL por pixel no servidor
Janela fixa de 5 dias         →    Janelas fenológicas por cultura
Aquisição única                →    Mosaico de menor cobertura de nuvens
Layer RGB+NIR (dashboard)      →    Evalscript customizado (NDVI, EVI, NDWI)
Sem SAR                        →    Sentinel-1 VV, VH, CR, RVI
Protocolo WCS legado           →    APIs REST Process + Statistical
Sem retry / sem resumo         →    Backoff exponencial + resumo SQLite
Uma imagem por vez             →    6 estágios × 7 índices × 5 stats por talhão
```

O código legado foi projetado para inspeção visual — baixar uma imagem, olhar para ela. O pipeline atual foi projetado para extração de feições para ML — obter feições numéricas limpas ao longo de todo o ciclo de crescimento da cultura, em escala, com resiliência.
