# Análise Técnica — Classificação de Culturas por Satélite

## Fase 1: Definição do Problema

Após uma primeira semana focada na análise dos dados obtidos e no estudo do estado da arte, avançamos para a segunda fase: a **definição do problema**. Comparando com outros estudos da área, observo que o nosso caso apresenta algumas particularidades importantes:

1. **Escala geográfica (múltiplos estados):** Estamos lidando com uma área territorial gigantesca. Isso desafia a generalização do modelo, pois ele precisará se adequar à diversidade de climas e níveis de luminosidade que variam significativamente entre os estados ao longo do ano.
2. **Sobreposição de culturas:** Trabalhamos com diversas culturas agrícolas que apresentam características muito semelhantes e que, frequentemente, se sobrepõem, dificultando a diferenciação.
3. **Resolução vs. Volume de requisições:** Atualmente, os modelos de reconhecimento de culturas por imagens de satélite que apresentam boa acurácia utilizam resoluções altas (3m e 10m) para a inferência. Esse nível de detalhe aumenta drasticamente o volume de requisições ao Sentinel, o que torna fundamental o estudo de abordagens alternativas para a coleta de dados.

Com base nesse cenário, defini o seguinte planejamento:

1. **Mapeamento de Desafios** — Identificar os principais obstáculos na classificação de cada cultura, utilizando os artigos lidos como referência.
2. **Estudo das Fases Fenológicas** — Analisar os estágios fenológicos de cada cultura, com base em materiais da Embrapa e outras fontes confiáveis.
3. **Definição de Parâmetros** — Criar uma relação detalhada dos parâmetros necessários para capturar as particularidades de cada cultura.
4. **Desenvolvimento do Pipeline de Dados** — Desenvolver o código do pipeline responsável por extrair e recuperar esses parâmetros para cada arquivo KML.

---

## Fase 2: Mapeamento de Desafios

Vamos classificar **7 tipos de culturas**:

| Cultura | Tipo | Safra | Principais Regiões |
|---|---|---|---|
| **SOJA** | Leguminosa | Verão (Out–Mar) | PR, RS, MT, GO |
| **MILHO** | Cereal | Verão (Out–Mar) | GO, MG, PR, MT |
| **ARROZ** | Cereal (irrigado) | Verão (Out–Mar) | RS |
| **FEIJÃO** | Leguminosa | Verão/Inverno | PR, MG, BA |
| **TRIGO** | Cereal de inverno | Inverno (Mai–Set) | PR, RS |
| **AVEIA** | Cereal de inverno | Inverno (Mai–Set) | PR, RS |
| **CAFÉ** | Perene | Ano todo | MG, SP, ES |

### Desafios Centrais

#### Trigo vs. Aveia

Assinatura espectral gêmea: ambos são cereais de inverno, plantados exatamente na mesma época (maio a julho) e nas mesmas regiões (Sul do Brasil). O ciclo de esverdeamento e senescência deles produz curvas de NDVI e EVI praticamente idênticas.

- O modelo anterior já mostrava sinais de dificuldade na diferenciação dessas culturas.

#### Soja vs. Milho

Ambas são plantadas na mesma janela (outubro a dezembro) e têm picos de vigor vegetativo na mesma época. Nos estágios iniciais (emergência), a resposta espectral do solo nu misturado com as pequenas folhas é muito parecida.

- Um desafio, porém, menor que os outros — com o aumento das amostras, quase certamente conseguimos resolver essa questão.

#### Feijão

Ciclo muito curto com múltiplas safras e alta variabilidade, fazendo com que seja necessária uma abordagem diferente para a requisição de dados sobre essa cultura.

#### Feijão e Aveia

Poucos trabalhos na literatura fazem o estudo de utilização de machine learning para inferência dessas culturas, o que intrinsecamente as torna desafiadoras.

---

## Fase 3: Análise Fenológica

Antes de tudo, precisamos fazer uma análise das fases fenológicas de cada cultura para planejar como a pipeline de recuperação dos dados deve funcionar.

### SOJA (Glycine max)

#### Escala Fenológica de Referência

A fenologia da soja segue a escala de **Fehr & Caviness (1977)**, adaptada pela Embrapa Soja para as condições brasileiras.

| Fase | Estádios | Duração típica (Brasil) | Descrição |
|---|---|---|---|
| Germinação | VE | 5–10 dias | Semente embebe água, radícula emerge |
| Emergência | VE–VC | 7–15 dias | Cotilédones acima do solo |
| Vegetativo | V1–Vn | 25–40 dias | Emissão de nós no caule principal |
| Florescimento | R1–R2 | 15–25 dias | Primeira flor aberta até plena floração |
| Form. de vagens | R3–R4 | 15–20 dias | Vagens visíveis com 5mm até 2cm |
| Enchimento | R5–R6 | 25–35 dias | Grão perceptível até grão cheio |
| Maturação | R7–R8 | 15–30 dias | Amarelecimento até maturação plena |

**Ciclo total:** 100–150 dias (grupos de maturidade 5.0–8.0 predominam no Brasil)

#### Referências

- **Embrapa Soja.** Tecnologias de Produção de Soja — Região Central do Brasil. Sistemas de Produção, nº 17. Londrina, PR.
- **Fehr, W.R.; Caviness, C.E.** Stages of soybean development. Special Report 80. Iowa State University, 1977.
- **Embrapa Soja.** Ecofisiologia da Soja. Circular Técnica 48, 2007.

---

### MILHO (Zea mays)

#### Escala Fenológica de Referência

A fenologia do milho segue a escala de **Ritchie et al. (1993)**, amplamente adotada pela Embrapa Milho e Sorgo.

| Fase | Estádios | Duração típica (Brasil) | Descrição |
|---|---|---|---|
| Germinação | VE | 5–10 dias | Coleóptilo emerge do solo |
| Emergência | VE–V3 | 10–20 dias | Primeira a terceira folha visível |
| Vegetativo | V4–VT | 30–40 dias | Expansão foliar, diferenciação da espiga |
| Pendoamento | VT | 2–5 dias | Pendão visível no topo da planta |
| Florescimento | R1 (embonecamento) | 5–10 dias | Estilo-estigmas (cabelo) visíveis |
| Enchimento | R2–R5 | 30–40 dias | Grão leitoso → grão farináceo duro |
| Maturação | R6 | 15–25 dias | Camada preta, ponto de maturidade fisiológica |

**Ciclo total:** 120–160 dias (híbridos precoces a tardios)

#### Referências

- **Embrapa Milho e Sorgo.** Cultivo do Milho. Sistemas de Produção, nº 1. Sete Lagoas, MG.
- **Ritchie, S.W.; Hanway, J.J.; Benson, G.O.** How a corn plant develops. Special Report 48. Iowa State University, 1993.
- **Magalhães, P.C.; Durães, F.O.M.** Fisiologia da Produção de Milho. Circular Técnica 76, Embrapa Milho e Sorgo, 2006.

---

### FEIJÃO (Phaseolus vulgaris)

#### Escala Fenológica de Referência

A fenologia do feijão segue a escala de **Fernández et al. (1986)**, adotada pela Embrapa Arroz e Feijão.

| Fase | Estádios | Duração típica (Brasil) | Descrição |
|---|---|---|---|
| Germinação | V0 | 3–7 dias | Embebição e protrusão da radícula |
| Emergência | V1–V2 | 5–10 dias | Cotilédones acima do solo, folhas primárias |
| Vegetativo | V3–V4 | 15–20 dias | Primeira folha trifoliolada até ramificação |
| Florescimento | R5–R6 | 10–20 dias | Pré-floração até plena floração |
| Form. de vagens | R7 | 10–15 dias | Primeira vagem visível |
| Enchimento | R8 | 15–20 dias | Enchimento das vagens |
| Maturação | R9 | 10–20 dias | Mudança de cor das vagens, secagem |

**Ciclo total:** 70–100 dias (cultivares do grupo I ao III)

**Nota:** O feijão tem um dos ciclos mais curtos entre as culturas do projeto. As janelas são comprimidas para capturar sua fenologia acelerada.

#### Referências

- **Embrapa Arroz e Feijão.** Informações Técnicas para o Cultivo do Feijoeiro-Comum na Região Central-Brasileira. Circular Técnica 272. Santo Antônio de Goiás, GO.
- **Fernández, F.; Gepts, P.; López, M.** Etapas de desarrollo de la planta de frijol común (Phaseolus vulgaris L.). CIAT, 1986.
- **Araújo, A.P.; Teixeira, M.G.** Fases de desenvolvimento do feijoeiro e demanda de nutrientes. Embrapa Agrobiologia, 2003.

---

### ARROZ (Oryza sativa)

#### Escala Fenológica de Referência

A fenologia do arroz segue a escala de **Counce et al. (2000)**, adaptada pela Embrapa Clima Temperado para arroz irrigado no RS.

| Fase | Estádios | Duração típica (Brasil — irrigado RS) | Descrição |
|---|---|---|---|
| Germinação | S0–S3 | 7–15 dias | Semente a coleóptilo emergido |
| Emergência | V1–V3 | 10–20 dias | Primeira a terceira folha no colmo principal |
| Vegetativo | V4–Vn | 30–45 dias | Perfilhamento ativo, expansão do dossel |
| Reprodutivo | R0–R4 | 25–35 dias | Iniciação da panícula até antese |
| Enchimento | R5–R8 | 25–30 dias | Grão leitoso até grão maduro |
| Maturação | R9 | 15–25 dias | Grão maduro, secagem na panícula |

**Ciclo total:** 120–150 dias (cultivares de ciclo médio a longo, irrigado)

**Nota:** O arroz irrigado no RS apresenta NDWI elevado na fase baseline por causa da lâmina d'água, o que é uma feição discriminativa importante para esta cultura.

#### Referências

- **Embrapa Clima Temperado.** Arroz Irrigado: Recomendações Técnicas da Pesquisa para o Sul do Brasil. Sistemas de Produção. Pelotas, RS.
- **Counce, P.A.; Keisling, T.C.; Mitchell, A.J.** A uniform, objective, and adaptive system for expressing rice development. Crop Science, v. 40, p. 436–443, 2000.
- **SOSBAI.** Arroz Irrigado: Recomendações Técnicas da Pesquisa para o Sul do Brasil. Reunião Técnica da Cultura do Arroz Irrigado, Cachoeirinha, RS.

---

### TRIGO (Triticum aestivum)

#### Escala Fenológica de Referência

A fenologia do trigo segue a escala de **Zadoks et al. (1974)**, adotada pela Embrapa Trigo como referência para o Brasil.

| Fase | Estádios Zadoks | Duração típica (Brasil — Sul) | Descrição |
|---|---|---|---|
| Germinação | 00–09 | 5–10 dias | Semente seca até coleóptilo emerge |
| Emergência | 10–13 | 10–20 dias | Primeira a terceira folha no colmo principal |
| Perfilhamento | 20–29 | 15–30 dias | Emissão de perfilhos (afilhos) |
| Alongamento | 30–39 | 15–25 dias | Elongação de entrenós, emborrachamento |
| Espigamento | 50–59 | 5–15 dias | Espiga visível até antese |
| Enchimento | 70–79 | 25–35 dias | Grão leitoso até grão farináceo |
| Maturação | 80–92 | 15–25 dias | Início maturação até ponto de colheita |

**Ciclo total:** 110–145 dias (cultivares precoces a tardias, plantio Mai–Jun no PR/RS)

#### Referências

- **Embrapa Trigo.** Informações Técnicas para Trigo e Triticale — Safra 2020. Passo Fundo, RS.
- **Zadoks, J.C.; Chang, T.T.; Konzak, C.F.** A decimal code for the growth stages of cereals. Weed Research, v. 14, p. 415–421, 1974.
- **Large, E.C.** Growth stages in cereals — illustration of the Feekes scale. Plant Pathology, v. 3, p. 128–129, 1954.
- **Embrapa Trigo.** Fenologia do Trigo: Caracterização dos Estádios de Desenvolvimento. Documentos Online 149, 2014.

---

### AVEIA (Avena sativa / Avena strigosa)

#### Escala Fenológica de Referência

A fenologia da aveia segue a mesma escala de **Zadoks et al. (1974)** usada para trigo, com ajustes de duração conforme a Embrapa Trigo e a pesquisa gaúcha.

| Fase | Estádios Zadoks | Duração típica (Brasil — Sul) | Descrição |
|---|---|---|---|
| Germinação | 00–09 | 5–10 dias | Embebição até emergência do coleóptilo |
| Emergência | 10–13 | 10–18 dias | Primeira a terceira folha |
| Perfilhamento | 20–29 | 15–25 dias | Perfilhamento ativo (aveia perfilha mais que trigo) |
| Alongamento | 30–39 | 15–20 dias | Elongação de entrenós |
| Emissão da panícula | 50–59 | 10–20 dias | Panícula emerge da bainha (diferente de espiga no trigo) |
| Enchimento | 70–79 | 20–30 dias | Grão leitoso a farináceo |
| Maturação | 80–92 | 15–25 dias | Senescência e ponto de colheita |

**Ciclo total:** 100–140 dias (aveia branca para grão; aveia preta para cobertura pode ser mais curta)

#### Diferenciação TRIGO vs AVEIA

Embora as janelas sejam similares, existem diferenças fenológicas relevantes:

| Característica | Trigo | Aveia |
|---|---|---|
| Inflorescência | Espiga compacta | Panícula aberta |
| Altura | 70–100 cm | 60–120 cm (mais variável) |
| Perfilhamento | Moderado (3–5 perfilhos) | Intenso (5–8 perfilhos) |
| Ciclo vegetativo | Ligeiramente mais longo | Ligeiramente mais curto |
| Arquitetura do dossel | Compacto, ereto | Aberto, folhas mais laxas |

Essas diferenças estruturais podem ser capturadas pelo SAR (Sentinel-1): dosséis mais abertos geram retroespalhamento VH diferente de dosséis compactos.

#### Referências

- **Embrapa Trigo.** Indicações Técnicas para a Cultura da Aveia. Comissão Brasileira de Pesquisa de Aveia. Passo Fundo, RS.
- **Floss, E.L.** Fisiologia e manejo da aveia. In: Reunião da Comissão Brasileira de Pesquisa de Aveia. UPF, Passo Fundo, 2011.
- **Zadoks, J.C.; Chang, T.T.; Konzak, C.F.** A decimal code for the growth stages of cereals. Weed Research, v. 14, p. 415–421, 1974.

---

### CAFÉ (Coffea arabica / Coffea canephora)

#### Escala Fenológica de Referência

O café é uma cultura perene com fenologia cíclica anual. A referência principal é o modelo de **Camargo & Camargo (2001)**, adotado pela Embrapa Café, que divide o ciclo fenológico reprodutivo em 6 fases.

| Fase | Meses típicos (Brasil) | Duração | Descrição |
|---|---|---|---|
| Vegetação/Dormência | Abr–Set | ~150 dias | Crescimento de ramos, gemas dormentes (seca) |
| Florada | Set–Out | 15–30 dias | Chuvas induzem abertura floral simultânea |
| Chumbinho | Out–Dez | 60–75 dias | Frutos em expansão celular inicial |
| Expansão rápida | Dez–Mar | 90–100 dias | Crescimento rápido do fruto (acúmulo de água) |
| Granação | Mar–Mai | 60–75 dias | Enchimento com matéria seca (açúcares, lipídeos) |
| Maturação | Mai–Jul | 45–60 dias | Mudança de cor (verde → cereja), colheita |

**Ciclo reprodutivo:** ~240–270 dias (da florada à colheita)

#### Referências

- **Embrapa Café.** Sistemas de Produção — Café Arábica e Conilon. Brasília, DF.
- **Camargo, A.P.; Camargo, M.B.P.** Definição e esquematização das fases fenológicas do cafeeiro arábica nas condições tropicais do Brasil. Bragantia, v. 60, n. 1, p. 65–68, 2001.

---

## Fase 4: Definição de Parâmetros

Após a pesquisa dos estágios fenológicos, é possível iniciar um mapeamento de quais informações será interessante utilizar para diferenciar cada uma dessas culturas. Utilizar somente a imagem como parâmetro para a rede aumenta muito a quantidade e qualidade dos dados necessários para treinar um modelo.

Com base nos artigos selecionados, escolhi um caso que pode se aplicar ao nosso contexto e que traz muitos insights sobre quais informações podemos utilizar. Abaixo, um sumário de como o artigo foi desenvolvido e como se compara ao nosso cenário.

### Estudo de Referência: Classificação de Culturas com ML e Dados Declaratórios

#### Pré-processamento das Imagens de Satélite

- **Fonte de dados:** Imagens do satélite Sentinel-2 (Nível-1C), com valores de refletância no topo da atmosfera (TOA).
- **Correção atmosférica:** Dados convertidos para refletância na base da atmosfera (BOA — Nível-2A) utilizando o plugin *Sen2cor*.
- **Mascaramento:** Nuvens e sombras foram detectadas e mascaradas para eliminar anomalias nos valores dos pixels.
- **Reamostragem:** Bandas espectrais com resolução original de 20m foram reamostradas para 10m para padronizar a entrada espacial do modelo.

> **Nosso contexto:** Atualmente possuímos configuração para recuperar as imagens do Sentinel-2 com resolução de 10m por pixel.

#### Preparação dos Dados de Referência (Ground Truth)

O grande diferencial deste estudo foi o uso de parcelas declaradas pelos agricultores (FDP — *Farmer Declaration Parcels*) como dados de treinamento, contornando o alto custo da coleta manual em campo.

- **Limpeza de dados:** Inicialmente havia 87.692 parcelas, muitas com erros topológicos ou declarações incorretas. Após limpeza, restaram 16.333 parcelas.
- **Detecção de outliers:** Foram calculadas curvas de séries temporais de índices de vegetação (NDVI e EVI) para cada parcela.
- **Validação fenológica:** Curvas foram comparadas com um calendário fenológico da região. Curvas anômalas foram descartadas, resultando em 8.549 parcelas confiáveis.

> **Nosso contexto:** Podemos pular essa fase, pois já possuímos o ground truth proveniente da base de dados do BACEN (data de plantio, cultura, KML).

#### Treinamento e Classificação

Os 8.549 dados de referência foram divididos igualmente em conjuntos de treinamento (4.274) e teste (4.275). Uma classificação baseada em pixels foi realizada testando quatro algoritmos:

- Random Forest (RF)
- Support Vector Machine (SVM)
- Redes Neurais Artificiais (ANN)
- Extreme Gradient Boosting (XGBoost)

> **Nosso contexto:** Anteriormente estávamos utilizando uma EfficientNet para a inferência — uma arquitetura muito mais complexa do que as utilizadas no artigo. Porém, o artigo obtém resultados melhores que os nossos utilizando XGBoost. Esse fato tornou interessante a ideia de replicar esse experimento com nossos dados para analisar se realmente conseguimos obter melhor resultado com modelos mais simples.

#### Features do Modelo e Suas Importâncias

A classificação utilizou um total de **121 features**: 11 variáveis (10 bandas espectrais + 1 índice) extraídas ao longo de 11 fases temporais diferentes durante a safra de 2021.

**Bandas Espectrais do Sentinel-2 (10 features por imagem):**

| Grupo | Bandas | O que captura |
|---|---|---|
| Visíveis | B2 (Azul), B3 (Verde), B4 (Vermelho) | Coloração das folhas, estado de saúde, pigmentos fotossintéticos |
| Red-Edge | B5, B6, B7 | Conteúdo de clorofila e estrutura foliar — sensível a diferenças sutis entre culturas fenologicamente semelhantes |
| Infravermelho Próximo (NIR) | B8, B8A | Estrutura celular foliar, base para índices de vegetação, vigor e densidade de biomassa |
| Infravermelho de Ondas Curtas (SWIR) | B11, B12 | Conteúdo de umidade no solo e na vegetação, estresse hídrico |

> **Nosso contexto:** Anteriormente trabalhávamos com esses dados "transcritos" dentro da imagem, porém dessa forma eles são simplificados e muitas vezes a rede neural não consegue extrair as informações necessárias para caracterizar cada cultura. A abordagem de recuperá-los de forma numérica e alimentar o modelo com cada banda separadamente se mostrou mais promissora.

**Índice de Vegetação EVI:**

O EVI foi incluído como feature separada para cada fase temporal. O estudo descobriu que o EVI foi um índice crítico para o sucesso do modelo: enquanto o NDVI apresentava curvas muito parecidas para culturas de colheita dupla (como Trigo-Algodão ou Trigo-Milho), o EVI demonstrou maior capacidade de separar as assinaturas espectrais dessas culturas com estágios fenológicos sobrepostos.

> **Nosso contexto:** O EVI não era utilizado anteriormente. Foi adicionado como novo parâmetro para ajudar a rede a distinguir as culturas.

#### A Dimensão Temporal

A característica mais importante da arquitetura dos dados não foi apenas "quais" bandas foram usadas, mas **"quando"**.

O modelo utilizou dados de 11 datas diferentes (março a outubro), cobrindo todo o ciclo de plantio, crescimento e colheita. Plantas como girassol e milho podem ser espectralmente idênticas em um dia específico de julho, mas a taxa de crescimento (curva fenológica) em maio ou agosto será diferente. O modelo utiliza essa "assinatura temporal" (11 bandas × 11 imagens = 121 features) para atingir altas precisões — o XGBoost alcançou **92.14% de acurácia**.

> **Nosso contexto:** Temos uma vantagem em relação ao modelo do artigo: conhecemos a data de plantio da cultura na área. Isso permite planejar as janelas de tempo de forma mais precisa do que um modelo que não possui essa informação (e que precisa utilizar datas muito mais esparsas).

---

## Fase 5: Desenvolvimento do Pipeline e Modelo

### 5.1 Criação do Serviço de Requisição

Criar um serviço que utilizará a **Statistical API** do Sentinel Hub ao invés do protocolo **WCS (Web Coverage Service)** — um padrão OGC legado que o Sentinel Hub ainda suporta, mas não recomenda mais.

Além disso, o novo serviço incorpora: requisição do Sentinel-2 para cada fase fenológica da cultura, tratamento de erros com retry exponencial, e gerenciamento de dados nulos.

### 5.2 O Que Será Recuperado na Nova Pipeline

Ao invés de buscar a imagem, utilizamos o Sentinel Hub para calcular as informações da imagem no servidor e enviar apenas os resultados agregados.

#### Índices Ópticos (Sentinel-2) — calculados por pixel

| Índice | Fórmula | O que mede |
|--------|---------|------------|
| **NDVI** | (B08 − B04) / (B08 + B04) | Vigor vegetativo — razão entre reflectância no infravermelho próximo (NIR) e no vermelho. Faixa: −1 a +1, culturas tipicamente 0.3–0.9. |
| **EVI** | 2.5 × (B08 − B04) / (B08 + 6×B04 − 7.5×B02 + 1) | Índice de vegetação aprimorado — corrige efeitos atmosféricos e de fundo do solo. Menos saturado que o NDVI em dossel denso. |
| **NDWI** | (B03 − B08) / (B03 + B08) | Índice de diferença normalizada de água — mede o conteúdo de água foliar usando reflectância no verde vs NIR. |

#### Índices SAR (Sentinel-1 GRD) — calculados por pixel

| Índice | Fórmula | O que mede |
|--------|---------|------------|
| **VV** | 10 × log₁₀(VV) em dB | Retroespalhamento na polarização vertical-vertical. Sensível à umidade do solo e altura da cultura. |
| **VH** | 10 × log₁₀(VH) em dB | Retroespalhamento na polarização vertical-horizontal. Sensível ao espalhamento volumétrico (biomassa da cultura). |
| **CR** | VH / VV (linear) | Razão de polarização cruzada — indica estrutura da cultura e complexidade do dossel. |
| **RVI** | 4 × VH / (VV + VH) | Índice de vegetação por radar — proxy de biomassa que funciona através de nuvens. |

### 5.3 Data Engineering

#### Filtragem de Nuvens (Apenas Óptico)

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

#### Estatísticas Zonais — O Que Recebemos

O pipeline não retorna valores individuais de pixel. Em vez disso, agrega todos os pixels válidos do polígono em **5 estatísticas resumo**:

| Estatística | Significado |
|-------------|-------------|
| **mean** | Valor médio de todos os pixels no talhão |
| **median** | Percentil 50 — robusto a outliers |
| **std** | Desvio padrão — variabilidade espacial dentro do talhão |
| **p10** | Percentil 10 — representa a área com pior desempenho |
| **p90** | Percentil 90 — representa a área com melhor desempenho |

**Exemplo:** `NDVI_mean_vegetative = 0.72` significa que o NDVI médio de todos os pixels no polígono durante o estágio vegetativo foi 0.72.

#### Saída Total por Talhão

| Categoria | Quantidade |
|-----------|------------|
| Feições ópticas | 3 índices × 5 estatísticas × 6 estágios = **90** |
| Feições SAR | 4 índices × 5 estatísticas × 6 estágios = **120** |
| Metadados | field_id, crop_label, planting_date, area_hectares, latitude, longitude, stages_covered = **7** |
| **Total de colunas** | **217** |

Cada talhão produz **uma linha** no banco de dados representando seu perfil fenológico completo.

### 5.4 Por Que Usar a Statistical API em Vez de Baixar Imagens?

- **Sem transferência de imagem** — mais rápido e mais barato (~1 KB JSON vs ~400 KB raster)
- **Máscara de nuvens no lado do servidor** — sem necessidade de filtrar pixels localmente
- **Sem necessidade de armazenamento local** — números vão direto para o SQLite
- **Menor custo de Unidades de Processamento (PU)** — resposta JSON consome menos PUs que rasters completos

### 5.5 Coleta dos Dados

A pipeline roda baixando os dados de cada talhão e gravando-os no banco de dados junto com os metadados e índices. A escolha do SQLite vem pela praticidade e pela possibilidade de pausar a pipeline e continuar de onde paramos.

Inicialmente serão baixados **500 KMLs de cada cultura** como amostra de teste, para verificar se a abordagem de coleta de dados é viável antes de escalar.

### 5.6 Escolha do Modelo

Após a coleta da amostra, rodamos os modelos. A sugestão inicial foi utilizar o **XGBoost** — um modelo ensemble rápido de treinar, que retorna resultados confiáveis e verificáveis para validar a abordagem de coleta de dados.

Uma vantagem adicional do XGBoost é que ele retorna a **importância de cada feature**, possibilitando visualizar quais parâmetros são mais relevantes e realizar mais data engineering se necessário.
