# Análise Fenológica das Culturas — Referências e Mapeamento no Pipeline

## 1. Introdução

Este documento detalha as fases fenológicas de cada cultura classificada neste projeto, as fontes de referência agronômica utilizadas (contexto brasileiro), e como cada fase foi mapeada nas janelas temporais do pipeline de extração de feições.

O pipeline utiliza **6 estágios genéricos** para todas as culturas:

| Estágio Pipeline | Significado agronômico |
|---|---|
| `baseline` | Pré-plantio — solo exposto ou cobertura anterior |
| `emergence` | Germinação e emergência — plântula rompe o solo |
| `vegetative` | Crescimento vegetativo — expansão foliar e perfilhamento |
| `flowering` | Florescimento/reprodução — antese e polinização |
| `grain_fill` | Enchimento de grãos — acúmulo de matéria seca |
| `maturity` | Maturação — senescência e ponto de colheita |

Cada cultura possui durações diferentes para esses estágios. As janelas foram calibradas com base na literatura agronômica brasileira.

---

## 2. SOJA (Glycine max)

### 2.1 Escala Fenológica de Referência

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

### 2.2 Mapeamento no Pipeline

```
Plantio (dia 0)
    │
    ├── baseline:   -15 a 0 dias   → solo pré-plantio
    ├── emergence:    0 a 20 dias  → VE a VC (cotilédones + V1)
    ├── vegetative:  20 a 55 dias  → V2 a Vn (expansão foliar plena)
    ├── flowering:   55 a 75 dias  → R1 a R2 (florescimento)
    ├── grain_fill:  75 a 100 dias → R3 a R6 (formação + enchimento de grãos)
    └── maturity:   100 a 130 dias → R7 a R8 (senescência + colheita)
```

### 2.3 Referências

- **Embrapa Soja.** Tecnologias de Produção de Soja — Região Central do Brasil. Sistemas de Produção, nº 17. Londrina, PR.
- **Fehr, W.R.; Caviness, C.E.** Stages of soybean development. Special Report 80. Iowa State University, 1977.
- **Embrapa Soja.** Ecofisiologia da Soja. Circular Técnica 48, 2007.

---

## 3. MILHO (Zea mays)

### 3.1 Escala Fenológica de Referência

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

### 3.2 Mapeamento no Pipeline

```
Plantio (dia 0)
    │
    ├── baseline:   -15 a 0 dias   → solo pré-plantio
    ├── emergence:    0 a 25 dias  → VE a V3 (plântula)
    ├── vegetative:  25 a 55 dias  → V4 a VT (expansão foliar + pendoamento)
    ├── flowering:   55 a 85 dias  → R1 (embonecamento + polinização)
    ├── grain_fill:  85 a 115 dias → R2 a R5 (enchimento de grãos)
    └── maturity:   115 a 140 dias → R6 (maturação + secagem)
```

### 3.3 Referências

- **Embrapa Milho e Sorgo.** Cultivo do Milho. Sistemas de Produção, nº 1. Sete Lagoas, MG.
- **Ritchie, S.W.; Hanway, J.J.; Benson, G.O.** How a corn plant develops. Special Report 48. Iowa State University, 1993.
- **Magalhães, P.C.; Durães, F.O.M.** Fisiologia da Produção de Milho. Circular Técnica 76, Embrapa Milho e Sorgo, 2006.

---

## 4. FEIJÃO (Phaseolus vulgaris)

### 4.1 Escala Fenológica de Referência

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

### 4.2 Mapeamento no Pipeline

```
Plantio (dia 0)
    │
    ├── baseline:   -15 a 0 dias  → solo pré-plantio
    ├── emergence:    0 a 10 dias → V0 a V2 (germinação rápida)
    ├── vegetative:  10 a 25 dias → V3 a V4 (crescimento vegetativo curto)
    ├── flowering:   25 a 45 dias → R5 a R6 (florescimento)
    ├── grain_fill:  45 a 65 dias → R7 a R8 (formação + enchimento de vagens)
    └── maturity:    65 a 90 dias → R9 (maturação e secagem)
```

**Nota:** O feijão tem um dos ciclos mais curtos entre as culturas do projeto. As janelas são comprimidas para capturar sua fenologia acelerada.

### 4.3 Referências

- **Embrapa Arroz e Feijão.** Informações Técnicas para o Cultivo do Feijoeiro-Comum na Região Central-Brasileira. Circular Técnica 272. Santo Antônio de Goiás, GO.
- **Fernández, F.; Gepts, P.; López, M.** Etapas de desarrollo de la planta de frijol común (Phaseolus vulgaris L.). CIAT, 1986.
- **Araújo, A.P.; Teixeira, M.G.** Fases de desenvolvimento do feijoeiro e demanda de nutrientes. Embrapa Agrobiologia, 2003.

---

## 5. ARROZ (Oryza sativa)

### 5.1 Escala Fenológica de Referência

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

### 5.2 Mapeamento no Pipeline

```
Plantio (dia 0)
    │
    ├── baseline:   -15 a 0 dias   → solo inundado pré-plantio (lâmina d'água)
    ├── emergence:    0 a 25 dias  → S0 a V3 (germinação + plântula)
    ├── vegetative:  25 a 60 dias  → V4 a Vn (perfilhamento + expansão)
    ├── flowering:   60 a 85 dias  → R0 a R4 (reprodutivo + antese)
    ├── grain_fill:  85 a 115 dias → R5 a R8 (enchimento de grãos)
    └── maturity:   115 a 145 dias → R9 (maturação)
```

**Nota:** O arroz irrigado no RS apresenta NDWI elevado na fase baseline por causa da lâmina d'água, o que é uma feição discriminativa importante para esta cultura.

### 5.3 Referências

- **Embrapa Clima Temperado.** Arroz Irrigado: Recomendações Técnicas da Pesquisa para o Sul do Brasil. Sistemas de Produção. Pelotas, RS.
- **Counce, P.A.; Keisling, T.C.; Mitchell, A.J.** A uniform, objective, and adaptive system for expressing rice development. Crop Science, v. 40, p. 436–443, 2000.
- **SOSBAI.** Arroz Irrigado: Recomendações Técnicas da Pesquisa para o Sul do Brasil. Reunião Técnica da Cultura do Arroz Irrigado, Cachoeirinha, RS.

---

## 6. TRIGO (Triticum aestivum)

### 6.1 Escala Fenológica de Referência

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

### 6.2 Mapeamento no Pipeline

```
Plantio (dia 0)
    │
    ├── baseline:   -15 a 0 dias   → solo pós-colheita da safra verão
    ├── emergence:    0 a 20 dias  → Zadoks 00–13 (germinação + emergência)
    ├── vegetative:  20 a 50 dias  → Zadoks 20–39 (perfilhamento + alongamento)
    ├── flowering:   50 a 75 dias  → Zadoks 50–69 (espigamento + antese)
    ├── grain_fill:  75 a 105 dias → Zadoks 70–79 (enchimento de grãos)
    └── maturity:   105 a 135 dias → Zadoks 80–92 (maturação + colheita)
```

### 6.3 Referências

- **Embrapa Trigo.** Informações Técnicas para Trigo e Triticale — Safra 2020. Passo Fundo, RS.
- **Zadoks, J.C.; Chang, T.T.; Konzak, C.F.** A decimal code for the growth stages of cereals. Weed Research, v. 14, p. 415–421, 1974.
- **Large, E.C.** Growth stages in cereals — illustration of the Feekes scale. Plant Pathology, v. 3, p. 128–129, 1954.
- **Embrapa Trigo.** Fenologia do Trigo: Caracterização dos Estádios de Desenvolvimento. Documentos Online 149, 2014.

---

## 7. AVEIA (Avena sativa / Avena strigosa)

### 7.1 Escala Fenológica de Referência

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

### 7.2 Mapeamento no Pipeline

```
Plantio (dia 0)
    │
    ├── baseline:   -15 a 0 dias   → solo pós-colheita safra verão
    ├── emergence:    0 a 20 dias  → Zadoks 00–13 (germinação + emergência)
    ├── vegetative:  20 a 45 dias  → Zadoks 20–39 (perfilhamento + alongamento)
    ├── flowering:   45 a 70 dias  → Zadoks 50–69 (emissão da panícula + antese)
    ├── grain_fill:  70 a 100 dias → Zadoks 70–79 (enchimento)
    └── maturity:   100 a 130 dias → Zadoks 80–92 (maturação)
```

### 7.3 Diferenciação TRIGO vs AVEIA

Embora as janelas sejam similares, existem diferenças fenológicas relevantes:

| Característica | Trigo | Aveia |
|---|---|---|
| Inflorescência | Espiga compacta | Panícula aberta |
| Altura | 70–100 cm | 60–120 cm (mais variável) |
| Perfilhamento | Moderado (3–5 perfilhos) | Intenso (5–8 perfilhos) |
| Ciclo vegetativo | Ligeiramente mais longo | Ligeiramente mais curto |
| Arquitetura do dossel | Compacto, ereto | Aberto, folhas mais laxas |

Essas diferenças estruturais são o que o SAR (Sentinel-1) deve capturar: dosséis mais abertos geram retroespalhamento VH diferente de dosséis compactos.

### 7.4 Referências

- **Embrapa Trigo.** Indicações Técnicas para a Cultura da Aveia. Comissão Brasileira de Pesquisa de Aveia. Passo Fundo, RS.
- **Floss, E.L.** Fisiologia e manejo da aveia. In: Reunião da Comissão Brasileira de Pesquisa de Aveia. UPF, Passo Fundo, 2011.
- **Zadoks, J.C.; Chang, T.T.; Konzak, C.F.** A decimal code for the growth stages of cereals. Weed Research, v. 14, p. 415–421, 1974.

---

## 8. CAFÉ (Coffea arabica / Coffea canephora)

### 8.1 Escala Fenológica de Referência

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

### 8.2 Mapeamento no Pipeline

Para o café, a "data de plantio" no pipeline corresponde ao **início da florada** (o marco que inicia o ciclo reprodutivo anual):

```
Florada (dia 0)
    │
    ├── baseline:   -15 a 0 dias   → dossel vegetativo pré-florada
    ├── emergence:    0 a 30 dias  → florada + queda de pétalas
    ├── vegetative:  30 a 90 dias  → fase chumbinho (expansão celular)
    ├── flowering:   90 a 150 dias → expansão rápida do fruto
    ├── grain_fill: 150 a 210 dias → granação (enchimento com matéria seca)
    └── maturity:   210 a 270 dias → maturação (cereja) e colheita
```

**Nota:** As janelas do café são ~2× mais longas que as das culturas anuais. O café é perene, então o "baseline" captura o dossel arbóreo já estabelecido (NDVI alto mesmo antes da florada), diferente das culturas anuais onde o baseline é solo exposto.

### 8.3 Particularidades do Café no Sensoriamento Remoto

| Aspecto | Impacto nos índices |
|---|---|
| Dossel perene | NDVI alto o ano todo (0.6–0.8), menor amplitude que culturas anuais |
| Espaçamento de plantio | Linhas de 3–4m criam padrão misto solo+copa no pixel |
| Sombreamento | Cafezais sombreados têm NDVI mais alto e estável |
| Latitude | Concentrado em MG/SP/ES (lat −15 a −23), separado geograficamente |

### 8.4 Referências

- **Embrapa Café.** Sistemas de Produção — Café Arábica e Conilon. Brasília, DF.
- **Camargo, A.P.; Camargo, M.B.P.** Definição e esquematização das fases fenológicas do cafeeiro arábica nas condições tropicais do Brasil. Bragantia, v. 60, n. 1, p. 65–68, 2001.
- **Pezzopane, J.R.M. et al.** Escala para avaliação de estádios fenológicos do cafeeiro arábica. Bragantia, v. 62, n. 3, p. 499–505, 2003.
- **Conab.** Acompanhamento da Safra Brasileira — Café (boletins mensais com estimativa de estágio fenológico por região).

---

## 9. Resumo das Janelas no Pipeline

### 9.1 Tabela Comparativa (dias após plantio)

| Estágio | SOJA | MILHO | FEIJÃO | ARROZ | TRIGO | AVEIA | CAFÉ |
|---|---|---|---|---|---|---|---|
| baseline | −15, 0 | −15, 0 | −15, 0 | −15, 0 | −15, 0 | −15, 0 | −15, 0 |
| emergence | 0, 20 | 0, 25 | 0, 10 | 0, 25 | 0, 20 | 0, 20 | 0, 30 |
| vegetative | 20, 55 | 25, 55 | 10, 25 | 25, 60 | 20, 50 | 20, 45 | 30, 90 |
| flowering | 55, 75 | 55, 85 | 25, 45 | 60, 85 | 50, 75 | 45, 70 | 90, 150 |
| grain_fill | 75, 100 | 85, 115 | 45, 65 | 85, 115 | 75, 105 | 70, 100 | 150, 210 |
| maturity | 100, 130 | 115, 140 | 65, 90 | 115, 145 | 105, 135 | 100, 130 | 210, 270 |

### 9.2 Ciclo Total Capturado

| Cultura | Dias totais | Classificação |
|---|---|---|
| FEIJÃO | 90 dias | Ciclo ultra-curto |
| SOJA | 130 dias | Ciclo curto |
| AVEIA | 130 dias | Ciclo médio |
| TRIGO | 135 dias | Ciclo médio |
| MILHO | 140 dias | Ciclo médio |
| ARROZ | 145 dias | Ciclo médio-longo |
| CAFÉ | 270 dias | Ciclo longo (perene) |

---

## 10. Justificativas das Decisões de Mapeamento

### 10.1 Por que 6 estágios genéricos?

A escolha de 6 estágios é um compromisso entre:
- **Granularidade suficiente** para capturar a dinâmica fenológica (baseline → pico → senescência)
- **Generalização entre culturas** — um modelo único precisa de colunas consistentes
- **Disponibilidade de dados** — Sentinel-2 revisita a cada ~5 dias, mas com nuvens a frequência efetiva pode ser ~15–20 dias; janelas menores que 15 dias teriam alta taxa de nulos

### 10.2 Por que usar janelas por cultura?

Duas culturas na mesma data absoluta podem estar em estágios completamente diferentes. Exemplo: em Janeiro, a SOJA plantada em Outubro está no estágio `flowering` (NDVI máximo), enquanto um TRIGO colhido em Setembro está sem dados. Ao ancorar as janelas na data de plantio individual, garantimos que estamos comparando "vegetative de SOJA" com "vegetative de MILHO" — mesma fase biológica, não mesma data do calendário.

### 10.3 Baseline: por que capturar pré-plantio?

O estágio baseline (−15 a 0 dias) captura o estado do solo/cobertura antes do plantio. Isso é discriminativo porque:
- **ARROZ irrigado:** baseline mostra lâmina d'água (NDWI alto, NDVI baixo)
- **CAFÉ perene:** baseline mostra dossel já verde (NDVI ~0.7)
- **Culturas anuais:** baseline mostra solo exposto ou restos culturais (NDVI ~0.2)

### 10.4 Limitações conhecidas

| Limitação | Impacto | Mitigação |
|---|---|---|
| Variabilidade de ciclo por cultivar | Janelas podem não alinhar perfeitamente com cultivares muito precoces ou tardias | Janelas largas o suficiente para cobrir a maioria dos genótipos comerciais |
| Safrinha (2ª safra) | Milho safrinha tem ciclo diferente do milho safra | Pipeline usa data de plantio individual, não assume safra |
| Sobreposição de estágios | Em condições adversas, estágios podem se sobrepor | Estatísticas de janela temporal (mean, p10, p90) capturam heterogeneidade |
| Café: ciclo bienal | Produção de café alterna anos de alta/baixa (bienalidade) | Não endereçado — classificação, não estimativa de produtividade |
