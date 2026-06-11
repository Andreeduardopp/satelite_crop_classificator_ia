# Análise Técnica — Classificação de Culturas por Satélite - terceira semana

## Fase 1: Definição do Problema
Para essa semana nosso principal objetivo é treinar o primeiro modelo utilizando os dados obtidos pela pipeline.
- Iterar por 1000 kml por cultura e recuperar os dados de acordo com a teoria que montamos na segunda semana
- Montar um banco de dados com os dados provenientes da pipeline
- Treinar o primeiro modelo com esses dados e verificar as metricas obtidas
- Estudar a criação de um servidor mlflow para gestão dos modelos criados

## Desenvolvimento da Pipeline
Primeiro passo dessa semana foi o desenvolvimento da pipeline junto com o modelo. 

### O Que Você Envia

Um arquivo KML contendo um polígono (o limite do seu talhão). O nome do arquivo codifica os metadados:
`CULTURA_idTalhao_plantio_DD-MM-AA_colheita_DD-MM-AA.kml`

O pipeline extrai o tipo de cultura, ID do talhão, data de plantio e data de colheita a partir dessa convenção de nomenclatura.


### O Que É Calculado

Para cada estágio fenológico (baseline, emergence, vegetative, flowering, grain_fill, maturity), o pipeline requisita **todos os pixels dentro do seu polígono** ao Sentinel Hub para a janela de tempo correspondente.

#### Índices Ópticos (Sentinel-2) — calculados por pixel

| Índice | Fórmula | O que mede |
|--------|---------|------------|
| **NDVI** | (B08 − B04) / (B08 + B04) | Vigor vegetativo — razão entre reflectância no infravermelho próximo (NIR) e no vermelho. Vegetação saudável absorve vermelho e reflete NIR fortemente. Faixa: −1 a +1, culturas tipicamente 0.3–0.9. |
| **EVI** | 2.5 × (B08 − B04) / (B08 + 6×B04 − 7.5×B02 + 1) | Índice de vegetação aprimorado — corrige efeitos atmosféricos e de fundo do solo. Menos saturado que o NDVI em dossel denso. |
| **NDWI** | (B03 − B08) / (B03 + B08) | Índice de diferença normalizada de água — mede o conteúdo de água foliar usando reflectância no verde vs NIR. |

#### Índices SAR (Sentinel-1 GRD) — calculados por pixel

| Índice | Fórmula | O que mede |
|--------|---------|------------|
| **VV** | 10 × log₁₀(VV) em dB | Retroespalhamento na polarização vertical-vertical. Sensível à umidade do solo e altura da cultura. |
| **VH** | 10 × log₁₀(VH) em dB | Retroespalhamento na polarização vertical-horizontal. Sensível ao espalhamento volumétrico (biomassa da cultura). |
| **CR** | VH / VV (linear) | Razão de polarização cruzada — indica estrutura da cultura e complexidade do dossel. |
| **RVI** | 4 × VH / (VV + VH) | Índice de vegetação por radar — proxy de biomassa que funciona através de nuvens. |

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

#### Janelas SAR Expandidas (v3)

O Sentinel-1 revisita a mesma localização a cada ~12 dias no Brasil, enquanto o Sentinel-2 (óptico) tem revisita de ~5 dias. Isso significa que as janelas curtas de estágio fenológico — especialmente para culturas de ciclo rápido como FEIJAO (10–20 dias por estágio) — frequentemente não contêm nenhuma aquisição SAR.

**O problema do sar:** Janelas ópticas e SAR eram idênticas. Resultado:

| Cultura | Janela típica por estágio | Cobertura SAR |
|---------|---------------------------|---------------|
| CAFE | 30–60 dias | ~97% |
| MILHO | 25–30 dias | ~41% |
| FEIJAO | 10–20 dias | ~6% |

Janelas SAR são expandidas simetricamente para garantir no mínimo **24 dias** de cobertura (≥ 1 passagem do Sentinel-1). As janelas ópticas permanecem inalteradas.

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

Além disso, o pipeline marca cada linha processada com uma flag `sar_backfill_done`, evitando re-requisições infinitas para campos que genuinamente não possuem cobertura SAR.

---
### O Que Você Recebe de Volta
O pipeline **não** retorna valores individuais de pixel. Em vez disso, ele agrega todos os pixels válidos em todo o polígono em **5 estatísticas resumo**:

| Estatística | Significado |
|-------------|-------------|
| **mean** | Valor médio de todos os pixels no seu talhão |
| **median** | Percentil 50 — robusto a outliers |
| **std** | Desvio padrão — variabilidade espacial dentro do talhão |
| **p10** | Percentil 10 — representa a área com pior desempenho |
| **p90** | Percentil 90 — representa a área com melhor desempenho |

**Exemplo:** `NDVI_mean_vegetative = 0.72` significa que o NDVI médio de todos os pixels no seu polígono durante o estágio vegetativo foi 0.72.

#### Saída Total por Talhão

| Categoria | Quantidade |
|-----------|------------|
| Feições ópticas | 3 índices × 5 estatísticas × 6 estágios = **90** |
| Feições SAR | 4 índices × 5 estatísticas × 6 estágios = **120** |
| Metadados | field_id, crop_label, planting_date, area_hectares, latitude, longitude, stages_covered = **7** |
| **Total de colunas** | **217** |

Cada talhão produz **uma linha** no banco de dados representando seu perfil fenológico completo.

---
#### Tempo previsto para treinamento
Para 1000 kmls teremos um tempo previsto de 2 dias para finalizar a pipeline.

### Desenvolvimento do Modelo

#### Feature Engineering

**Objetivo:** Transformar 217 features brutas (índices ópticos + SAR por estágio fenológico) em features derivadas que capturem os padrões fenológicos discriminativos entre culturas.

**Estratégia:** O XGBoost é utilizado como modelo base devido a suas vantagens:
- Velocidade de treinamento otimizada (tree_method="hist")
- Transparência na importância das features (gain, cover, frequency)
- Suporte nativo a valores ausentes (NaNs são modelados como divisões especiais)
- Controle de overfitting através de regularização (reg_alpha, reg_lambda) e poda (max_delta_step, gamma)
- Suporte para multi-classe sem one-hot-encoding

**Ensemble:** Combina XGBoost com ExtraTrees (voting='soft') para:
- Aumentar robustez através de diversidade de modelos
- Capturar padrões não-lineares (XGBoost via splitting ganancioso) e aleatórios (ExtraTrees via splitting aleatório)
- Reduzir variância e overfitting

---

#### Features Derivadas Implementadas

Todas as features derivadas são calculadas no módulo `engineer_features()` em **train_xgboost_v3.py**:

| Categoria | Features | Lógica |
|-----------|----------|--------|
| **Plantio** | `planting_doy`, `planting_doy_sin`, `planting_doy_cos` | Codificação cíclica da data de plantio para capturar culturas de inverno vs verão |
| **Dinâmica Temporal** | `{IDX}_mean_delta_{S1}_to_{S2}` para cada índice e pares de estágios | Mudanças entre estágios sucessivos revelam velocidade de desenvolvimento |
| **Pico Fenológico** | `{IDX}_peak_stage`, `{IDX}_peak_value`, `{IDX}_min_value`, `{IDX}_amplitude` | Identifica em qual estágio o índice atinge máximo; amplitude é discriminativa (ex: FEIJAO tem picos baixos) |
| **Taxa de Crescimento** | `{IDX}_greenup_rate` = (peak_value - baseline) / peak_stage_index | Velocidade de crescimento da cultura — FEIJAO cresce muito rápido em dias, CAFE é lento |
| **Taxa de Senescência** | `{IDX}_senescence_rate` = (peak_value - maturity) / stages_after_peak | Velocidade de degradação — TRIGO/AVEIA têm senescência pronunciada, SOJA é gradual |
| **Razões Ópticas** | `NDVI_EVI_ratio_{stage}`, `NDVI_NDWI_ratio_{stage}` | Razões entre índices revelam composição do dossel e teor de água foliar |
| **Variabilidade Temporal** | `{IDX}_temporal_cv` = std(means_all_stages) / mean(means_all_stages) | Culturas com fenologia bem definida (ex: CAFE) têm CV baixa; FEIJAO tem CV alta |
| **Heterogeneidade Espacial** | `{IDX}_mean_spread` = mean(p90 - p10 por estágio) | Culturas bem manejadas têm spread baixo; parcelas com problemas têm spread alto |
| **Cumulative Index** | `{IDX}_cumulative` = sum(means_all_stages) | Proxy de biomassa acumulada — culturas de ciclo longo (CAFE) têm valores altos |
| **Late-Stage Divergence** | `NDVI_minus_EVI_{stage}`, `NDVI_minus_NDWI_{stage}`, `std_ratio_NDVI_EVI_{stage}` | Diferenças em grain_fill/maturity são discriminativas entre grupos (ex: TRIGO vs SOJA) |
| **Razão Ciclo Completo** | `{IDX}_early_late_ratio` = mean(baseline, emergence) / mean(grain_fill, maturity) | Culturas com crescimento robusto têm razão alta; culturas precoces com senescência rápida têm razão baixa |

**Total de features derivadas:** ~80 novas features + 12 null-indicators (se NaNs > 10%) = ~100+ features para seleção

---

#### Seleção de Features

**Método:** Feature Importance via XGBoost rápido (n_estimators=200, max_depth=5)

1. Treina modelo rápido em todas as features brutas + derivadas
2. Extrai importância via `gain` (redução de loss por split)
3. Seleciona top 60% das features (mantém mínimo de 10 features)

**Resultado esperado:** ~50-60 features selecionadas para treinamento final

**Vantagem:** Reduz overfitting, acelera CV, torna o modelo interpretável

---

#### Hiperparâmetros Otimizados (Optuna)

**XGBoost (80 trials):**
- `n_estimators`: 200–1000
- `max_depth`: 3–10 (profundidade da árvore; valores altos → overfitting)
- `learning_rate`: 0.01–0.3 (taxa de aprendizado; valores baixos → lentidão)
- `subsample`, `colsample_bytree`: 0.3–1.0 (regularização via sub-amostragem)
- `reg_alpha`, `reg_lambda`: 1e-3–10.0 (regularização L1/L2)
- `min_child_weight`, `gamma`, `max_delta_step`: controle de overfitting

**ExtraTrees (40 trials):**
- `n_estimators`: 200–1000
- `max_depth`: 5–30
- `min_samples_split`, `min_samples_leaf`: controle de nó folha
- `max_features`: 0.3–1.0 (fração de features por split)

**Métrica:** F1 Macro (balanceado entre classes) via 5-fold cross-validation

---

#### Validação

**Cross-Validation:** 5-fold StratifiedKFold
- Estratificado: mantém proporção de classes em cada fold
- Random shuffle: evita viés de ordem

**Métricas:**
- Accuracy global
- F1 Macro (média de F1 por classe — penaliza desempenho ruins em classes pequenas)
- Confusion matrix por fold (diagnostica confusões específicas entre culturas)
- Classification report detalhado (precision, recall, F1 por classe)

---

## Resultados

### Treinamento (Cross-Validation 5-fold)

O modelo foi treinado com **6.872 amostras** (7.000 no banco, 128 filtradas por `stages_covered < 3`), 510 features originais reduzidas a 306 após seleção, e 7 classes de cultura.

| Modelo | Accuracy | F1 Macro | F1 Weighted |
|--------|----------|----------|-------------|
| **XGBoost** | **89.74%** | **0.8965** | **0.8967** |
| ExtraTrees | 88.85% | 0.8877 | 0.8878 |
| Ensemble | 89.39% | 0.8932 | 0.8933 |

O XGBoost individual superou o ensemble — a combinação com ExtraTrees via soft voting diluiu levemente as predições.

**Resultado por classe (treino CV):**

| Cultura | Precision | Recall | F1 | Suporte |
|---------|-----------|--------|----|---------|
| CAFE | 0.975 | 0.988 | 0.982 | 994 |
| FEIJAO | 0.945 | 0.983 | 0.963 | 991 |
| ARROZ | 0.952 | 0.962 | 0.957 | 976 |
| SOJA | 0.893 | 0.943 | 0.917 | 995 |
| MILHO | 0.930 | 0.812 | 0.867 | 937 |
| AVEIA | 0.785 | 0.811 | 0.798 | 988 |
| TRIGO | 0.805 | 0.779 | 0.792 | 991 |

---

### Avaliação em Dados de Teste (hold-out)

Avaliação com **347 amostras** de teste independentes (50 KMLs por cultura, nunca vistos durante o treinamento), processadas pela mesma pipeline de extração de features.

| Métrica | Treino (CV) | Teste | Gap |
|---------|-------------|-------|-----|
| Accuracy | 89.74% | 85.88% | -3.86pp |
| F1 Macro | 0.8965 | 0.8601 | -0.0364 |
| F1 Weighted | 0.8967 | 0.8601 | -0.0366 |

O gap treino→teste de ~3.6pp em F1 é moderado, indicando um nível aceitável de overfitting. A taxa de nulos no teste (39.2%) foi significativamente maior que no treino (23.1%), o que contribui para a queda.

**Resultado por classe (teste) e comparação com treino:**

| Cultura | F1 Treino | F1 Teste | Delta | Precision | Recall |
|---------|-----------|----------|-------|-----------|--------|
| CAFE | 0.982 | 0.980 | -0.002 | 1.000 | 0.960 |
| ARROZ | 0.957 | 0.957 | +0.000 | 1.000 | 0.918 |
| FEIJAO | 0.963 | 0.862 | **-0.101** | 0.783 | 0.959 |
| SOJA | 0.917 | 0.870 | -0.047 | 0.952 | 0.800 |
| TRIGO | 0.792 | 0.808 | +0.016 | 0.778 | 0.840 |
| AVEIA | 0.798 | 0.792 | -0.006 | 0.826 | 0.760 |
| MILHO | 0.867 | 0.753 | **-0.115** | 0.731 | 0.776 |

---

### Análise da Matriz de Confusão (Teste)

A matriz de confusão do teste revela os padrões de erro do modelo em dados não vistos:

**Classes estáveis (generalizam bem):**
- **CAFE** (96% recall, 100% precision): Fenologia de ciclo longo (~270 dias) produz um perfil espectral completamente distinto. Apenas 2 amostras confundidas com MILHO.
- **ARROZ** (91.8% recall, 100% precision): Perfil hídrico (NDWI) característico de cultivo irrigado/alagado. Erros menores para MILHO (6.1%) e SOJA (2%).
- **TRIGO** (84% recall): Mantém desempenho similar ao treino. Confusão exclusiva com AVEIA (16%) — ambas cereais de inverno, plantadas na mesma época.

**Classes com queda de desempenho:**
- **MILHO** (F1 caiu de 0.867 → 0.753): Principal confusão com FEIJAO (22.4% dos MILHO classificados como FEIJAO). Em dados novos, a sobreposição espectral no estágio vegetativo é mais problemática. Também absorve erros de SOJA (16% de SOJA → MILHO).
- **FEIJAO** (F1 caiu de 0.963 → 0.862): O recall permanece alto (95.9%), mas a precision caiu para 78.3%. O modelo classifica muitas amostras de MILHO como FEIJAO, porque ambas têm picos NDVI semelhantes no ciclo curto.
- **SOJA** (F1 caiu de 0.917 → 0.870): 16% de SOJA classificado como MILHO e 4% como FEIJAO. A alta precision (95.2%) indica que quando o modelo prediz SOJA, geralmente está correto.

**Padrões de confusão dominantes:**
1. **AVEIA ↔ TRIGO** (24% de AVEIA→TRIGO, 16% de TRIGO→AVEIA): Cereais de inverno com fenologia quase idêntica. Este é um limite estrutural — a separação depende fortemente de features SAR (VV_median_maturity) e localização geográfica.
2. **MILHO ↔ FEIJAO** (22.4% de MILHO→FEIJAO): Ambas culturas de verão com ciclo relativamente curto. No teste, este par de confusão se tornou mais severo que no treino, sugerindo que o modelo memorizou padrões geográficos do treino que não transferem bem.
3. **SOJA → MILHO** (16%): Sobreposição no estágio vegetativo entre as duas maiores culturas de verão do Brasil.

---

### Diagnóstico e Próximos Passos

**O que funciona:**
- CAFE, ARROZ e TRIGO generalizam bem — seus perfis fenológicos são suficientemente distintos
- O gap geral treino→teste de 3.6pp F1 é aceitável para um primeiro modelo
- SAR (VV, VH) é a categoria de features mais importante, validando a decisão de incluir Sentinel-1

**O que precisa melhorar:**
- **MILHO ↔ FEIJAO**: A confusão sugere que features de ciclo de crescimento (duração, velocidade) são mais discriminativas que os valores absolutos dos índices. Considerar adicionar features baseadas na diferença de duração do ciclo e na taxa de mudança do SAR
- **Dependência geográfica**: O modelo usa lat/lon como features top-5 em importância. Isso ajuda no treino mas indica risco de overfitting regional. Testar desempenho com exclusão de lat/lon
- **Taxa de nulos no teste (39.2% vs 23.1% no treino)**: A cobertura de nuvens nos períodos de teste impacta mais as fases iniciais (baseline, emergence). Considerar estratégias de imputação temporal
- **Amostra de teste pequena (50/classe)**: A avaliação com 347 amostras tem variância alta. Executar a avaliação completa com as 7.000 amostras de teste para métricas mais confiáveis

