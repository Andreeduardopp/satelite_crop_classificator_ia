# Classificação de Culturas por Satélite — Arquitetura & Resultados

## 1. Visão Geral do Projeto

Este projeto classifica **7 tipos de culturas** cultivadas no Brasil usando dados de sensoriamento remoto por satélite:

| Cultura | Tipo | Safra | Principais Regiões |
|---|---|---|---|
| **SOJA** | Leguminosa | Verão (Out–Mar) | PR, RS, MT, GO |
| **MILHO** | Cereal | Verão (Out–Mar) | GO, MG, PR, MT |
| **ARROZ** | Cereal (irrigado) | Verão (Out–Mar) | RS |
| **FEIJÃO** | Leguminosa | Verão/Inverno | PR, MG, BA |
| **TRIGO** | Cereal de inverno | Inverno (Mai–Set) | PR, RS |
| **AVEIA** | Cereal de inverno | Inverno (Mai–Set) | PR, RS |
| **CAFÉ** | Perene | Ano todo | MG, SP, ES |

Cada amostra é um **talhão agrícola** definido por um arquivo KML poligonal, com datas de plantio e colheita conhecidas, obtidas de registros agrícolas brasileiros. O pipeline extrai feições espectrais e de radar das séries temporais dos satélites Sentinel, e então um classificador ensemble baseado em árvores prediz o tipo de cultura.

---

## 2. Fontes de Dados

### 2.1 Sentinel-2 (Óptico)

- **Satélite:** Sentinel-2 L2A (corrigido atmosfericamente)
- **Resolução:** 10m/pixel
- **Revisita:** ~5 dias
- **Bandas utilizadas:** B02 (Azul), B03 (Verde), B04 (Vermelho), B08 (NIR), B11 (SWIR), SCL (Classificação de Cena)
- **Máscara de nuvens:** Classes SCL 2, 4, 5, 6, 7 são consideradas céu limpo. Pixels nublados/sombreados são excluídos antes do cálculo das estatísticas.

Três índices de vegetação/água são derivados por pixel:

| Índice | Fórmula | O que captura |
|---|---|---|
| **NDVI** | (B08 − B04) / (B08 + B04) | Verdor do dossel, atividade clorofiliana |
| **EVI** | 2.5 × (B08 − B04) / (B08 + 6×B04 − 7.5×B02 + 1) | Vegetação aprimorada, menor saturação em dossel denso |
| **NDWI** | (B03 − B08) / (B03 + B08) | Conteúdo de água foliar, estado de irrigação |

### 2.2 Sentinel-1 (SAR — Radar de Abertura Sintética)

- **Satélite:** Sentinel-1 GRD (Ground Range Detected)
- **Modo:** IW (Interferometric Wide Swath), polarização dupla VV+VH
- **Resolução:** ~10m/pixel
- **Revisita:** ~6 dias
- **Correção de retroespalhamento:** GAMMA0_TERRAIN (corrigido pelo terreno)
- **Vantagem principal:** Penetra nuvens — zero perda de dados por clima

Quatro índices de radar são derivados:

| Índice | Fórmula | O que captura |
|---|---|---|
| **VV** (dB) | 10 × log10(VV) | Rugosidade da superfície, umidade do solo |
| **VH** (dB) | 10 × log10(VH) | Espalhamento volumétrico — biomassa, densidade do dossel |
| **CR** (Razão de polarização cruzada) | VH / VV (linear) | Feição SAR mais discriminativa para culturas |
| **RVI** (Índice de Vegetação por Radar) | 4 × VH / (VV + VH) | Proxy de biomassa, razão de despolarização |

### 2.3 Por que óptico + SAR juntos?

Índices ópticos (NDVI, EVI) capturam **qual é a cor da cultura** — clorofila, verdor, senescência. SAR captura **qual é a estrutura da cultura** — altura do dossel, geometria das folhas, densidade dos caules. Duas culturas podem parecer idênticas no NDVI (mesmo verdor) mas ter assinaturas de radar diferentes porque sua arquitetura física difere.

Isso é crítico para **TRIGO vs AVEIA**: ambos são cereais de inverno plantados na mesma região e época, com curvas espectrais quase idênticas. Mas o trigo cresce mais alto com espigas mais densas, produzindo um perfil de retroespalhamento VH diferente.

---

## 3. Arquitetura do Pipeline

### 3.1 Fluxo de Dados

```
Arquivos KML poligonais (limites dos talhões + datas de plantio/colheita)
    │
    ├── Parsing do nome do arquivo → crop_label, field_id, planting_date
    ├── Parsing do XML KML → coordenadas do polígono → centroide (lat, lon) + área (ha)
    │
    ▼
Janelas Fenológicas por Estágio (específicas por cultura)
    │
    ├── baseline    (−15 a 0 dias do plantio)
    ├── emergence   (0 a 20 dias)
    ├── vegetative  (20 a 50 dias)
    ├── flowering   (50 a 80 dias)
    ├── grain_fill  (80 a 110 dias)
    └── maturity    (110 a 140 dias)
    │
    ▼
Para cada estágio: chamadas à API do Sentinel Hub
    │
    ├── Statistical API (Sentinel-2) → NDVI/EVI/NDWI mean, median, std, p10, p90
    ├── Statistical API (Sentinel-1) → VV/VH/CR/RVI mean, median, std, p10, p90
    │   (SAR roda independentemente — sem dependência de nuvens)
    │
    ▼
Banco de Dados SQLite (uma linha por talhão)
    │
    ├── Metadados: field_id, crop_label, planting_date, area_hectares, lat, lon
    ├── Óptico:  90 colunas (3 índices × 5 estatísticas × 6 estágios)
    ├── SAR:    120 colunas (4 índices × 5 estatísticas × 6 estágios)
    └── Total:  ~215 colunas por talhão
```

### 3.2 Janelas Fenológicas Específicas por Cultura

Cada cultura tem um timing de crescimento diferente. Usar janelas de estágio específicas por cultura garante que capturamos o momento fenológico correto:

| Estágio | SOJA | CAFÉ | TRIGO |
|---|---|---|---|
| baseline | −15 a 0 d | −15 a 0 d | −15 a 0 d |
| emergence | 0–20 d | 0–30 d | 0–20 d |
| vegetative | 20–55 d | 30–90 d | 20–50 d |
| flowering | 55–75 d | 90–150 d | 50–75 d |
| grain_fill | 75–100 d | 150–210 d | 75–105 d |
| maturity | 100–130 d | 210–270 d | 105–135 d |

CAFÉ tem janelas muito mais longas porque é uma cultura perene com um ciclo de frutos de 9 meses.

### 3.3 Estratégia de API

**Primária: Statistical API** — O Sentinel Hub calcula estatísticas zonais no lado do servidor. A resposta é ~1 KB JSON vs ~400 KB de download raster. A máscara de nuvens é integrada ao evalscript via dataMask.

**Fallback: Process API** — Baixa rasters completos como TIFF quando a Statistical API está indisponível ou quando imagens brutas são necessárias para visualização.

### 3.4 Segurança contra Falhas

Cada talhão é commitado no SQLite imediatamente após seus 6 estágios serem completados. Na reinicialização, field_ids existentes são carregados e pulados. Um crash perde no máximo o trabalho de um talhão.

---

## 4. Engenharia de Feições

A partir das 90 colunas ópticas base (+ 120 colunas SAR), o script de treinamento deriva feições adicionais que capturam **dinâmica temporal** e **relações entre índices**:

### 4.1 Feições Temporais

| Feição | Fórmula | Propósito |
|---|---|---|
| **Deltas entre estágios** | valor(estágio_n+1) − valor(estágio_n) | Taxa de crescimento/senescência entre estágios consecutivos |
| **Estágio de pico** | argmax dos valores médios entre estágios | Qual estágio de crescimento tem o maior índice |
| **Valor de pico / Valor mínimo** | max / min entre estágios | Amplitude total da curva de crescimento |
| **Amplitude** | pico − mínimo | Faixa dinâmica total |
| **Taxa de esverdeamento** | (pico − baseline) / índice_estágio_pico | Quão rápido a cultura verdeja |
| **Taxa de senescência** | (pico − maturidade) / estágios_após_pico | Quão rápido a cultura senesce |
| **CV temporal** | std / média entre estágios | Variabilidade temporal geral |
| **Acumulado** | soma das médias entre estágios | Verdor integrado total (proxy de biomassa) |

### 4.2 Feições Entre Índices

| Feição | Fórmula | Propósito |
|---|---|---|
| **Razão NDVI/EVI** | NDVI_mean / EVI_mean por estágio | Saturação do dossel — denso vs esparso |
| **Razão NDVI/NDWI** | NDVI_mean / NDWI_mean por estágio | Balanço entre verdor e conteúdo de água |
| **Diferença NDVI−EVI** | NDVI − EVI nos estágios tardios | Divergência tardia para separação de culturas |
| **Diferença NDVI−NDWI** | NDVI − NDWI em grain_fill/maturity | Separa culturas com padrões de secagem diferentes |
| **Razão de std** | NDVI_std / EVI_std nos estágios tardios | Diferenças na estrutura de variabilidade |

### 4.3 Feições Geográficas e de Calendário

| Feição | Fonte | Propósito |
|---|---|---|
| **latitude, longitude** | Centroide do polígono KML | Distribuição regional de culturas (CAFÉ em MG vs TRIGO no RS) |
| **planting_doy** | Dia do ano da data de plantio | Inverno (TRIGO/AVEIA: ~Mai–Jun) vs verão (SOJA/MILHO: ~Out–Dez) |
| **planting_doy_sin/cos** | Codificação cíclica do dia do ano | Garante que 1º de Jan ≈ 31 de Dez no espaço de feições |
| **area_hectares** | Área geodésica do polígono | Tamanho do talhão se correlaciona com tipo de cultura no Brasil |

### 4.4 Indicadores de Nulo

Com ~9–15% de taxa de nulos por cobertura de nuvens, colunas binárias `_is_null` são adicionadas para feições com >10% de valores ausentes. Isso permite ao modelo distinguir "NDVI baixo" de "dado não disponível" — significados diferentes que o tratamento bruto de NaN obscurece.

---

## 5. Evolução do Modelo

### 5.1 Progressão da Arquitetura

O modelo evoluiu através de quatro iterações, cada uma endereçando gargalos identificados na versão anterior:

**v1 — XGBoost Baseline** (`train_xgboost.py`)
- Classificador XGBoost único com hiperparâmetros manuais
- 91 feições brutas (sem engenharia)
- Validação cruzada estratificada de 5 folds

**v2 — Engenharia de Feições + Optuna** (`train_xgboost_v2.py`)
- 63 feições engenheiradas adicionadas (deltas, picos, razões, etc.)
- Busca de hiperparâmetros com Optuna (80 trials)
- Mesma arquitetura XGBoost única
- Filtro `min_stages >= 3` para remover amostras de baixa qualidade

**v3 — Ensemble + Seleção de Feições + Geográfico** (`train_xgboost_v3.py`)
- **Ensemble:** XGBoost + ExtraTreesClassifier com votação suave
  - XGBoost: gradient boosting — forte em dados tabulares/estruturados
  - ExtraTrees: divisões aleatórias — descorrelacionado do XGBoost, captura padrões diferentes
  - Votação suave: média das probabilidades preditas, reduz variância
- **Seleção de feições:** treina um XGBoost rápido, descarta os 40% piores feições por ganho, retreina no conjunto limpo (270 → 162 feições)
- **Feições geográficas:** latitude, longitude, planting_doy (codificação sin/cos)
- **Indicadores de nulo:** flags binários para colunas com muitos nulos
- Ambos os modelos sintonizados independentemente pelo Optuna (80 XGB + 40 ET trials)
- Melhor modelo auto-selecionado por F1 macro do CV

### 5.2 Por Que Não Random Forest?

A exploração inicial usou Random Forest, mas o XGBoost consistentemente o superou porque:

1. **Boosting vs bagging:** XGBoost constrói árvores sequencialmente, cada uma corrigindo os erros da anterior. Random Forest constrói árvores independentes e faz a média. Para dados tabulares com interações complexas (estágios fenológicos × índices × geografia), boosting captura mais sinal.
2. **Tratamento de valores ausentes:** XGBoost aprende nativamente direções ótimas de divisão para valores NaN (nossa taxa de nulos de 9–15%). Random Forest requer imputação, que introduz ruído.
3. **Regularização:** A regularização L1/L2 do XGBoost (reg_alpha, reg_lambda) previne overfitting nas 160+ feições. Random Forest depende apenas de max_depth e min_samples.

ExtraTrees foi adicionado de volta ao ensemble (não Random Forest) porque usa **limiares de divisão aleatórios** em vez de divisões ótimas, tornando-o maximamente descorrelacionado do XGBoost — melhor para diversidade do ensemble.

### 5.3 Hiperparâmetros Finais (v3)

**XGBoost** (sintonizado com Optuna, 80 trials):
| Parâmetro | Valor |
|---|---|
| n_estimators | 823 |
| max_depth | 9 |
| learning_rate | 0.014 |
| subsample | 0.83 |
| colsample_bytree | 0.88 |
| min_child_weight | 9 |
| gamma | 0.18 |
| reg_alpha | 0.20 |
| reg_lambda | 0.84 |

**ExtraTrees** (sintonizado com Optuna, 40 trials):
| Parâmetro | Valor |
|---|---|
| n_estimators | 807 |
| max_depth | 19 |
| min_samples_split | 7 |
| min_samples_leaf | 1 |
| max_features | 0.90 |

---

## 6. Progressão dos Resultados

### 6.1 Métricas Gerais

| Versão | Amostras | Feições | Acurácia | F1 macro | Mudança Principal |
|---|---|---|---|---|---|
| v1 baseline (50 KML/cultura) | ~350 | 91 | 45.4% | 0.451 | Feições brutas, params manuais |
| v2 (50 KML + eng. feições + Optuna) | ~350 | 154 | 54.1% | 0.519 | +63 feições engenheiradas |
| v1 baseline (500 KML/cultura) | 3500 | 91 | 56.6% | 0.565 | 10x mais dados |
| v2 (500 KML + eng. feições + Optuna) | 3407 | 154 | 59.9% | 0.598 | Engenharia + tuning em escala |
| **v3 (ensemble + seleção + planting_doy)** | 3407 | 162 | **80.0%** | **0.800** | Ensemble, data de plantio, indicadores de nulo |
| **v3 + lat/lon** | 3407 | 162 | **89.8%** | **0.898** | Coordenadas geográficas |

### 6.2 Progressão do F1 Score por Classe

| Cultura | v1 (500 KML) | v2 (500 KML) | v3 | v3 + lat/lon |
|---|---|---|---|---|
| ARROZ | 0.672 | 0.685 | 0.807 | **0.957** |
| AVEIA | 0.496 | 0.537 | 0.733 | 0.780 |
| CAFÉ | 0.721 | 0.748 | 0.873 | **0.979** |
| FEIJÃO | 0.465 | 0.517 | 0.876 | **0.946** |
| MILHO | 0.643 | 0.635 | 0.808 | **0.917** |
| SOJA | 0.505 | 0.564 | 0.796 | **0.942** |
| TRIGO | 0.452 | 0.497 | 0.707 | 0.767 |

### 6.3 O Que Impulsionou Cada Salto

**v1 → v2 (+8.7pp):** Engenharia de feições — deltas entre estágios e detecção de pico deram ao modelo informação sobre a forma temporal em vez de snapshots brutos.

**v2 → v3 (+20.1pp):** Três mudanças empilhadas:
- `planting_doy` (feição #1 por ganho) — separou culturas de inverno vs verão imediatamente
- Ensemble (XGB + ET) — reduziu variância nos pares difíceis
- Seleção de feições — remover 40% de feições ruidosas melhorou a generalização

**v3 → v3+latlon (+9.8pp):** Coordenadas geográficas. CAFÉ é cultivado em Minas Gerais (lat ~−20), longe do cinturão de grãos do sul (lat ~−28). Latitude sozinha quase resolveu a classificação de CAFÉ (97.9% F1). Também melhorou ARROZ (concentrado no RS) e SOJA (espalhada pelo centro-sul).

### 6.4 Desafio Remanescente: TRIGO vs AVEIA

A matriz de confusão mostra o erro dominante remanescente:
- 24% do TRIGO verdadeiro predito como AVEIA
- 20% da AVEIA verdadeira predita como TRIGO

Esses cereais de inverno compartilham:
- Mesma janela de plantio (Maio–Junho)
- Mesma região geográfica (Paraná/Rio Grande do Sul)
- Curvas NDVI/EVI quase idênticas
- Tamanhos de talhão similares

É por isso que SAR está sendo adicionado — o retroespalhamento radar pode detectar as diferenças estruturais (trigo: mais alto, espigas mais densas; aveia: mais baixa, dossel mais aberto) que os sensores ópticos não conseguem ver.

---

## 7. Análise de Importância de Feições

### 7.1 Top 10 Feições por Ganho (v3 + lat/lon)

| Ranking | Feição | Ganho | Categoria |
|---|---|---|---|
| 1 | planting_doy | ~40 | Calendário |
| 2 | planting_doy_cos | ~35 | Calendário |
| 3 | latitude | ~22 | Geográfico |
| 4 | longitude | ~18 | Geográfico |
| 5 | planting_doy_sin | ~14 | Calendário |
| 6 | NDVI_peak_stage | ~8 | Pico/amplitude |
| 7 | EVI_peak_stage | ~7 | Pico/amplitude |
| 8 | NDWI_std_grain_fill | ~6 | Feição base |
| 9 | EVI_p90_grain_fill | ~6 | Feição base |
| 10 | area_hectares | ~5 | Geográfico |

### 7.2 Importância por Categoria (ganho total)

| Categoria | Ganho Total | Papel |
|---|---|---|
| Feições base (estatísticas brutas) | ~163 | Fundação — medições espectrais/água diretas |
| Data de plantio | ~68 | Separador de safra — culturas de inverno vs verão |
| Deltas entre estágios | ~54 | Dinâmica temporal — taxas de crescimento/declínio |
| Outros (lat/lon, área) | ~34 | Contexto geográfico |
| Pico/amplitude | ~32 | Forma da curva de crescimento |
| Razões entre índices | ~11 | Relações multi-índice |
| Esverdeamento/senescência | ~8 | Caracterização da velocidade de crescimento |
| CV temporal | ~6 | Assinatura de variabilidade geral |
| Divergência tardia | ~5 | Diferenciação no final da safra |
| Razão início/fim | ~4 | Comparação de forma do ciclo completo |
| Acumulado | ~2 | Índice integrado total |

### 7.3 Insights Principais

1. **Feições de calendário + geográficas contribuem ~40% da importância total**, apesar de serem apenas 5 de 162 feições. Saber *quando* e *onde* uma cultura é plantada é tão informativo quanto *como ela se parece* espectralmente.

2. **Feições base ainda dominam** com ~163 de ganho total. As medições espectrais brutas permanecem como fundação — feições derivadas refinam o sinal mas não o substituem.

3. **Deltas entre estágios são a categoria de engenharia mais valiosa** (~54 de ganho). A *taxa de mudança* entre estágios fenológicos contém mais poder discriminativo do que valores absolutos em qualquer estágio isolado.

4. **Feições dos estágios grain_fill e maturity aparecem desproporcionalmente** no top 30. A divergência no final da safra é onde as culturas mais se diferenciam — até o florescimento, a maioria das culturas parece "verde"; é como elas amadurecem e secam que as separa.

---

## 8. Próximos Passos: Integração SAR

O pipeline v2 (`phenology_feature_pipeline_v2.py`) adiciona dados SAR do Sentinel-1 junto com os ópticos. Impacto esperado:

| Benefício Esperado | Mecanismo |
|---|---|
| Separação TRIGO vs AVEIA | Estrutura de dossel diferente → retroespalhamento VH diferente |
| Redução da taxa de nulos | SAR penetra nuvens — preenche lacunas na cobertura óptica |
| Sinal complementar | Estrutura (SAR) + cor (óptico) = classificação mais robusta |

**Backfill SAR para linhas existentes:** 3500 talhões × 6 estágios = 21.000 chamadas API. Apenas SAR é requisitado — dados ópticos existentes permanecem intocados. A migração é segura contra falhas e resumível.

**Meta:** Com feições SAR adicionadas, o modelo deve empurrar o F1 de TRIGO/AVEIA de ~0.77 para 0.85+, levando a acurácia geral acima de 92%.
