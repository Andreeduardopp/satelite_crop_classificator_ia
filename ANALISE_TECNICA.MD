# Análise Técnica - Classificador de Culturas

## Visão Geral do Projeto

Sistema de classificação de culturas agrícolas usando imagens temporais de satélite (talhoes) com arquiteturas deep learning.

---

## Implementação Atual

### Modelo: EfficientNetB0 + FiLM + Temporal Attention (V7)

**Arquitetura:**
```
Para cada talhão com até MAX_SEQ_LEN imagens temporais:
1. EfficientNetB0 (timm, pretrained) extrai features 1280-dim por imagem
2. FiLM conditioning: concat(dia_normalizado, mes_embedding_8d) -> Dense(64)
   -> gamma, beta que modulam as features: x * (1+gamma) + beta
3. 2 camadas de MultiHeadAttention cruzam info entre timesteps
4. Mean-pooling dos tokens válidos -> Dense(256) -> classificação
```

**Características:**
- Mes tratado como CATEGÓRICO via Embedding(12, 8)
- gamma/beta são zero-initialized → começa como identidade
- Suporte AMP (mixed precision FP16)
- torch.compile
- cudnn.benchmark
- Classificação: 5 culturas (soja, milho, trigo, aveia, feijão)

**Arquivo:** `src/models/efficientnet_v7/train.py`

---

## Trabalhos Futuros

### Fase 1: Ensemble de Modelos com XGBoost

Testar e treinar múltiplos backbones com atenção temporal para extrair representações complementares:

```
├─── EfficientNetB0 + Temporal Attention (V7)  ─── features + probs
├─── ResNet50 + Temporal Attention              ─── features + probs
├─── ConvNeXt-Tiny + Temporal Attention         ─── features + probs
│
└─── XGBoost (learns optimal combination)  ─── final prediction
```

**Estratégia:**
1. Treinar cada backbone com a mesma arquitetura temporal (FiLM + Attention)
2. Extrair featureslate e probabilidades (logits) de cada modelo
3. Concatenar outputs → treinar XGBoost para combinação ótima
4. Objetivo: melhorar acurácia através de diversidade de representações

### Fase 2: Expansão de Dados + Otimização Genética

Se a Fase 1 não melhorar significativamente:

1. **Coleta de mais dados:** Treinar EfficientNet com dataset expandido
2. **Algoritmo Genético:** Desenvolver GA para otimização de:
   - Hyperparameters (learning rate, dropout, etc.)
   - Arquitetura (número de camadas, attention heads)
   - Estratégia de data augmentation

---

## Métricas de Sucesso

- Acurácia > 90% no conjunto de validação
- F1-macro > 0.88
- XGBoost ensemble demonstrando melhoria > 2% sobre modelo único
# Arquitetura dos Modelos de Classificação de Culturas

Na primeira fase da análise de arquitetura sera feita a classificação de apenas 3 culutras e utilizando um dataset menor, para fim de se escolher a arquitura que se adapta melhor ao problema e assim podemos partir para a fase de seleção dos hiperparametros, podendo ser utilizado um algoritmo genêtico, grid search etc)

**Análise inicial das arquiteturas**

Classificar culturas agrícolas (milho, soja, trigo) a partir de imagens de satélite processadas e mascaradas ao contorno do talhão.

Cada talhão possui até 3 imagens capturadas em diferentes estágios de crescimento após o plantio:

| Cultura | Datas de observação (dias após plantio) |
| --- | --- |
| Milho | d21, d31, d56 |
| Soja | d21, d31, d56 |
| Trigo | d26, d32, d47 |

---



Pipeline de Dados

```
KMLs (polígonos dos talhões)
  └─→ Sentinel Hub API (download das imagens por data)
        └─→ máscara KML aplicada (recorta ao contorno do talhão)
              └─→ ./processadas/mascara_{uuid}_v_d{dia}.png
                    └─→ SQLite (cultura, mês, lista de caminhos)
                          └─→ gerar_sample_treino.py (amostra balanceada para analisar previamente as arquiteturas)
                                └─→ gerar_db_completo.db (amostra com limpeza dos dados, utilizada para arquitetura final)
```



### **Análise de Baseline: Serviço Focusnet**

O modelo atual está operando num nível de acurácia pouco superior ao mero acaso e com uma latência que inviabiliza o seu uso em produção.

**Parâmetros do Teste**

* **Tamanho da Amostra:** 482 requisições completadas e avaliadas.
* **Distribuição:** Dataset de teste equilibrado (aprox. 96 a 97 amostras por cultura).
* **Latência:** Média de 369,4 segundos (\~6,2 min) por requisição (picos de 9 minutos).
* **Desempenho Global:** Acurácia de apenas 21,78% (105 acertos em 482) e F1-Score Macro de 0,1521.

**Resultados Detalhados por Cultura**

| **Cultura** | **Amostras** | **Acertos (TP)** | **Falsos Positivos** | **Taxa de Acerto (Recall)** | **F1-Score** | **Diagnóstico** |
| --- | --- | --- | --- | --- | --- | --- |
| **Aveia** | 97 | 63 | 247 | 65,0% | 0,3096 | Falso Positivo Crítico |
| **Trigo** | 96 | 27 | 88 | 28,1% | 0,2559 | Baixa Confiança |
| **Feijão** | 97 | 15 | 42 | 15,5% | 0,1948 | Baixa Confiança |
| **Milho** | 96 | 0 | 0 | 0,0% | 0,0000 | Falha Estrutural |
| **Soja** | 96 | 0 | 0 | 0,0% | 0,0000 | Falha Estrutural |

### **Causas Raiz do Colapso**

Um resultado de exatamente 0% de acerto numa amostra de quase 200 itens (Milho e Soja) indica que o problema transcende a simples "dificuldade de classificação do modelo". Os indícios apontam para:

1. **Colapso do Modelo (*Mode Collapse*):** O modelo desenvolveu um viés extremo em que utiliza "Aveia" como resposta padrão (default) sempre que não consegue extrair características úteis da imagem. Isto explica os 247 falsos positivos nesta classe e a supressão total das outras.
2. **Erro de Pipeline ou Mapeamento:** O zero absoluto sugere um provável erro de engenharia de dados no atual provedor, como o descarte de características (features) cruciais para as classes de verão ou um erro no mapeamento de *labels* na

### **Proposta de Nova Arquitetura: Modelo (Deep Learning Multi-Temporal)**

A nova abordagem testada utiliza um modelo preditivo baseado no histórico temporal do talhão agrícola. O pipeline (desenvolvido em PyTorch) processa sequências de até 3 imagens Sentinel-hub, utilizando metadados de plantio para contornar a similaridade visual inicial entre as culturas.

#### **1. Dados de Entrada (Por Talhão)**

O modelo abandona a inferência estática e passa a exigir uma sequência temporal associada a metadados climáticos/sazonais.

| **Variável** | **Formato/Tipo** | **Função no Modelo** |
| --- | --- | --- |
| **images** | (T, 3, 224, 224) | Sequência de até 3 imagens RGB normalizadas. |
| **dias** | float32 (T,) | Dias após o plantio (normalizado). |
| **mes** | int64 categórico | Mês de plantio (0-11). Captura o ciclo sazonal/safra. |
| **mask** | float32 (T,) | Máscara de padding (1 = real, 0 = ausente). |

#### **2. Arquitetura da Rede**

A rede processa a dimensão espacial e temporal em três estágios principais:

1. **Extração de Features (Espacial):** Utiliza uma **EfficientNetB0** (pré-treinada em ImageNet). Os pesos são partilhados ao longo do tempo, gerando um vetor de 1280 dimensões para cada imagem individual da sequência.
2. **Modulação FiLM (Sazonal/Temporal):** Os embeddings do mês de plantio e dos dias decorridos são usados para condicionar as features visuais extraídas. O modelo ajusta dinamicamente a importância das características da imagem usando a equação matemática:
   $tokens = features \\times (1 + \\gamma) + \\beta$
   *(Onde $\gamma$ e $\beta$ são iniciados a zero, garantindo estabilidade no início do treino).*
3. **Atenção Temporal (Agregação):** Duas camadas de **MultiHeadAttention** (8 cabeças) avaliam a sequência temporal, ignorando imagens ausentes via *key padding masks*. Um *Mean Pooling* consolida as leituras numa única representação antes da classificação final (Linear -\> Dropout -\> 5 logits).

#### **3. Estratégia de Treino (Otimização e Prevenção de Viés)**

Para evitar o *mode collapse* (o viés extremo de classificar tudo como Aveia observado na baseline), o treino implementa táticas agressivas de regularização:

* **Label Smoothing (0.1):** Reduz a tendência do modelo em fases de crescimento onde as plantas são indistinguíveis.
* **Treino em 2 Fases:**
  * \* *Fase 1 (10 épocas):* Extrator base congelado, treinando apenas a cabeça de atenção e modulação.
  * *Fase 2 (15 épocas):* Descongelamento das últimas 20 camadas da EfficientNet para *fine-tuning* no domínio agrícola.
* **Otimização de Hardware:** Uso nativo de AMP (Precisionamento Misto) e *DataLoaders* em *pin_memory* para reduzir o gargalo de latência.

#### **4. Resultados de Validação**

Os testes foram realizados em uma amostra contendo um intervalo de 2500 a 9000 talhões para cada cultura (contendo 3 imagens cada). Os resultados obtidos estão descritos na tabela abaixo.

| **Tamanho da Amostra** | **Acurácia Global** | **F1-Macro** | **F1 Soja** | **F1 Milho** | **F1 Trigo** | **F1 Aveia** | **F1 Feijão** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Máx. 2500 | 85,7% | 0,856 | 0,820 | 0,719 | 0,995 | 0,990 | 0,756 |
| **Máx. 6000** | **87,7%** | **0,876** | **0,853** | **0,754** | **1,000** | **0,997** | **0,776** |
| Máx. 9000 | 85,5% | 0,854 | 0,822 | 0,738 | 1,000 | 0,994 | 0,718 |

**Conclusão Técnica:** O modelo apresentou resultados definitivamente melhores do que o modelo atual. Apesar de que ainda apresenta dificuldades em diferenciar culturas como milho, feijão e soja. O que deve ser otimizado via aumento da quantidades de dados ou ajuste nos hiperparâmetros.

#### **5. Validação Cruzada 5-Fold (V7 — Dataset Max. 6000)**

Avaliação de robustez e generalização via 5-fold cross-validation no conjunto com até 6000 amostras por cultura.

| **Fold** | **Acurácia** | **F1-Macro** | **Tempo Médio (ms/amostra)** | **GPU Peak (MB)** |
| --- | --- | --- | --- | --- |
| Fold 1 | 85,44% | 0,8529 | 87,31 | 2082,5 |
| Fold 2 | 86,24% | 0,8609 | 37,96 | 1319,3 |
| Fold 3 | 83,60% | 0,8348 | 39,54 | 1319,3 |
| Fold 4 | 85,44% | 0,8539 | 37,53 | 1319,3 |
| Fold 5 | 84,80% | 0,8473 | 37,82 | 1319,3 |
| **Média** | **85,10% ± 0,88%** | **0,8500 ± 0,0087** | **48,03** | **1471,97** |

**Interpretação:**
- Baixo desvio padrão (±0,88% acc / ±0,0087 F1) indica que o modelo generaliza de forma consistente entre os folds.
- Tempo médio de inferência de ~48 ms/amostra é ordens de magnitude inferior ao baseline da FocusNet (~369.400 ms).
- O Fold 1 apresenta consumo de GPU ligeiramente superior (2082 MB vs ~1319 MB nos demais), possivelmente por efeito de alocação inicial do CUDA.

#### **6. Métricas de Treino por Fold (artifacts_5_folds)**

Resultados registrados ao final do treino de cada modelo individual, incluindo F1 por classe e tempo de treino.

| **Fold** | **Acurácia** | **F1-Macro** | **F1 Soja** | **F1 Milho** | **F1 Trigo** | **F1 Aveia** | **F1 Feijão** | **Tempo Treino** | **Inf. (ms)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fold 1 | 84,23% | 0,8588 | 0,8156 | 0,7164 | 0,9994 | 0,9951 | 0,7676 | 39,7 min | 4,57 |
| Fold 2 | 84,08% | 0,8546 | 0,8042 | 0,7020 | 0,9982 | 0,9871 | 0,7814 | 36,4 min | 4,64 |
| Fold 3 | 83,70% | 0,8483 | 0,8029 | 0,7067 | 0,9988 | 0,9630 | 0,7703 | 55,7 min | 6,19 |
| Fold 4 | 84,51% | 0,8610 | 0,8116 | 0,7260 | 0,9994 | 0,9935 | 0,7745 | 54,3 min | 5,55 |
| Fold 5 | 84,35% | 0,8583 | 0,8079 | 0,7236 | 0,9991 | 0,9871 | 0,7736 | 44,3 min | 4,93 |
| **Média** | **84,17%** | **0,8562** | **0,8084** | **0,7149** | **0,9990** | **0,9852** | **0,7735** | **46,1 min** | **5,18** |

**Observações por cultura:**
- **Trigo e Aveia:** F1 próximo de 1,0 e 0,99 respectivamente — classes bem separadas espectralmente.
- **Soja:** F1 médio de 0,81 — distinguível, mas com margem de melhora.
- **Feijão:** F1 médio de 0,77 — classe mais desafiadora junto ao Milho.
- **Milho:** F1 médio de 0,71 — pior desempenho consistente; confusão provável com Soja e Feijão em fases iniciais de crescimento.
- O Fold 3 destaca-se negativamente em Aveia (0,963) e apresentou o maior tempo de treino (55,7 min), sugerindo possível variação no split ou maior complexidade no subset de dados.

## Trabalhos Futuros

### Fase 1:  Ajuste no serviço atual  (Fase crítica)

Será necessário realizar ajustes na forma como o serviço atual da Focus Net trata os dados recebidos do usuário, melhorando o tratamento de erros e as chamadas aos serviços externos.
Além disso, é importante aprimorar o tratamento de erros na aquisição de imagens e avaliar a necessidade de alterar a estratégia de chamadas, com o objetivo de reduzir o tempo de espera do cliente.

### Fase 2: Ensemble de Modelos com XGBoost

Testar e treinar múltiplos backbones com atenção temporal para extrair representações complementares:

```
├─── EfficientNetB0 + Temporal Attention (V7)  ─── features + probs
├─── ResNet50 + Temporal Attention              ─── features + probs
├─── ConvNeXt-Tiny + Temporal Attention         ─── features + probs
│
└─── XGBoost (learns optimal combination)  ─── final prediction
```

**Estratégia:**

1. Treinar cada backbone com a mesma arquitetura temporal (FiLM + Attention)
2. Extrair featureslate e probabilidades (logits) de cada modelo
3. Concatenar outputs → treinar XGBoost para combinação ótima
4. Objetivo: melhorar acurácia através de diversidade de representações

### Fase 3: Expansão de Dados + Otimização Genética

Se a Fase 2 não melhorar significativamente:

1. **Coleta de mais dados:** Treinar EfficientNet com dataset expandido
2. **Algoritmo Genético:** Desenvolver GA para otimização de:
   * Hyperparameters (learning rate, dropout, etc.)
   * Arquitetura (número de camadas, attention heads)
   * Estratégia de data augmentation

---

## Métricas de Sucesso

* Acurácia \> 90% no conjunto de validação
* F1-macro \> 0.88
* XGBoost ensemble demonstrando melhoria \> 2% sobre modelo único