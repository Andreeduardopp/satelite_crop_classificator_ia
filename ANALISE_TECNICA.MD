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
