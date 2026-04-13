# Classificador de Culturas — Arquitetura da Rede

Documentação técnica do pipeline de deep learning para classificação de
**milho / soja / trigo** a partir de imagens Sentinel-2 multi-temporais.

---

## 1. Problema

Dado um **talhão** agrícola, identificar qual cultura foi plantada usando
até 3 imagens de satélite tiradas em diferentes dias após o plantio.

**Por que é difícil?**
- Milho e soja são **visualmente indistinguíveis** em RGB nos primeiros 30–56 dias
- A janela de observação (d21, d31, d56) cobre fases de crescimento similares
- RGB (3 bandas visíveis) não captura diferenças de canópia no infravermelho
- O modelo precisa lidar com sequências de comprimento variável (1–3 imagens)

---

## 2. Dados de Entrada

### Por talhão (uma amostra de treino)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `images` | `float32 (3, 224, 224, 3)` | Até 3 fotos RGB em sequência temporal |
| `dias` | `float32 (3,)` | Dia após plantio de cada foto, normalizado / 100 |
| `mes` | `float32 ()` | Mês de plantio (1–12), escalar por talhão |
| `mask` | `float32 (3,)` | 1 = timestep real, 0 = padding |

As imagens vêm de `./processadas/mascara_{uuid}_v_d{dia}.png` — recortes
do talhão sobre a imagem Sentinel-2, aplicados via máscara KML.

### Base de dados

- **Treino**: `sample_treino_6k.db` — 6.000 talhões por cultura (18k total)
- **Split**: 80% treino / 20% validação, estratificado por cultura
- **Coluna `mes`**: mês do plantio, disponível no banco, usado a partir do v6

---

## 3. Arquitetura — Visão Geral

```
┌─────────────────────────────────────────────────────────────────┐
│                      POR TIMESTEP (compartilhado)               │
│                                                                 │
│  imagem (224×224×3)                                             │
│       │                                                         │
│  [Augmentation]  ← só no treino                                 │
│  flip, rotation, zoom, contrast, brightness                     │
│       │                                                         │
│  [EfficientNetB0]  ← pesos ImageNet, backbone compartilhado     │
│  7×7×1280 feature map                                           │
│       │                                                         │
│  [GlobalAveragePooling2D]                                       │
│  → vetor 1280-dim por timestep                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
         features (B, T=3, 1280)
                          │
┌─────────────────────────────────────────────────────────────────┐
│                   FiLM CONDITIONING (v6)                        │
│                                                                 │
│  mes (B,) → Embedding(12, 8) → tile × T → (B, T, 8)            │
│  dias (B, T) → expand → (B, T, 1)                               │
│                                                                 │
│  context = concat([dia, mes_emb]) → (B, T, 9)                   │
│       │                                                         │
│  Dense(64, relu) → γ (B, T, 1280)  [zero-init]                  │
│                 → β (B, T, 1280)  [zero-init]                   │
│                                                                 │
│  tokens = features × (1 + γ) + β                                │
└─────────────────────────────────────────────────────────────────┘
                          │
         tokens (B, T=3, 1280)
                          │
┌─────────────────────────────────────────────────────────────────┐
│                  TEMPORAL ATTENTION (2 camadas)                 │
│                                                                 │
│  [MultiHeadAttention]  8 cabeças, key_dim=160, dropout=0.1      │
│  [LayerNorm + residual]                                         │
│                                                                 │
│  [MultiHeadAttention]  8 cabeças, key_dim=160, dropout=0.1      │
│  [LayerNorm + residual]                                         │
│                                                                 │
│  mask impede timesteps de padding de contribuir                 │
└─────────────────────────────────────────────────────────────────┘
                          │
         mean pool sobre timesteps válidos → (B, 1280)
                          │
┌─────────────────────────────────────────────────────────────────┐
│                     CABEÇA DE CLASSIFICAÇÃO                     │
│                                                                 │
│  Dense(256, relu) → Dropout(0.3)                                │
│  Dense(3, softmax)                                              │
│       │                                                         │
│  [milho, soja, trigo] — probabilidades                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Componentes Detalhados

### 4.1 EfficientNetB0 (Extração de Features)

- Pré-treinado no ImageNet (~5.3M parâmetros)
- Aceita entrada `(224, 224, 3)` normalizada pelo preprocess interno
- Aplicado **individualmente** a cada timestep com **pesos compartilhados**
  (mesma rede processa d21, d31 e d56 do mesmo talhão)
- Fase 1: backbone **congelado** — só a cabeça treina
- Fase 2: **últimas 20 camadas** descongeladas para fine-tuning

### 4.2 FiLM — Feature-wise Linear Modulation

FiLM é uma técnica de condicionamento que usa informação auxiliar (aqui,
dia + mês) para **modular as features visuais**:

```
tokens = features × (1 + γ) + β
```

Onde γ e β são gerados a partir do contexto temporal:

```
context = [dia_normalizado | mes_embedding]  → shape (9,)
       ↓
Dense(64, relu)
       ↓
γ = Dense(1280)  — ganho multiplicativo
β = Dense(1280)  — deslocamento aditivo
```

**Por que não apenas concatenar `mes` à feature final?**
Concatenar no final deixa o modelo decidir *se* usa o mês. FiLM força o
modelo a perguntar: *"dado que o plantio foi em outubro, quais features
visuais devo amplificar ou suprimir?"* — a informação temporal permeia
toda a representação.

**Inicialização zero de γ e β:**
No início do treino, γ=0 e β=0, então `tokens = features × 1 + 0 = features`.
O modelo começa exatamente como sem FiLM e aprende gradualmente quanto
usar o contexto. Isso evita instabilidade no início e permite convergência
estável.

### 4.3 Mes Embedding (novidade do v6)

O mês de plantio é tratado como variável **categórica**, não contínua:

```python
Embedding(input_dim=12, output_dim=8)   # índices 0-11
```

**Por que embedding e não escalar normalizado?**
Dezembro não é "mais que" novembro em termos fenológicos — o ciclo
agrícola é circular. Um embedding aprendido pode capturar que
setembro/outubro/novembro se comportam como um cluster (pico de
plantio de soja) independentemente da ordem numérica.

O embedding de 8 dimensões (96 parâmetros totais) é pequeno o suficiente
para não overfitar e grande o suficiente para representar padrões sazonais.

### 4.4 Temporal Attention

Após o FiLM, os 3 tokens (um por timestep) passam por 2 camadas de
atenção multi-head:

- A atenção permite que o modelo aprenda **qual timestep é mais informativo**
  para cada cultura. Por exemplo: para trigo, o d100 (senescência) é muito
  discriminativo; para milho/soja nos d21-d31, o modelo pode aprender a
  dar mais peso ao d56.
- A **attention mask** garante que timesteps de padding (mask=0) não
  influenciem os timesteps reais.
- Ao final, **mean pooling** sobre os timesteps válidos produz um vetor
  fixo de 1280-dim independente do número de observações.

---

## 5. Treinamento

### Fase 1 — Backbone Congelado

| Parâmetro | Valor |
|-----------|-------|
| Epochs | 10 (com early stopping patience=3) |
| LR | 1e-3 |
| Treináveis | FiLM + mes_embedding + temporal attention + cabeça |

O backbone EfficientNet permanece congelado. O modelo aprende a usar
as features ImageNet + contexto temporal para classificar.

### Fase 2 — Fine-tuning

| Parâmetro | Valor |
|-----------|-------|
| Epochs | 15 (com early stopping patience=4) |
| LR | 1e-5 |
| Treináveis | Últimas 20 camadas do backbone + tudo da Fase 1 |
| ReduceLROnPlateau | factor=0.5, patience=2, min_lr=1e-7 |

LR muito menor para não destruir os pesos ImageNet — apenas ajuste fino
para o domínio de imagens de satélite agrícola.

### Loss e Regularização

```python
loss = CategoricalCrossentropy(label_smoothing=0.1)
```

- **Label smoothing 0.1**: suaviza os targets de [0,1] para [0.033, 0.9].
  Evita que o modelo fique excessivamente confiante em classes ambíguas
  (milho/soja com aparência similar).
- **Class weights**: calculados automaticamente para compensar qualquer
  desbalanceamento entre culturas.
- **Dropout(0.3)**: regularização na cabeça de classificação.

---

## 6. Histórico de Versões e Resultados

| Versão | Novidade | Val Acc | F1 macro | F1 milho | F1 soja | F1 trigo |
|--------|----------|---------|----------|----------|---------|---------|
| v2 (EfficientNet) | Baseline temporal com MHA | ~70% | 0.70 | 0.54 | 0.53 | 1.00 |
| v3 | 6k/classe + FiLM(dia) + augment + label smooth | ~70% | 0.70 | ~0.54 | ~0.54 | 1.00 |
| XGBoost (sigmoids) | Tabular com mes + last_dia | 90.8% | 0.908 | 0.856 | 0.869 | 1.00 |
| v5 (CNN+XGBoost) | MobileNetV3 emb + XGBoost | 88.3% | 0.882 | 0.813 | 0.834 | 1.00 |
| **v5** | **MobileNetV3 + FiLM(dia) + Late Fusion(mes)** | ? | ? | ? | ? | ? |
| **v6** | **v3 + FiLM(dia, mes_embedding)** | ? | ? | ? | ? | 1.00 |

**Diagnóstico do teto em 70%:**
Ambos os modelos v2 e v3 atingem o mesmo teto. O XGBoost sobre vetores
sigmoides (features tabulares de outros modelos) chega a 91% — a diferença
principal é que ele tem acesso a `mes`. O v6 testa essa hipótese diretamente.

---

## 7. Por Que `mes` É o Feature Mais Importante

O XGBoost revelou que `last_dia` (22% de importância) e `mes` (10.3%)
dominam a classificação. Por que?

**Soja** é plantada quase exclusivamente em setembro/outubro/novembro
(91% das amostras). **Milho** tem dois ciclos: verão (set/out) e safrinha
(jan/fev). **Trigo** é de inverno (abr/mai/jun).

Isso significa:
- Um talhão com `mes = fevereiro` **não pode ser soja** (quase certamente é
  milho safrinha ou trigo de ciclo tardio)
- Um talhão com `mes = outubro` pode ser milho ou soja, mas a distribuição
  muda muito

O CNN v3 via apenas imagens RGB + dia relativo ao plantio. Sem saber
o mês absoluto, ele não consegue usar essa informação fenológica crítica.

---

## 8. Arquivos Relevantes

| Arquivo | Descrição |
|---------|-----------|
| [src/treinamento/treinar_classificador_v6.py](src/treinamento/treinar_classificador_v6.py) | Script de treino v6 (este modelo) |
| [src/treinamento/treinar_classificador_v3.py](src/treinamento/treinar_classificador_v3.py) | Baseline v3 (sem `mes`) |
| [src/treinamento/treinar_xgboost_sigmoides.py](src/treinamento/treinar_xgboost_sigmoides.py) | XGBoost tabular (referência 91%) |
| [src/dados/processamento_sentinel_indices.py](src/dados/processamento_sentinel_indices.py) | Downloader NDVI/NDWI/NDBI (para v4) |
| [DATASET_IMPROVEMENT_PLAN.md](DATASET_IMPROVEMENT_PLAN.md) | Plano de melhoria do dataset |
| `sample_treino_6k.db` | 18k talhões com coluna `mes` |
