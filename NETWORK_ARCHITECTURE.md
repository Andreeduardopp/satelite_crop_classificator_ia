# Classificador de Culturas — Arquitetura da Rede

Documentação técnica do pipeline de deep learning para classificação de **5 culturas (soja, milho, trigo, aveia e feijão)** a partir de imagens Sentinel-2 multi-temporais.

---

## 1. Problema

Dado um **talhão** agrícola, identificar qual cultura foi plantada usando até 3 imagens de satélite tiradas em diferentes dias após o plantio.

**Por que é difícil?**
- Milho, soja e feijão muitas vezes apresentam assinaturas visuais similares no estágio de desenvolvimento inicial.
- A janela de observação cobre fases de crescimento que visualmente se confundem.
- RGB (3 bandas visíveis) não captura diferenças de canópia no infravermelho de forma explícita, logo dependemos fortemente da extração de boas features texturais e de coloração pelo modelo.
- O modelo precisa lidar com sequências de comprimento variável (1–3 imagens por talhão).

---

## 2. Dados de Entrada

### Por talhão (uma amostra de treino)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `images` | `float32 (T, 3, 224, 224)` | Até 3 fotos RGB em sequência temporal (normalizadas) |
| `dias` | `float32 (T,)` | Dia após plantio de cada foto, normalizado (`/ 100`) |
| `mes` | `int64 ()` | Mês de plantio (1–12), tratado como variável categórica (índice 0-11) |
| `mask` | `float32 (T,)` | 1 = timestep real, 0 = padding |

As imagens advêm de recortes do talhão sobre as composições do Sentinel-2, aplicadas via máscara KML e processadas.

### Base de dados
- **Treino e Validação**: Diversos bancos de dados em progressão, com o mais recente sendo o conjunto até `max9000.db`.
- **Split**: 80% treino / 20% validação, estratificado por cultura para manter as proporções representativas nas classes menos prevalentes.

---

## 3. Arquitetura — Visão Geral

### 3.1 V7 - Pipeline End-to-End (PyTorch)

```text
┌─────────────────────────────────────────────────────────────────┐
│                      POR TIMESTEP (compartilhado)               │
│                                                                 │
│  imagem (3×224×224)                                             │
│       │                                                         │
│  [EfficientNetB0 (timm)]  ← pesos ImageNet, compartilhado       │
│  → GlobalAveragePooling interno                                 │
│  → vetor 1280-dim por timestep                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
         features (B, T=3, 1280)
                          │
┌─────────────────────────────────────────────────────────────────┐
│                   FiLM CONDITIONING                             │
│                                                                 │
│  mes (B,) → nn.Embedding(12, 8) → expand × T → (B, T, 8)        │
│  dias (B, T) → expand → (B, T, 1)                               │
│                                                                 │
│  context = concat([dia, mes_emb]) → (B, T, 9)                   │
│       │                                                         │
│  Linear(64, relu) → γ (B, T, 1280)  [zero-init]                 │
│                   → β (B, T, 1280)  [zero-init]                 │
│                                                                 │
│  tokens = features × (1 + γ) + β                                │
└─────────────────────────────────────────────────────────────────┘
                          │
         tokens (B, T=3, 1280)
                          │
┌─────────────────────────────────────────────────────────────────┐
│                  TEMPORAL ATTENTION (2 camadas)                 │
│                                                                 │
│  [MultiHeadAttention]  8 cabeças, dropout=0.1                   │
│  [LayerNorm + residual]                                         │
│                                                                 │
│  [MultiHeadAttention]  8 cabeças, dropout=0.1                   │
│  [LayerNorm + residual]                                         │
│                                                                 │
│  (key_padding_mask impede influências de timesteps com padding) │
└─────────────────────────────────────────────────────────────────┘
                          │
         mean pool sobre timesteps válidos → (B, 1280)
                          │
┌─────────────────────────────────────────────────────────────────┐
│                     CABEÇA DE CLASSIFICAÇÃO                     │
│                                                                 │
│  Linear(256, relu) → Dropout(0.3)                               │
│  Linear(5) (logits)                                             │
│       │                                                         │
│  [soja, milho, trigo, aveia, feijão] — logits                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Híbrido XGBoost V1 (Two-Stage)

Uma abordagem alternativa implementada para ganerar inferências mais limpas e extrair um grau de compreensibilidade maior.
O pipeline se divide em:
1. **Extrator de Features** (`extrator.py`): O modelo base pré-treinado V7 processa o banco e exporta as características visuais extraídas para todos os talhões, achatando-as sequencialmente numa grade linear gigantesca: `[Feature1(1280), Feature2(1280), Feature3(1280), Dia1, Dia2, Dia3, Mês, Count_img]`. Isso resulta num tabular imenso com 3.845 colunas puras.
2. **Algoritmo de Árvore** (`train_xgboost.py`): Como o dado já não é mais imagem, entra-se com o XGBoost com as features tabuladas em RAM em pouquíssimo tempo. O algoritmo ganha vantagens naturais clássicas sobre as ramificações visuais.

---

## 4. Componentes Detalhados

### 4.1 EfficientNetB0 (Extração de Features)

- Pré-treinado no ImageNet (via pacote de visão `timm`).
- Entradas de tamanho `(3, 224, 224)` são normalizadas perfeitamente de acordo com as estatísticas exigidas pelo ImageNet.
- A função conv passará a observar **cada imagem sequencialmente e individualmente** com **pesos compartilhados** ao longo do tempo (compartilhamento de pesos de feature extractor).
- **Fase 1**: modelo base **congelado** — treina-se apenas a cabeça classificatória, o módulo de modulação e atenção.
- **Fase 2**: as **últimas 20 camadas** do extrator basal são descongeladas para fine-tuning orientado aos dados agrícolas locais.

### 4.2 FiLM — Feature-wise Linear Modulation

FiLM é um mecanismo de modulação em que vetores auxiliares (contexto temporal local, neste caso mês e dia) afetam **linear e intrinsecamente as características visuais** logo que saem do extrator:

```
tokens = features × (1 + γ) + β
```

Onde γ (ganho) e β (desvio) são derivados sob contexto:
`context = [dia_normalizado | mes_embedding]` -> `Linear(64, relu)` -> `Linear(1280)`.

Ambos γ e β são inicializados deliberadamente como zero no começo. Em outras palavras, `tokens = features × 1 + 0 = features`. Assim o fine-tuning temporal começa ameno como uma rede de extração simples, tornando a aprendizagem regularizada visando maior robustez no descobrimento de pesos não-lineares temporais.

### 4.3 Mes Embedding (Sazonalidade)

O mês do plantio é considerado uma feição **categórica**, não linear escalada:

```python
nn.Embedding(12, 8)
```
E não um valor linear de 1 a 12 porque fenologicamente a agricultura não lida com um encadeamento linear onde dezembro seja intrinsecamente "muito distanciado" de janeiro. A relação sazonal é cíclica por safras, o que a camada de embedding se torna excelente em extrair (ex: agrupar padrões no cluster de safrinha / inverno).

### 4.4 Temporal Attention (Atenção sobre Observações)

Uma série de `nn.MultiheadAttention` capturam a dependência de um estágio final sobre uma observação particular efetuada dentro do limiar observacional.
- Timesteps faltantes (sequências em que a amostra teve apenas 2 das 3 imagens obtidas) são mascarados com _key padding masks_ para não arruinar os coeficientes de atenção.
- Após o cruzamento multi-nível, faz-se agregamento _Mean pooling_ em todos os timesteps ativos, obtendo um único grande sumário que unifica as leituras orbitais na janela.

---

## 5. Treinamento na Arquitetura (PyTorch/CUDA)

Com a arquitetura reescrita em PyTorch, diversas otimizações nativas de GPU/CUDA de baixo nível tomam protagonismo como:
- **AMP (`torch.amp`)** (Precisionamento de tipos matemáticos Misturado para cálculos matriciais mais densos)
- **cuDNN benchmark embutido** para escolha rápida na rotina das passagens por convulsão.
- **DataLoaders concorrentes com buffers em pin_memory (`non_blocking=True`)**.

### Fluxo de Épocas e Congelamentos

**Fase 1 — Extrator Congelado:**
- Resumo de hiperparâmetros: 10 épocas e sub-patience=3 para validação.
- _LR = 1e-3_
- Componentes treináveis: apenas Attention, Cabeça Densa Superior, FiLM Module, e Mês Embs.

**Fase 2 — Modulação Fina:**
- Resumo de hiperparâmetros: 15 épocas e sub-patience=4.
- _LR = 1e-5_
- Componentes treinados: os anteriores unidos agora às últimas **20 camadas** da EfficientNetB0 sendo lentamente refinadas do domínio ImageNet para o domínio óptico agrícola.

### Função Custos e Combate a Enviezamento

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```
- A prática de *label smoothing* (`0.1`) é crítica pois ameniza classificações dicotomicamente dogmáticas do modelo num grupo onde soja/milho jovens são altamente irreconhecíveis em partes isoladas.
- Aplicação ativa de _Class Weights_ dentro da CrossEntropyLoss de forma automatizada por lotes para combater eventuais assimetrias.

---

## 6. Histórico de Versões e Resultados Recentes

O registro demonstra a transição crucial da arquitetura para a de **v7**, capaz de distinguir **5 culturas** (em contraponto aos trabalhos de base de 3 categorias). Abaixo encontram-se as métricas consolidadas sobre os artefatos gerados:

| Versão | Novidade / Estruturação | Acc global | F1-macro | F1 soja | F1 milho | F1 trigo | F1 aveia | F1 feijão |
|--------|--------------------------|------------|----------|---------|----------|----------|----------|-----------|
| **v2** | CNN Temporal (3 classes) | ~70% | 0.700 | 0.53 | 0.54 | 1.00 | - | - |
| **XGBoost** | XGBoost nas relativas | 90.8% | 0.908 | 0.869 | 0.856 | 1.00 | - | - |
| **v6** | FiLM de calendário (TF) | - | - | - | - | - | - | - |
| **v7** | PyTorch, **5 Classes, 2500 max samples** | 85.7% | 0.856 | 0.820 | 0.719 | 0.995 | 0.990 | 0.756 |
| **v7** | PyTorch, **5 Classes, 6000 max samples** | **87.7%** | **0.876** | **0.853** | **0.754** | **1.000** | **0.997** | **0.776** |
| **v7** | PyTorch, **5 Classes, 9000 max samples** | 85.5% | 0.854 | 0.822 | 0.738 | 1.000 | 0.994 | 0.718 |
| **Híbrido v1** | Extrator DL bruto + XGBoost Tabular | 85.2% | 0.850 | 0.828 | 0.703 | 1.000 | 1.000 | 0.722 |

> **Análise do Híbrido V1:** O Híbrido V1 não superou a versão puramente neural (`v7`), alcançando F1-macro de 0.850 (versus 0.876). Uma causa documentada é a forma como o `extrator.py` original extrai a predição da EfficientNet. Ele obtém o _backbone bruto_ (ignorando as essenciais camadas FiLM e Temporal Attention do V7) e concatena no longo eixo espacial (3x1280), delegando todo o trabalho de entendimento seqüencial ao XGBoost, desprivilegiando interações temporais vitais de plantio da soja e milho (que caíram em F1).

> **Observação dos resultados (v7):** Observamos que a arquitetura em conjunto ao embedding alcança performances notórias quase absolutas para `trigo` e `aveia` graças também aos claros marcadores fenológicos e desvios de plantio. Apesar da extrema confusão inerente que paira sobre as classes `soja`, `milho` e `feijão` nas estepes do dia a dia, mantivemos e elevamos o sucesso nos exames amplos girando firmemente a baliza em ~85%-87% de F1-macro com base de treinamento alargada.
