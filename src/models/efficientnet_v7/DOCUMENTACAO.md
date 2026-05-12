# Documentação — EfficientNet V7 (FiLM + Temporal Attention)

Este documento descreve a arquitetura, o pipeline de dados e o treinamento do modelo de
classificação de culturas implementado em [train.py](train.py).

A rede V7 trabalha com **séries temporais de imagens de satélite** (até 3 imagens por talhão),
condicionadas pelo dia após plantio (DAP) e pelo mês de plantio. A construção do dataset
acontece em duas grandes fases — aquisição (Sentinel Hub) e processamento (máscara/limiar) —
detalhadas abaixo.

---

## 1. Aquisição e Processamento das Imagens

Toda a etapa de obtenção e tratamento das imagens vive em [src/dados/](../../dados/) e é
orquestrada por [src/pipeline.py](../../pipeline.py) através da função `baixar_imagens()`.
O fluxo macro é:

```
KML do talhão → Sentinel Hub → ./imagens/<ref>_d<dap>/<hash>/response.png   (bruta)
                              ↓
                       cria_mascara()  → ./mascaras/mascara_<ref>_d<dap>.png   (silhueta)
                              ↓
                       aplica_mascara() → ./processadas/mascara_<ref>_d<dap>.png  (entrada do modelo)
```

### 1.1 Entrada: KML do talhão

Cada talhão é descrito por um arquivo KML contendo o polígono do campo agrícola.
[gerador_dataframe.py](../../dados/gerador_dataframe.py) varre uma pasta de KMLs e produz
o `dataframe_20k.csv` extraindo, do nome do arquivo, os metadados:

| Coluna | Origem | Exemplo |
|---|---|---|
| `cultura` | Primeiro token do nome do arquivo | `soja`, `milho`, `trigo`, … |
| `ref_infra_v` | UUID + sufixo `_v` (chave única do talhão) | `a3f1b2c4_v` |
| `data` | Data de plantio do nome do arquivo | `2024-01-01` |
| `mes` | Mês extraído de `data` | `1`–`12` |
| `path` | Caminho absoluto do KML | `/.../SOJA_…_plantio_01-12-24_…kml` |

### 1.2 Download via Sentinel Hub

Função: `request_sentinel_hub()` em
[processamento_sentinel_Hub.py](../../dados/processamento_sentinel_Hub.py).

Para cada talhão, o pipeline solicita **uma imagem por DAP**, onde os DAPs são definidos
por cultura em `lista_datas_cultura()`:

| Cultura | DAPs (dias após plantio) |
|---|---|
| Soja, Milho, Feijão | 21, 31, 56 |
| Trigo | 26, 32, 47 |
| Aveia | 29, 44, 64 |
| Arroz | 15, 30, 95 |
| Café | 30, 50, 75, 100 |

Para cada DAP, [pipeline.py:152-157](../../pipeline.py#L152-L157) calcula
`data_dap = plantio + timedelta(days=dap)` e chama o Sentinel Hub com:

1. **Bounding box** calculado a partir das coordenadas do KML (min/max de lat/lon).
2. **Janela temporal** de 5 dias antes da data alvo (`data_dap - 5 dias` até `data_dap`),
   para tolerar dias com nuvens.
3. **Layer** `BANDAS_RBN` (NIR + Red + Blue, otimizado para vegetação) sobre `SENTINEL2_L2A`.
4. **Resolução** 10 m × 10 m, formato PNG.

A resposta é gravada em
`./imagens/<ref_infra_v>_d<dap>/<hash>/response.png`. Ainda dentro de
`request_sentinel_hub()`, `analisar_cobertura_de_nuvens()` ordena as candidatas e
**retorna apenas a melhor** (menor cobertura de nuvens) — por isso o slice `[:1]` no
final da função.

### 1.3 Máscara do polígono

Função: `cria_mascara()` em
[processamento_imagens.py](../../dados/processamento_imagens.py).

A partir do mesmo KML, o código:

1. Extrai latitudes e longitudes via `geo_json()`.
2. Plota o polígono **preenchido em preto** sobre fundo branco com matplotlib, sem eixos
   nem margens.
3. Salva em `./mascaras/mascara_<ref>_d<dap>.png`.

A máscara descreve apenas a *forma* do talhão; o redimensionamento para casar com a
imagem do satélite acontece na etapa seguinte.

### 1.4 Aplicação da máscara — `aplica_mascara()`

É aqui que a imagem bruta vira a entrada do modelo. As operações, na ordem em que ocorrem:

| Passo | Operação OpenCV | Efeito |
|---|---|---|
| 1 | `cv2.resize(imagem, mascara.shape)` | Alinha a resolução da imagem do satélite à da máscara |
| 2 | `cv2.cvtColor(mascara, BGR→GRAY)` | Reduz a máscara a 1 canal |
| 3 | `cv2.bitwise_not(mascara)` | Inverte: o talhão fica branco (255), o exterior preto (0) |
| 4 | `cv2.erode(kernel=5×5, iter=5)` | Encolhe a borda do talhão, eliminando pixels que vazam para fora do polígono |
| 5 | `cv2.bitwise_or(img, img, mask=erosão)` | Aplica a máscara: pixels fora do talhão viram zero |
| 6 | `cv2.imwrite(destino)` | Salva em `./processadas/mascara_<ref>_d<dap>.png` |

A imagem resultante tem **fundo preto** e mostra somente o conteúdo espectral
correspondente ao polígono do campo.

### 1.5 Variante com limiar — `treshold_indice()`

Existe também `treshold_indice()`, usada quando se quer manter apenas regiões com
índice acima de um limiar (`limite = max * 0.23`). O código aplica:

1. `cv2.threshold` binário sobre a imagem em escala de cinza.
2. Erosão (kernel 5×5, 1 iteração) sobre o threshold.
3. `bitwise_or` com a imagem original (mantém o sinal dentro das regiões "válidas").
4. Aplica a mesma máscara do polígono (com erosão de 10 iterações desta vez), removendo o exterior.
5. Salva o resultado em `./processadas/`.

Essa variante isola **somente o cultivo dentro do talhão**, ignorando solo exposto ou
áreas com baixo NDVI. No fluxo de treinamento atual da V7, o que de fato vai para o modelo
é o resultado de `aplica_mascara()`.

### 1.6 Persistência — `mover_processadas.py`

Após percorrer todos os DAPs, [pipeline.py:173](../../pipeline.py#L173) chama
`inserir_registro()`, que grava no SQLite (`dados.db`, tabela `culturas`) por talhão:

| Coluna | Conteúdo |
|---|---|
| `cultura`, `ref_infra_v`, `ref_rgb`, `data`, `mes`, `path` | Metadados do KML |
| `area` | Área do polígono em m² (calculada por `calcular_area2()`) |
| `imagens_baixadas` | Lista de caminhos para as `response.png` brutas |
| `imagens_processadas` | Lista de caminhos para os PNGs em `./processadas/` |

[mover_processadas.py](../../dados/mover_processadas.py) oferece uma rotina alternativa
(`data_arquivos`) que reconstrói essas listas varrendo o disco — útil quando o banco
ficou dessincronizado, mas não está no caminho crítico do treinamento.

### 1.7 O que chega ao modelo

Para cada talhão presente no banco, `carregar_dados()` em
[train.py:118-155](train.py#L118-L155) lê a coluna `imagens_processadas`, descarta os
caminhos que não existem mais no disco, ordena pelas datas extraídas dos sufixos `_d<dap>`
e produz uma sequência:

```
[
  ("./processadas/mascara_a3f1b2c4_v_d21.png", 21),
  ("./processadas/mascara_a3f1b2c4_v_d31.png", 31),
  ("./processadas/mascara_a3f1b2c4_v_d56.png", 56),
]
```

Cada registro vira um sample do `TemporalCulturaDataset` (até `MAX_SEQ_LEN = 3` timesteps),
e cada PNG passa por `preprocessar_imagem()` ([train.py:158-167](train.py#L158-L167)):

1. `cv2.imread` → BGR uint8.
2. `cv2.cvtColor(BGR→RGB)`.
3. `cv2.resize` para `224×224`.
4. Normalização por canal: `(img/255 - mean) / std`, com `mean = [0.485, 0.456, 0.406]`
   e `std = [0.229, 0.224, 0.225]` (estatísticas ImageNet, exigidas pelo backbone EfficientNetB0).
5. Transposição `HWC → CHW` para o formato esperado pelo PyTorch.

Se a imagem não puder ser lida, retorna um tensor zero `(3, 224, 224)`.

---

## 2. Construção do Dataset e Split

A partir do banco SQLite alimentado na etapa 1, o `train.py` constrói o dataset que vai
de fato alimentar a rede. Esta seção descreve como cada talhão vira um sample, como as
sequências de imagens são empacotadas e como o split treino/validação é montado.

### 2.1 Carregamento bruto — `carregar_dados()`

Em [train.py:118-155](train.py#L118-L155), o código abre `dados_v2.db` e lê
`SELECT cultura, mes, imagens_processadas FROM culturas`. Para cada linha:

1. **Filtro de classe**: só passam talhões cuja `cultura` está em
   `CLASSES = ['soja', 'milho', 'trigo', 'aveia', 'feijão']`. O alias `feijao` (sem
   acento) é mapeado para `feijão` em [train.py:124](train.py#L124).
2. **Parsing da lista de imagens**: a coluna `imagens_processadas` é uma string Python
   (`"['./processadas/mascara_xxx_d21.png', ...]"`). `ast.literal_eval` converte em lista;
   se falhar (`ValueError`/`SyntaxError`), o talhão é descartado.
3. **Verificação de existência**: só caminhos que ainda existem em disco entram. Cada
   caminho relativo é resolvido contra `SRC_DIR` (`src/`), permitindo rodar o treino de
   qualquer working directory.
4. **Extração do DAP**: `extrair_dia()` ([train.py:113-115](train.py#L113-L115)) faz
   regex `_d(\d+)\.png$` no nome do arquivo, recuperando o número de dias após plantio.
5. **Ordenação temporal**: a lista `[(caminho, dia), ...]` é ordenada por `dia`, de modo
   que o índice 0 sempre é a imagem mais cedo na temporada.

A função retorna três listas paralelas:

| Lista | Tipo | Conteúdo |
|---|---|---|
| `registros` | `list[list[tuple[str, int]]]` | Sequência ordenada `(caminho, DAP)` por talhão |
| `labels` | `list[int]` | Índice da classe em `CLASSES` (0 = soja, 1 = milho, …) |
| `meses` | `list[int]` | Mês de plantio (1–12), default `1` se ausente |

Talhões sem nenhuma imagem válida são silenciosamente excluídos. O log mostra a contagem
final de talhões, o histograma de comprimentos de sequência e a distribuição por mês —
útil para detectar viés sazonal nos dados.

### 2.2 Split treino/validação

Feito em [train.py:470-472](train.py#L470-L472) com `sklearn.model_selection.train_test_split`:

```python
reg_treino, reg_val, lab_treino, lab_val, mes_treino, mes_val = train_test_split(
    registros, labels, meses,
    test_size=0.2,        # 80/20
    stratify=labels,      # mantém a proporção das 5 classes em ambas as partes
    random_state=SEED,    # SEED = 42 -> split reprodutível
)
```

Pontos importantes:

- **Granularidade do split é o talhão**, não a imagem. Como cada talhão pode ter até 3
  imagens, manter o split por talhão evita vazamento temporal (treino e validação
  vendo a mesma área em DAPs diferentes).
- **Stratify por classe** garante que classes raras (ex.: aveia) tenham presença na
  validação proporcional ao treino.
- O split **não** é estratificado por mês — se o objetivo for medir generalização
  para meses não vistos, isso teria que ser feito manualmente.

### 2.3 `TemporalCulturaDataset` — empacotando uma sequência

Definido em [train.py:172-225](train.py#L172-L225). Cada `__getitem__(idx)` produz uma
tupla com 5 tensores de tamanho fixo, mesmo quando o talhão tem menos de
`MAX_SEQ_LEN = 3` imagens.

**Entradas do modelo** (passadas para `modelo.forward(images, dias, mes, mask)`):

| Tensor | Shape | Dtype | Significado |
|---|---|---|---|
| `images` | `(3, 3, 224, 224)` | `float32` | Sequência de até 3 imagens RGB normalizadas |
| `dias` | `(3,)` | `float32` | DAP normalizado (`dia / MAX_DIA`, com `MAX_DIA = 100.0`) |
| `mes` | `()` | `long` | Mês 0-indexado (`mes - 1`) — alimenta o `nn.Embedding(12, 8)` |
| `mask` | `(3,)` | `float32` | `1.0` para timesteps válidos, `0.0` para padding |

**Alvo / ground truth** (usado apenas pela loss, **nunca** entra no modelo):

| Tensor | Shape | Dtype | Significado |
|---|---|---|---|
| `label` | `()` | `long` | Índice da classe verdadeira em `CLASSES`, comparado contra os logits via `CrossEntropyLoss` |

A tupla agrupa entradas e alvo no mesmo `__getitem__` por convenção do PyTorch — o
DataLoader produz batches com tudo junto, e o loop em
[train.py:352-363](train.py#L352-L363) faz a separação:

```python
for images, dias, mes, mask, labels in loader_treino:
    logits = modelo(images, dias, mes, mask)   # modelo recebe só as 4 entradas
    loss = criterion(logits, labels).mean()    # labels usado só aqui, na loss
```

**Padding**: se a sequência tem 1 ou 2 imagens, as posições restantes ficam zeradas
em `images` e `dias`, e a `mask` marca essas posições com `0.0`. Mais à frente, no
modelo, essa máscara vira `key_padding_mask` da `MultiheadAttention` e também é usada
no mean pooling para que tokens de padding **não contribuam** para o vetor final.

### 2.4 Data augmentation geométrica

Aplicada apenas no split de treino (`augment=True`). Para preservar o alinhamento
temporal — uma rotação só faz sentido se for igual em todos os timesteps do mesmo
talhão —, os parâmetros são sorteados **uma vez por sequência** em
[train.py:198-204](train.py#L198-L204):

| Parâmetro | Distribuição | Operação |
|---|---|---|
| `flip_h` | Bernoulli(0.5) | `img[:, :, ::-1]` (espelho horizontal) |
| `flip_v` | Bernoulli(0.5) | `img[:, ::-1, :]` (espelho vertical) |
| `rot_k` | Uniforme em `{0,1,2,3}` | `np.rot90(img, k, axes=(1,2))` (0°, 90°, 180°, 270°) |

A combinação cobre as 8 simetrias do quadrado (D4), totalmente apropriada para imagens
aéreas onde não há um "para cima" canônico. **Não há** jitter de cor, blur, nem
recorte — o sinal espectral original é preservado, o que importa para um problema de
classificação de cultura baseado em assinatura espectral.

### 2.5 Balanceamento de classes — `WeightedRandomSampler`

Em [train.py:480-487](train.py#L480-L487):

```python
class_counts = np.bincount(lab_tr_np, minlength=len(CLASSES))
sample_w = (1.0 / np.maximum(class_counts, 1))[lab_tr_np]
sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_w, dtype=torch.double),
    num_samples=len(lab_tr_np),
    replacement=True,
)
```

Cada talhão recebe peso `1 / contagem_da_sua_classe`, fazendo com que **cada batch tenda
a ser aproximadamente uniforme entre as 5 classes** — mesmo quando o dataset bruto é
desbalanceado (no caso atual, soja e milho dominam). Como `replacement=True`, talhões
de classes raras são reamostrados várias vezes por época; combinado com a augmentation
estocástica, o modelo vê variações geométricas diferentes a cada visita.

`num_samples=len(lab_tr_np)` mantém o tamanho da época próximo ao do dataset original,
então o tempo por época não muda.

### 2.6 DataLoaders

Em [train.py:498-505](train.py#L498-L505):

| DataLoader | Shuffle/Sampler | Augment | `pin_memory` | `num_workers` |
|---|---|---|---|---|
| `loader_tr` | `WeightedRandomSampler` | Sim | True (em CUDA) | 4 |
| `loader_val` | `shuffle=False` | Não | True (em CUDA) | 4 |

`persistent_workers=True` evita o custo de respawnar processos a cada época.
`worker_init_fn=_worker_init` ([train.py:491-496](train.py#L491-L496)) chama
`cv2.setNumThreads(0)` em cada worker para evitar **segfault** decorrente do threadpool
interno do OpenCV não ser fork-safe — o mesmo motivo da chamada no topo do módulo
em [train.py:41](train.py#L41).

`BATCH_SIZE = 64` ([train.py:82](train.py#L82)) é dimensionado para a memória da GPU
(cada batch carrega `64 × 3 × 3 × 224 × 224 × 4 bytes ≈ 115 MB` apenas em pixels, sem
contar ativações da EfficientNet).

---

## 3. Arquitetura do Modelo

A classe `EfficientNetTemporalV6` ([train.py:230-334](train.py#L230-L334)) — o nome
"V6" persiste por razões históricas, mas é a rede V7 — é composta por quatro blocos:

```
images (B, 3, 3, 224, 224)         dias (B, 3)        mes (B,)        mask (B, 3)
        │                              │                 │                  │
        ▼                              │                 ▼                  │
┌─────────────────────┐                │         ┌────────────────┐        │
│  EfficientNetB0     │                │         │ Embedding(12,8)│        │
│  (timm, pretrained) │                │         └────────┬───────┘        │
└──────────┬──────────┘                │                  │                │
           │ features (B, 3, 1280)     │                  │                │
           │                           └──────┐  ┌────────┘                │
           │                                  ▼  ▼                         │
           │                           concat (dia, mes_emb) → (B, 3, 9)  │
           │                                     │                         │
           │                            ┌────────▼────────┐                │
           │                            │ Linear(9 → 64)  │                │
           │                            │ ReLU            │                │
           │                            │  → γ, β (1280)  │  (zero-init)   │
           │                            └────────┬────────┘                │
           │                                     │                         │
           ▼                                     ▼                         │
     features × (1 + γ) + β   ──────►  tokens (B, 3, 1280)                │
                                              │                            │
                                              ▼                            │
                                  MultiHeadAttention #1  ◄─ key_padding ◄─┤
                                       + residual + LayerNorm              │
                                              │                            │
                                              ▼                            │
                                  MultiHeadAttention #2  ◄─ key_padding ◄─┘
                                       + residual + LayerNorm
                                              │
                                              ▼
                              masked mean pooling (sobre os T tokens)
                                              │
                                              ▼
                              Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→5)
                                              │
                                              ▼
                                       logits (B, 5)
```

### 3.1 Backbone — EfficientNetB0

[train.py:245-250](train.py#L245-L250):

```python
self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
self.feature_dim = self.backbone.num_features  # 1280
for param in self.backbone.parameters():
    param.requires_grad = False
```

- `num_classes=0` remove a cabeça de classificação original (1000 classes ImageNet),
  expondo a saída do *global average pooling*: vetor `(N, 1280)` por imagem.
- `pretrained=True` baixa pesos ImageNet — daí o uso da normalização
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` na entrada.
- O backbone é **congelado na construção**. A Fase 1 do treino só ajusta o que vem depois;
  a Fase 2 descongela as últimas 20 camadas (ver seção 4).

A EfficientNetB0 é aplicada em paralelo a todas as `B × T` imagens via reshape em
[train.py:293-295](train.py#L293-L295), evitando um loop temporal explícito:

```python
imgs_flat = images.reshape(B * T, *images.shape[2:])  # (B*T, 3, 224, 224)
feats_flat = self.backbone(imgs_flat)                  # (B*T, 1280)
features = feats_flat.reshape(B, T, -1)                # (B, T, 1280)
```

### 3.2 FiLM — condicionamento por dia e mês

**FiLM** (Feature-wise Linear Modulation, Perez et al. 2018) modula um vetor de
features `f` por `f' = (1 + γ) ⊙ f + β`, onde `γ` e `β` são gerados por uma rede
condicional. A V7 usa FiLM para injetar a **temporalidade** (DAP + mês de plantio)
diretamente nos canais espaciais da EfficientNet.

**Rede condicional** ([train.py:253-263](train.py#L253-L263)):

```python
self.mes_embedding = nn.Embedding(12, MES_EMBED_DIM)        # MES_EMBED_DIM = 8
self.film_hidden   = nn.Linear(1 + MES_EMBED_DIM, 64)        # 9 → 64
self.film_gamma    = nn.Linear(64, self.feature_dim)         # 64 → 1280
self.film_beta     = nn.Linear(64, self.feature_dim)         # 64 → 1280
nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
nn.init.zeros_(self.film_beta.weight);  nn.init.zeros_(self.film_beta.bias)
```

Por que `Embedding(12, 8)` para o mês? Tratar mês como **categórico** evita o problema
de descontinuidade de janeiro→dezembro que um valor escalar não captura, sem precisar
de codificações trigonométricas. 8 dimensões dão folga para a rede aprender estruturas
sazonais (ex.: cluster de meses chuvosos).

A **inicialização zero** de `γ` e `β` é deliberada: no primeiro forward,
`f' = (1 + 0) ⊙ f + 0 = f`. A rede começa como **identidade** sobre as features da
EfficientNet, então o backbone pretreinado não é perturbado no início — o sinal
condicional só entra na medida em que o gradiente justifica.

**Forward FiLM** ([train.py:298-307](train.py#L298-L307)):

```python
mes_emb = self.mes_embedding(mes)               # (B, 8) — mesmo p/ todos os T
mes_emb = mes_emb.unsqueeze(1).expand(-1, T, -1) # (B, T, 8)
dia_exp = dias.unsqueeze(-1)                     # (B, T, 1)
context = torch.cat([dia_exp, mes_emb], dim=-1)  # (B, T, 9)

film_h = F.relu(self.film_hidden(context))       # (B, T, 64)
gamma  = self.film_gamma(film_h)                 # (B, T, 1280)
beta   = self.film_beta(film_h)                  # (B, T, 1280)

tokens = features * (1.0 + gamma) + beta         # (B, T, 1280)
```

O mês é constante dentro da sequência (a data de plantio não muda), mas o DAP varia,
então `γ` e `β` são **diferentes para cada timestep**. Na prática, isso permite ao
modelo aprender coisas como "no DAP 21 a banda NIR pesa mais para distinguir soja de
trigo" sem precisar codificar essa regra explicitamente.

### 3.3 Atenção temporal — 2 camadas de MultiHeadAttention

Após o FiLM, os tokens `(B, T, 1280)` são processados por duas camadas de
self-attention ([train.py:266-273](train.py#L266-L273)):

```python
self.attn1 = nn.MultiheadAttention(embed_dim=1280, num_heads=8, dropout=0.1, batch_first=True)
self.norm1 = nn.LayerNorm(1280)
self.attn2 = nn.MultiheadAttention(embed_dim=1280, num_heads=8, dropout=0.1, batch_first=True)
self.norm2 = nn.LayerNorm(1280)
```

Forward ([train.py:309-317](train.py#L309-L317)):

```python
key_pad_mask = (mask == 0)   # (B, T) — True onde é padding

attn_out1, _ = self.attn1(tokens, tokens, tokens, key_padding_mask=key_pad_mask)
tokens = self.norm1(tokens + attn_out1)         # residual + LN

attn_out2, _ = self.attn2(tokens, tokens, tokens, key_padding_mask=key_pad_mask)
tokens = self.norm2(tokens + attn_out2)
```

Pontos-chave:

- **Self-attention puro**: `Q = K = V = tokens`. Cada timestep "olha" para todos os
  outros e produz uma representação enriquecida com contexto temporal.
- **`key_padding_mask`** inverte a semântica de `mask` (em PyTorch, `True` = ignorar):
  posições de padding deixam de contribuir para o `softmax(QK^T)`, então tokens reais
  nunca são "contaminados" por zeros.
- **Conexão residual + LayerNorm** segue o padrão Transformer (post-norm): preserva o
  sinal original do FiLM mesmo quando a atenção contribui pouco no início.
- 8 cabeças sobre 1280 dim → 160 dim por cabeça. Com `T = 3`, cada cabeça aprende um
  "padrão de mistura" diferente sobre os 3 timesteps (ex.: dar peso ao último DAP, ou
  à diferença entre o primeiro e o último).

Não há **positional encoding**: a posição temporal já está injetada via FiLM (pelo DAP
normalizado). Se as imagens viessem com DAPs diferentes em ordens diferentes, a atenção
ainda funcionaria pois é permutação-invariante e o DAP carrega a posição.

### 3.4 Masked mean pooling

Antes da cabeça de classificação, os T tokens são colapsados em um único vetor
([train.py:319-321](train.py#L319-L321)):

```python
mask_exp = mask.unsqueeze(-1)                                # (B, T, 1)
pooled = (tokens * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
```

A máscara zera os tokens de padding **antes** da soma, e o divisor é o número de
timesteps válidos (não `T`). O `clamp(min=1.0)` é uma proteção contra div-por-zero
caso uma sequência venha totalmente vazia — não deve ocorrer pois `carregar_dados()`
filtra esse caso, mas o guardrail está lá.

Mean pooling foi escolhido em vez de attention pooling ou `[CLS]` token: com `T ≤ 3`,
a média já captura bem o que importa, e mantém o número de parâmetros baixo.

### 3.5 Cabeça de classificação

[train.py:276-281](train.py#L276-L281):

```python
self.head = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 5),    # 5 = len(CLASSES)
)
```

Uma cabeça compacta — uma camada oculta de 256 neurônios com dropout de 30%. A saída
são **logits** (sem softmax), porque a `nn.CrossEntropyLoss` aplica `log_softmax`
internamente; ver seção 4.

### 3.6 Resumo dos hiperparâmetros estruturais

| Constante | Valor | Onde usado |
|---|---|---|
| `MAX_SEQ_LEN` | 3 | Tamanho fixo da sequência temporal |
| `MAX_DIA` | 100.0 | Divisor de normalização do DAP |
| `MES_EMBED_DIM` | 8 | Dimensão do embedding categórico de mês |
| `feature_dim` | 1280 | Saída da EfficientNetB0 (não configurável) |
| `num_heads` | 8 | Cabeças de atenção em ambas as camadas |
| `dropout (attn)` | 0.1 | Dropout interno da MultiHeadAttention |
| `dropout (head)` | 0.3 | Dropout da camada oculta da cabeça |
| `FINE_TUNE_LAYERS` | 20 | Últimos parâmetros do backbone descongelados na Fase 2 |
| `num_classes` | 5 | `['soja', 'milho', 'trigo', 'aveia', 'feijão']` |

### 3.7 Contagem de parâmetros (ordem de grandeza)

- Backbone EfficientNetB0: ~4,0 M parâmetros (congelados na Fase 1).
- Mes embedding: `12 × 8 = 96`.
- FiLM (`9→64` + `64→1280` × 2): `9·64 + 64 + 2 · (64·1280 + 1280) ≈ 166 k`.
- 2 × MultiHeadAttention(1280, 8 heads): cada uma com `4 · 1280² ≈ 6,55 M`, total ~13 M.
- 2 × LayerNorm(1280): `4 · 1280 ≈ 5 k`.
- Cabeça: `1280·256 + 256 + 256·5 + 5 ≈ 329 k`.

**Total ≈ 17,5 M parâmetros**, dos quais ~13,5 M são treináveis na Fase 1 (o backbone
de 4 M está congelado). O log inicial em [train.py:512-514](train.py#L512-L514)
imprime os números exatos a cada execução.

---

## Próximas seções (a documentar)

- 4. Estratégia de treinamento em 2 fases (cabeça congelada → fine-tuning)
- 5. Avaliação e métricas
