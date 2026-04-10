# Arquitetura dos Modelos de Classificação de Culturas

## Problema

Classificar culturas agrícolas (milho, soja, trigo) a partir de imagens de satélite Sentinel-2 processadas e mascaradas ao contorno do talhão.

Cada talhão possui até 3 imagens capturadas em diferentes estágios de crescimento após o plantio:

| Cultura | Datas de observação (dias após plantio) |
|---------|----------------------------------------|
| Milho   | d21, d31, d56                          |
| Soja    | d21, d31, d56                          |
| Trigo   | d26, d32, d47                          |

---

## Evolução dos Modelos

### v1 — EfficientNetB0 + ViT-Base (TensorFlow)

**Abordagem**: Cada imagem classificada independentemente. Sem informação temporal.

```
imagem.png → EfficientNetB0/ViT-Base → softmax → cultura
```

**Problemas encontrados**:

1. **Data leakage**: O split treino/validação era feito por imagem, não por talhão. Imagens do mesmo campo (d21, d31, d56) apareciam nos dois conjuntos. O modelo memorizava a forma do talhão em vez da cultura. Resultado: 95% em validação, 41% em teste real.

2. **Modelo grande demais**: ViT-Base tem 86M parâmetros. Com ~3000 talhões de treino, o modelo sobreajusta facilmente.

3. **Dependência de HuggingFace TF**: A classe `TFViTModel` está sendo descontinuada no Transformers v5. Bugs com safetensors e incompatibilidades crescentes.

**Correção aplicada**: Split por registro (todas as imagens de um talhão vão para o mesmo lado). A accuracy de validação caiu para ~63% — o número honesto.

---

### v2 — ViT-Small + Dia como Feature (PyTorch/timm)

**Abordagem**: Cada imagem ainda é classificada individualmente, mas o dia após plantio é fornecido ao modelo como feature adicional.

```
imagem.png ──→ vit_small (384-dim) ──┐
                                      ├─ concat (416) → Dense(256) → softmax → cultura
dia (normalizado) → Linear(32-dim) ──┘
```

**Melhorias sobre v1**:

- **vit_small** (22M params) em vez de vit_base (86M) — melhor para datasets pequenos
- **PyTorch + timm** em vez de TensorFlow + HuggingFace — ecossistema mais estável
- **Informação temporal**: o modelo sabe que está vendo dia 21 vs dia 56, podendo aprender que certas aparências são normais para um estágio mas anômalas para outro
- **Ablação multi-temporal**: flag `--dias` permite treinar com subconjuntos de datas para identificar quais estágios são mais discriminativos

**Limitação**: Ainda faz 3 predições independentes por talhão. Não aprende a evolução temporal.

---

### v3 — ViT-Small + Temporal Transformer (PyTorch/timm)

**Abordagem**: Classifica o talhão inteiro combinando todas as imagens temporais em uma única predição. O modelo aprende como a cultura evolui ao longo do tempo.

```
img_d21 → vit_small → 384-dim ─┐
img_d31 → vit_small → 384-dim ─┼→ Temporal Transformer (2 camadas) → mean pool → Dense(256) → softmax
img_d56 → vit_small → 384-dim ─┘
              +
         dia_embedding (384-dim)
```

**Arquitetura detalhada**:

1. **Backbone ViT compartilhado**: Cada imagem temporal passa pelo mesmo `vit_small_patch16_224` (pesos compartilhados). Produz um vetor de 384 dimensões por imagem.

2. **Embedding temporal**: O dia normalizado (`dia / 100`) passa por `Linear(1→64) → ReLU → Linear(64→384)`. O resultado é somado ao feature vector da imagem, codificando o estágio de crescimento.

3. **Temporal Transformer Encoder**: 2 camadas de self-attention (`d_model=384, nhead=6, ff=512`). Cada token temporal pode atender aos outros, aprendendo relações como "se em d21 era verde-claro e em d56 ficou alto e denso, provavelmente é milho."

4. **Attention mask**: Talhões com menos de 3 imagens usam padding com zeros. O `src_key_padding_mask` garante que o transformer ignore posições vazias.

5. **Mean pooling**: Os tokens válidos (não-padding) são agregados por média ponderada.

6. **Cabeça de classificação**: `Dense(384→256) → ReLU → Dropout(0.3) → Dense(256→3)`.

**Por que funciona melhor (em teoria)**:

- **Milho**: cresce rápido verticalmente. Mudança dramática entre d21 (plântula) e d56 (canopy alta e espaçada).
- **Soja**: se espalha horizontalmente. Cobertura densa e uniforme em d56.
- **Trigo**: ciclo mais curto. Pico vegetativo em d32, início de senescência em d47.

Um snapshot único pode ser ambíguo (milho e soja jovens são parecidos). A trajetória temporal desambigua.

---

## Pipeline de Dados

```
KMLs (polígonos dos talhões)
  └─→ Sentinel Hub API (download das imagens por data)
        └─→ Máscara KML aplicada (recorta ao contorno do talhão)
              └─→ ./processadas/mascara_{uuid}_v_d{dia}.png
                    └─→ SQLite (cultura, mês, lista de caminhos)
                          └─→ gerar_sample_treino.py (amostra balanceada)
                                └─→ sample_treino_v2.db (3000/classe, 3 imgs cada)
```

### Pré-processamento no treinamento

| Etapa | v1 (TF) | v2/v3 (PyTorch) |
|-------|---------|-----------------|
| Resize | 224x224 | 224x224 |
| Color | BGR→RGB | BGR→RGB |
| Normalização | `pixel / 127.5 - 1.0` (range [-1,1]) | ImageNet: `(pixel/255 - mean) / std` |
| Channel order | HWC → NHWC (TF default) | HWC → CHW (PyTorch default) |

---

## Treinamento em 2 Fases (Transfer Learning)

Todas as versões usam a mesma estratégia:

### Fase 1 — Base congelada
- Backbone ViT/EfficientNet completamente congelado
- Apenas a cabeça de classificação (+ temporal transformer no v3) é treinada
- Learning rate alto: `1e-3`
- Early stopping: patience 3
- Objetivo: ajustar a cabeça sem destruir features pré-treinadas

### Fase 2 — Fine-tuning
- Últimos 2 blocos do backbone são descongelados
- Learning rate muito baixo: `1e-5`
- Early stopping: patience 4
- Objetivo: adaptar features visuais de alto nível ao domínio de satélite

---

## Comparação dos Modelos

| | v1 EfficientNet | v1 ViT-Base | v2 ViT-Small | v3 ViT-Small+Temporal |
|---|---|---|---|---|
| Framework | TensorFlow | TensorFlow | PyTorch/timm | PyTorch/timm |
| Backbone params | 4M | 86M | 22M | 22M |
| Input | 1 imagem | 1 imagem | 1 imagem + dia | 3 imagens + dias |
| Predição | por imagem | por imagem | por imagem | por talhão |
| Info temporal | nenhuma | nenhuma | dia como feature | sequência temporal |
| Unidade de avaliação | imagem | imagem | imagem | talhão |

---

## Métricas

- **Accuracy**: proporção de predições corretas
- **F1-score macro**: média dos F1 por classe (não pondera pelo tamanho da classe)
- **F1-score por classe**: identifica se alguma cultura é sistematicamente confundida
- **Confusion matrix**: mostra os padrões de erro (milho↔soja, trigo↔soja, etc.)
- **Tempo de inferência**: ms por amostra (imagem para v1/v2, talhão para v3)

---

## Prevenção de Data Leakage

O principal risco neste dataset é que imagens do mesmo talhão em datas diferentes são **visualmente quase idênticas** (mesma forma de campo, mesma região).

**Regra**: o split treino/validação/teste é sempre feito **por registro** (talhão), nunca por imagem individual. Todas as imagens de um campo vão para o mesmo conjunto.

```python
# CORRETO: split por registro
train_test_split(registros, labels, ...)

# ERRADO: split por imagem (data leakage)
train_test_split(caminhos_individuais, labels, ...)
```

O banco de teste (`sample_teste.db`) é gerado com `NOT IN` no SQL, garantindo que nenhum talhão do teste apareça no treino.
