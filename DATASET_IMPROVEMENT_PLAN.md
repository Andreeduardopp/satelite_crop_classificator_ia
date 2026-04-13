# Plano de Melhoria do Dataset — Crop Classifier

## Diagnóstico: por que estamos em 70%

Dois modelos arquiteturalmente diferentes (ViT + Temporal Transformer, EfficientNet + MultiHeadAttention) chegaram **exatamente** no mesmo teto de ~70% / F1 macro ~0.70. Isso não é coincidência — é evidência de que o **gargalo está no dado, não no modelo**.

O padrão é idêntico nos dois:

| Classe | F1 ViT v3 | F1 EfficientNet v2 |
|---|---|---|
| trigo | 1.00 | 1.00 |
| milho | 0.55 | 0.54 |
| soja  | 0.54 | 0.53 |
| **macro** | **0.696** | **0.687** |

O modelo é essencialmente um **classificador de trigo** com um coin flip para milho↔soja.

---

## Causa Raiz

### Problema 1 — Mesmas datas, culturas diferentes

```
milho:  d21, d31, d56
soja:   d21, d31, d56   ← idêntico ao milho
trigo:  d26, d32, d47   ← diferente → por isso F1 = 1.00
```

O modelo aprendeu a usar a janela temporal do trigo como fingerprint. Para milho e soja, essa alavanca não existe — as datas são idênticas.

### Problema 2 — Sobreposição de meses de plantio

```
Meses em que milho E soja são plantados simultaneamente: {1, 9, 10, 11, 12}
```

Em 5 dos 12 meses, os dois campos coexistem na mesma região, na mesma fase fenológica, com as mesmas datas de observação. Do ponto de vista da imagem de satélite, **são indistinguíveis nesses períodos**.

### Problema 3 — Apenas RGB em 3 datas precoces

As imagens atuais usam somente os canais **visíveis (RGB)** do Sentinel-2. O Sentinel-2 tem **13 bandas** — as mais discriminativas para vegetação estão sendo ignoradas:

| Banda | Nome | Por que importa |
|---|---|---|
| B08 (NIR) | Near Infrared | Biomassa, cobertura foliar |
| B05, B06, B07 | Red-Edge | Conteúdo de clorofila — soja tem muito mais em d56 |
| B11, B12 (SWIR) | Short-Wave IR | Umidade da planta e do solo, estrutura do canopy |

**NDVI calculado só com RGB** é uma proxy pobre. Com NIR+Red, o NDVI real discrimina muito melhor a fenologia de milho vs soja.

### Problema 4 — d21 e d31 têm baixo poder discriminativo para milho↔soja

```
d21: ambas são plântulas jovens com 2-3 folhas → visualmente idênticas
d31: milho tem 5-6 folhas, soja tem 3-4 folhas → ainda muito parecidas
d56: milho tem canopy alto e esparso, soja tem cobertura densa → diferença real
```

O modelo treina 3 timesteps, mas 2 deles (d21, d31) carregam ruído ou informação redundante para o par milho/soja. O terceiro (d56) carrega quase todo o sinal.

---

## Plano de Melhoria em 4 Frentes

---

### Frente 1 — Adicionar Bandas Espectrais (maior impacto esperado)

**O que fazer**: ao baixar imagens do Sentinel Hub, incluir não só RGB mas também:
- **NDVI** = (NIR - Red) / (NIR + Red) — índice de vegetação normalizado
- **Red-Edge (B05 ou B07)** — sensível à clorofila, alto em soja madura
- **SWIR (B11)** — umidade foliar, alto em milho no pico vegetativo

Cada imagem vira um **tensor de 6 canais** em vez de 3 (RGB + NDVI + RedEdge + SWIR).

**Por que funciona**: Literatura de remote sensing mostra que adicionando NIR+RedEdge, a confusão milho↔soja cai de ~50% de erro para ~15-20% de erro mesmo com modelos simples. É o maior ganho disponível sem coletar novos talhões.

**Mudança no pipeline**:
```
Atual:  mascara_uuid_v_d56.png              → 3 canais (RGB)
Novo:   mascara_uuid_v_d56.png              → 6 canais (RGB + NDVI + RedEdge + SWIR)
```

**Mudança no modelo**: alterar `in_channels=3` para `in_channels=6` no backbone. EfficientNet e ViT aceitam isso com um ajuste na primeira camada convolucional (inicialização dos canais extras com zeros ou cópia dos canais RGB existentes).

**Requer**: Nova rodada de download Sentinel Hub com as bandas adicionais para os talhões já existentes. Os polígonos KML não mudam.

---

### Frente 2 — Adicionar Data d70–d80 para Milho e Soja

**O que fazer**: Para milho e soja especificamente, coletar uma 4ª observação entre d70 e d80 após plantio:

```
milho d70–80: Pendoamento/espigamento — estrutura vertical inconfundível
soja  d70–80: Floração plena — cobertura horizontal densa, tonalidade amarelada
```

Essa é a janela mais discriminativa do ciclo todo. Com d56 + d70, a diferença visual é clara mesmo em imagens RGB puras.

**Mudança no pipeline**: aumentar `MAX_SEQ_LEN = 3 → 4` e incluir a data nova na sequência do dataset. Trigo não precisa de 4ª data — ele já está perfeito.

**Requer**: Nova rodada de download Sentinel Hub, apenas para milho e soja, apenas a janela d70–d80. Custo moderado.

---

### Frente 3 — Estratificação por Mês de Plantio

**O que fazer**: Analisar a performance do modelo por mês de plantio para confirmar a hipótese de que os erros milho↔soja se concentram nos meses de plantio simultâneo (set/out/nov).

**Diagnóstico**:
```python
# Para cada talhão no conjunto de validação:
# registrar: cultura_real, cultura_pred, mês_de_plantio
# agrupar erros por mês → identificar quais meses têm maior confusão
```

Se confirmado, três opções:

| Opção | Descrição | Custo |
|---|---|---|
| A | Adicionar mês de plantio como feature ao modelo (igual ao dia) | Baixo |
| B | Dois classificadores especializados: safra de verão vs inverno | Médio |
| C | Sobreamostrar meses onde milho e soja não coexistem (jan, fev, abr) | Baixo |

**Recomendação**: começar pela Opção A — já existe infraestrutura de `dia_embedding` que pode ser replicada para `mes_embedding`.

**Não requer novo download** — o campo `mes` já existe no banco de dados.

---

### Frente 4 — Limpeza de Qualidade das Imagens

**O que fazer**: Inspecionar e remover imagens com problemas:

1. **Cobertura de nuvem > 20%**: imagens nubladas treinam o modelo em ruído. O Sentinel-2 inclui banda SCL (Scene Classification Layer) que mascara nuvens.
2. **Talhões de borda pequenos** (< 1 ha): pixels misturados com o campo vizinho ensinam o modelo a reconhecer o entorno, não a cultura.
3. **Imagens completamente pretas ou saturadas**: falhas de download ou processamento.

**Diagnóstico simples** (sem novo download):
```python
# Para cada PNG no dataset:
import cv2, numpy as np
img = cv2.imread(caminho)
mean_val = img.mean()
std_val  = img.std()

if mean_val < 5:      # imagem preta / nublada escura
    descartar()
if std_val < 2:       # sem variação de pixels (uniforme)
    descartar()
if mean_val > 250:    # saturada
    descartar()
```

**Não requer novo download** — análise das imagens já existentes.

---

## Resumo de Impacto Esperado

| Frente | Dificuldade | Novo download? | Impacto esperado em F1 milho/soja |
|---|---|---|---|
| 1. Bandas espectrais (NDVI, RedEdge, SWIR) | Alta | Sim | **+15–20 pp** |
| 2. Nova data d70–d80 | Alta | Sim | **+10–15 pp** |
| 3. Estratificação por mês | Baixa | Não | **+3–8 pp** |
| 4. Limpeza de qualidade | Baixa | Não | **+2–5 pp** |

Com Frentes 1+2 implementadas, expectativa realista: **F1 milho/soja > 0.75**, F1 macro geral de **0.85+**.

---

## Ordem de Execução Recomendada

### Fase A — Imediata (sem novo download)

1. **Diagnóstico de erros por mês** (Frente 3):
   - Modificar o script de avaliação para logar `mês de plantio` por predição
   - Identificar em quais meses a confusão milho↔soja é maior
   - Decidir entre Opções A/B/C da Frente 3

2. **Limpeza de qualidade** (Frente 4):
   - Escanear todas as imagens com o critério de mean/std
   - Remover registros com imagens ruins do banco
   - Retreinar com dataset limpo → novo baseline honesto

3. **Mês como feature** (Frente 3, Opção A):
   - Adicionar `mes_embedding` ao modelo (replicar infraestrutura do `dia_embedding`)
   - Retreinar e comparar com baseline

### Fase B — Novo Download Sentinel Hub

4. **Coletar bandas NIR, RedEdge, SWIR** para talhões existentes (Frente 1):
   - Adaptar o pipeline de download para incluir B05, B08, B11
   - Calcular NDVI e índices derivados por data
   - Atualizar gerador de dataset para imagens de 6 canais

5. **Adaptar modelo para 6 canais de entrada** (Frente 1):
   - EfficientNetB0: substituir primeira camada Conv2d(3→32) por Conv2d(6→32)
   - ViT: ajustar `patch_embed` para aceitar 6 canais
   - Inicializar os 3 novos canais como cópia dos 3 originais (warm start)

6. **Coletar data d70–d80** para milho e soja (Frente 2):
   - Apenas os polígonos de milho e soja já existentes
   - Atualizar banco e dataset para `MAX_SEQ_LEN = 4`

7. **Retreinar modelo final** com 6 canais + 4 timesteps (milho/soja) e 3 timesteps (trigo)

---

## Referências

- **Sentinel-2 bands**: ESA Sentinel-2 User Guide — bandas B05 (RedEdge1), B08 (NIR), B11 (SWIR1)
- **NDVI discrimination**: Zheng et al. 2019, "Soybean and maize classification using time-series Sentinel-2 data" — ganho de ~18pp com NDVI vs RGB puro
- **FiLM conditioning**: Perez et al. 2017, "FiLM: Visual Reasoning with a General Conditioning Layer"
- **Red-Edge for crop typing**: Delegido et al. 2011 — Red-Edge é o índice mais sensível ao estágio fenológico de leguminosas (soja) vs gramíneas (milho, trigo)
