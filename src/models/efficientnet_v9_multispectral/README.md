# EfficientNet V9 — Multispectral TIFF Pipeline

Plano para o próximo treino. Mantém a arquitetura central do V7 (EfficientNetB0 +
FiLM(dia, mês) + Temporal Attention) e a cabeça hierárquica do V8, mas troca a
**entrada do modelo** de PNG 3 canais para **TIFF 6 bandas FLOAT32**.

Esta pasta é um **plano implementado**: cada arquivo aqui roda standalone assim
que (a) os pesos do dashboard estiverem migrados e (b) o primeiro lote de TIFFs
estiver baixado para calcular as estatísticas de normalização.

---

## Por que mudar para TIFF multispectral

Três motivações concretas:

1. **Mais informação espectral.** O `BANDAS_RBN` do V7 entrega 3 bandas (B02/B04/B08
   compactadas em PNG 8-bit). Aveia × trigo, que são o ponto fraco do V7/V8, são
   gramíneas visualmente parecidas no visível e no NIR — a separação melhora
   notavelmente quando você adiciona **SWIR** (B11/B12, sensíveis à umidade) e
   eventualmente uma banda de **red edge** (B05/B06).
2. **Sem perda de quantização.** PNG 8-bit comprime reflectância contínua em 256
   níveis. TIFF FLOAT32 preserva os valores nativos (`[0, 1]` em S2 L2A), o que
   importa especialmente nas extremidades do espectro (SWIR varia em uma faixa
   estreita).
3. **Migração da API.** O `WcsRequest` (atual) é legado. O `SentinelHubRequest`
   (Processing API) com `evalscript` é o caminho recomendado e dá:
   - filtro `maxcc` no servidor;
   - `mosaicking_order='leastCC'` que entrega a melhor imagem da janela
     direto, sem precisar baixar várias e escolher;
   - controle total das bandas via evalscript versionado no repo (não depende
     mais de uma layer configurada à mão no dashboard).

## O que muda exatamente vs V7/V8

| Componente | V7/V8 | V9 |
|---|---|---|
| Bandas | 3 (B02, B04, B08) | 6 (B02, B03, B04, B08, B11, B12) |
| Formato | PNG uint8 | TIFF FLOAT32 |
| API Sentinel | `WcsRequest` (layer) | `SentinelHubRequest` (evalscript) |
| Seleção da imagem | client-side por bbox | `leastCC` server-side + client-side fino |
| Normalização | ImageNet `mean/std` | Stats por banda do dataset (`compute_band_stats.py`) |
| Backbone | EfficientNetB0 com `conv_stem` 3-canais | EfficientNetB0 com `conv_stem` adaptado para 6 canais |
| Cabeça | V7: 5-way / V8: hierárquica | V8: hierárquica (mantida) |
| `MAX_SEQ_LEN` | 3 | 5 (aproveitando os DAPs adicionais do `pipeline_daps_fixos.py`) |

Tudo o que **não** está nessa tabela permanece igual: FiLM(dia, mês), Temporal
Attention de 2 camadas, augmentation D4, `WeightedRandomSampler`, treino em duas
fases (cabeça → fine-tuning das últimas N camadas).

## Estrutura da pasta

```
efficientnet_v9_multispectral/
├── README.md                     ← este arquivo
├── evalscript_s2_6bands.js       ← evalscript que define as 6 bandas
├── sentinel_tiff_request.py      ← wrapper baixo-nível: 1 chamada → 1 TIFF
├── tiff_io.py                    ← read/write/mask de TIFF multi-band
├── pipeline_tiff.py              ← orquestração: CSV → DAPs fixos → DB
├── compute_band_stats.py         ← util: calcula mean/std por banda do dataset
├── dataset.py                    ← Dataset PyTorch para TIFF 6 bandas
└── train.py                      ← treino V9 (cabeça hierárquica V8 + entrada V9)
```

## Ordem de migração

Não tente trocar tudo de uma vez. Sequência recomendada:

### Passo 1 — descobrir a definição atual de BANDAS_RBN (1 hora)

A camada `BANDAS_RBN` no dashboard do Sentinel Hub define quais bandas e qual
escala/normalização o V7/V8 estão usando hoje. Antes de criar o evalscript do
V9, abra o dashboard e exporte a definição da layer (Configuration Utility →
Layers → BANDAS_RBN → "Show source"). Isso garante que sua decisão de bandas no
V9 seja informada, não chutada.

### Passo 2 — gerar dataset TIFF em paralelo ao PNG (vários dias de download)

Rode `pipeline_tiff.py` para baixar TIFFs em uma pasta separada
(`./imagens_tiff_v9/`), **sem** apagar os PNGs do V7/V8. Use os mesmos DAPs
fixos definidos em [pipeline_daps_fixos.py](../../pipeline_daps_fixos.py).
Estimativa: ~9 DAPs × ~20k talhões × ~600 KB por TIFF de 6 bandas = ~110 GB.

### Passo 3 — calcular estatísticas de normalização (5 minutos)

Execute `compute_band_stats.py` apontando para `./processadas_tiff_v9/`. Vai
imprimir os arrays `BAND_MEAN` e `BAND_STD` que devem ser colados em
`dataset.py` substituindo os placeholders.

### Passo 4 — treinar V9 (4–8 horas em GPU)

`python src/models/efficientnet_v9_multispectral/train.py`.

### Passo 5 — comparar V8 vs V9 com `avaliar_focusnet_v3_adversarial.py`

Roda o teste adversarial nos dois modelos e compara taxa de defesa por cultura.
A hipótese é que aveia/trigo melhorem mais que soja/milho/feijão (porque o
sinal do SWIR é mais informativo para gramíneas em estágios diferentes).

## Decisões deferidas

Coisas que escolhi defaults para mas que merecem revisão antes de treinar para
valer:

- **Quais 6 bandas exatamente.** Comecei com `[B02, B03, B04, B08, B11, B12]` —
  RGB completo + NIR + 2 SWIR. Alternativas a considerar:
  - **Adicionar red edge** (B05 ou B06) e remover B03 (verde) → 6 bandas mais
    informativas para vegetação, perde fidelidade visual.
  - **8 bandas** (B02, B03, B04, B05, B06, B08, B11, B12) — mais sinal, ~33%
    mais memória/disco e o `conv_stem` adaptado fica mais distante do pretraining.
- **`MAX_SEQ_LEN = 5` ou maior.** O `pipeline_daps_fixos.py` baixa 9 DAPs por
  talhão. V7/V8 usam 3. Mais timesteps ajudam a atenção temporal mas aumentam
  custo quadrático. 5 é meio-termo razoável; 9 vale testar se a GPU aguentar.
- **Estratégia de inflar o `conv_stem`.** Coloquei "copia pesos RGB e usa média
  de RGB para os outros 3 canais". Alternativas: zero-init nas extras, ou
  inicialização com features pretrained de algum modelo BigEarthNet (S2
  multispectral pretrained). Última opção é a melhor mas requer download de
  pesos externos.
- **Resampling de B11/B12.** S2 entrega B11/B12 a 20 m nativo; o evalscript
  faz o Sentinel Hub reamostrar para 10 m no servidor. É o padrão e funciona,
  mas perde nada de qualidade — se a quota for um problema, vale considerar
  request a 20 m e fazer upsample local.

## Compatibilidade

- **DB**: o V9 pode usar o **mesmo** `dados_v2.db` que o pipeline antigo. Vai
  apenas escrever em colunas novas (ou em uma tabela `culturas_tiff` paralela
  — decidi por tabela paralela em `pipeline_tiff.py` para não misturar
  formatos no mesmo schema).
- **`processamento_imagens.py`**: a função `cria_mascara` é reutilizada. A
  função `aplica_mascara` é **substituída** por `tiff_io.aplicar_mascara_tiff`
  porque a antiga assume entrada/saída uint8.
- **V7/V8**: continuam funcionando com seus PNGs. Nada nesta pasta toca neles.

## Referências

- [Plano V7 (DOCUMENTACAO.md)](../efficientnet_v7/DOCUMENTACAO.md) — arquitetura base.
- [V8 train.py](../efficientnet_v8/train.py) — cabeça hierárquica reaproveitada.
- [pipeline_daps_fixos.py](../../pipeline_daps_fixos.py) — origem do conjunto fixo de DAPs.
- Sentinel Hub Processing API: https://docs.sentinel-hub.com/api/latest/api/process/
- Evalscript V3: https://docs.sentinel-hub.com/api/latest/evalscript/v3/
