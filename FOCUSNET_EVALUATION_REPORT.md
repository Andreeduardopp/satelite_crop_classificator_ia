# Relatório de Avaliação do Serviço Focusnet — Classificação de Culturas

**Data:** 2026-04-14  
**Amostras:** 482 requisições completadas (de ~500 solicitadas)  
**Tempo Total:** 178.068 segundos (~2.968 horas / 49.5 minutos de processamento sequencial)

---

## Resumo de Desempenho

| Métrica | Valor |
|---------|-------|
| **Acurácia Global** | 21.78% (105 acertos / 482 total) |
| **F1-Score Macro** | 0.1521 |
| **Tempo Médio por Request** | 369.4s (~6.2 min) |
| **Tempo Min/Max** | 317.6s / 528.4s |

### Análise do Tempo de Processamento

- **Tempo médio de 369.4s por requisição** indica que o serviço está processando com considerável latência
- Cada requisição passa por: 1) envio do KML (~30s de timeout), 2) polling com intervalo de 10s por até 100 tentativas
- A variação (317-528s) sugere processamentos inconsistentes — alguns KMLs terminam em ~5 min, outros em até 9 min

---

## Desempenho por Cultura

| Cultura | F1-Score | Precision | Recall | TP | FP | FN | Taxa Acerto |
|---------|----------|-----------|--------|----|----|----|----|
| **AVEIA** | 0.3096 | 0.2032 | 0.6495 | 63 | 247 | 34 | **65.0%** |
| **FEIJÃO** | 0.1948 | 0.2632 | 0.1546 | 15 | 42 | 82 | **15.5%** |
| **MILHO** | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 96 | **0.0%** |
| **SOJA** | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 96 | **0.0%** |
| **TRIGO** | 0.2559 | 0.2348 | 0.2812 | 27 | 88 | 69 | **28.1%** |

### Observações Críticas

1. **MILHO e SOJA: Falha Total (0%)**
   - Nenhuma amostra de MILHO foi classificada corretamente
   - Nenhuma amostra de SOJA foi classificada corretamente
   - Ambas são frequentemente confundidas com **AVEIA** ou **FEIJÃO**
   - Isso sugere que o modelo tem dificuldade em distinguir essas culturas no espectro Sentinel

2. **AVEIA: Melhor Desempenho (65% Recall)**
   - 63 verdadeiros positivos de 97 amostras
   - Alto FP (247) — frequentemente predito quando outras culturas são submetidas
   - Pode ser um "viés de confiança" do modelo (prediz AVEIA mesmo em dúvida)

3. **TRIGO e FEIJÃO: Desempenho Baixo**
   - TRIGO: apenas 28% recall, confundido principalmente com AVEIA
   - FEIJÃO: apenas 15.5% recall, baixa confiança nas predições

---

## Padrões de Erro Observados

Analisando os primeiros 20 erros (377 erros totais):

```
True        Pred         % de Confusão
MILHO  →    FEIJÃO       (frequente)
MILHO  →    AVEIA        (frequente)
SOJA   →    AVEIA        (frequente)
SOJA   →    FEIJÃO       (ocasional)
TRIGO  →    AVEIA        (muito frequente)
FEIJÃO →    AVEIA        (ocasional)
AVEIA  →    TRIGO        (ocasional)
AVEIA  →    FEIJÃO       (ocasional)
```

**Conclusão:** O modelo tem forte tendência a classificar como **AVEIA**, sugerindo:
- Possível desbalanceamento nos dados de treino
- AVEIA pode ter assinatura espectral mais "genérica" ou robusta
- Outras culturas podem ter baixa representatividade ou assinaturas muito similares

---

## Fatores Potenciais de Degradação

### 1. **Qualidade dos Dados KML**
- Os arquivos KML vêm de períodos variados (2024-2025)
- Datas de plantio/colheita variam, podem afetar assinatura espectral
- Possível ruído ou dados incompletos em alguns polígonos

### 2. **Latência e Degradação de Serviço**
- Tempo médio de 369s por requisição é muito alto
- Pode indicar fila congestionada ou processamento degradado no servidor
- Possível que o modelo subjacente do Focusnet esteja desatualizado ou descalibrado

### 3. **Resolução Temporal**
- Sentinel-2 tem revisita a cada 5 dias (em média 10 dias com cobertura de nuvem)
- Uma única data de plantio pode não capturar toda a assinatura da cultura ao longo do ciclo
- Idealmente, seria necessário usar séries temporais (NDVI, EVI ao longo de meses)

### 4. **Regiões Geográficas**
- Os KMLs provêm de diferentes regiões do Brasil
- Variações pedológicas, climáticas e de manejo afetam a assinatura espectral
- Modelo pode não estar bem calibrado para todas as regiões

---

## Recomendações

### Imediatas
1. **Verificar o Focusnet em Produção**
   - 21.78% de acurácia é inaceitável para decisões agronômicas
   - Contactar provedor (SoftFocus) sobre possível degradação

2. **Análise de Dados**
   - Verificar se os KMLs estão corretos (validar geometrias, datas)
   - Confirmar se as labels de "true" no dataframe estão precisas

3. **Modelo Alternativo**
   - Considerar usar modelos de visão por computador (Vision Transformers, EfficientNet)
   - Treinar localmente com dados de satélite (Sentinel-2) processados

### Médio Prazo
1. **Usar Série Temporal**
   - Não usar uma única data — coletar NDVI/EVI ao longo de todo o ciclo vegetativo
   - Modelos de LSTM/ConvLSTM são mais apropriados para dados temporais

2. **Aumentar Resolução Espacial**
   - Considerar Planet Labs, Maxar, ou UAV se disponível orçamento
   - Sentinel-2 (10-20m) pode ser insuficiente para distinções finas

3. **Calibração Regional**
   - Treinar modelos por região geográfica (cerrado, caatinga, etc.)
   - Dados regionais reduzem variabilidade não controlada

### Longo Prazo
1. **Pipeline Híbrido**
   - Combinar aprendizado de máquina com índices espectrais (NDVI, NDMI, EVI)
   - Usar ensemble de modelos

2. **Feedback Loop**
   - Coletar dados de campo para validação
   - Retreinar modelo com ground truth regional

---

## Conclusão

O serviço Focusnet atual **não está adequado** para classificação confiável de culturas com os dados e parâmetros testados. A acurácia de 21.78% (apenas ~5% acima do aleatório para 5 classes) sugere problemas sistêmicos:

- Modelo pode estar degradado, desatualizado ou mal calibrado
- Dados de entrada (KML + datas) podem ser insuficientes
- Uma única imagem de satélite por polígono é tecnicamente inadequado

**Recomendação:** Não utilizar este serviço em produção até melhorias significativas serem implementadas.
