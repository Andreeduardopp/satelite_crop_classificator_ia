# Plano de melhoria — predição de MILHO (XGBoost v4)

Data: 2026-06-12
Autor: André Pedroso (com Claude Code)

## Contexto

Teste em batch via API (`xgboost_kml_batch_20260611_163625`): 300 amostras,
acurácia global 92,67%. MILHO é o ponto fraco isolado: **68%** de acerto,
confundido principalmente com **FEIJAO (9)** e **ARROZ (5)**.

No treino (CV), MILHO já é a pior classe: recall **84,2%** (`metrics.json` do run
`runs_v4/20260611_140453`). Ou seja, MILHO é difícil *em distribuição* e piora em
produção (84% → 68%) — sinal de **domain shift** + **desalinhamento fenológico**,
não de desbalanceamento de classe (o treino tem 1000 amostras por classe — já é
perfeitamente balanceado).

## Diagnóstico das mudanças já feitas (não commitadas)

### 🔴 CRÍTICO — `is_c4` é vazamento de rótulo (target leakage)
```python
df["is_c4"] = (df["crop_label"] == "MILHO").astype(float)
df["is_c4_x_NDVI_peak_stage"]   = df["is_c4"] * df.get("NDVI_peak_stage", 0)
df["is_c4_x_EVI_senescence_rate"] = df["is_c4"] * df.get("EVI_senescence_rate", 0)
```
`is_c4` **é** o rótulo. Consequências:
- Treino: seleção de features coloca `is_c4` no topo; modelo aprende `is_c4=1 ⇔ MILHO`.
- `evaluate_test.py`: `crop_label` existe no DB de teste → `is_c4` é recriado a partir
  do gabarito → métricas offline ficam **infladas e falsas**.
- Produção (API): não existe `crop_label` → `is_c4` nunca é criado → feature mais
  importante some → modelo **para de prever MILHO**.
- **Ação: remover as 3 features.**

### 🟡 `compute_sample_weight("balanced")` é no-op e está meio-aplicado
- Treino já é balanceado (1000/classe) → pesos "balanced" saem todos iguais → não faz nada.
- Só é aplicado no objetivo do Optuna; o modelo final (`ensemble.fit`) e o
  `cross_val_predict` do relatório **não** usam `sample_weight` → inconsistente.
- **Ação: reverter para o `cross_val_score` limpo.** (Sem ganho aqui.)

### ✅ Mudanças boas (manter)
- Fix do import em `evaluate_test.py` (`train_xgboost_v4` em vez de `v3`) — correto e importante.
- Não dividir o rótulo em SAFRA/SAFRINHA (mantém o DB de 6k válido) — decisão certa.
- Features novas legítimas (`NDVI_c4_cumulative_half_stage`, `*_rise_*_vs_baseline`,
  `NDVI_c4_milestone_idx`) — derivadas do espectro, não vazam. Porém **duplicam** em parte
  `NDVI_peak_stage` e `greenup_rate`. Manter, ganho marginal.

## Causa raiz provável do erro em MILHO

1. **Desalinhamento de janela fenológica.** MILHO no Brasil é bimodal:
   safra (~150d) vs safrinha (~130d), mas `CROP_STAGE_WINDOWS["MILHO"]` usa janela
   única de 140d. Quando o ciclo real é menor, a janela `maturity` (115–140d) amostra
   solo nu/pós-colheita → trajetória NDVI parece de cultura curta → confunde com FEIJAO (90d).
   Casa com o erro dominante MILHO→FEIJAO.
2. **Estimativa de data de plantio ruim** quando o KML não tem plantio (`plantio_nan`):
   data é retro-calculada da colheita com `CROP_CYCLE_DAYS["MILHO"]=140` fixo. Ciclo
   errado desloca **todas** as features de estágio.

## Plano de ação

### Fase 0 — Decisão de dados (pedido do usuário)
- **Treinar somente com KMLs que possuem data de plantio real** (não estimada da colheita).
  - Filtrar em `load_data`: descartar linhas com `planting_date` nula/não-parseável.
  - Garantir que a extração de features de treino use apenas KMLs com `plantio_<data>`
    no nome (o `FILENAME_REGEX` de treino já exige `\d{2}-\d{2}-\d{2}`; confirmar que o
    DB foi construído assim e não com datas estimadas).
  - ⚠️ Pendência: o `features.db` referenciado no comando não existe no workspace; os
    `.db` presentes têm tabela `culturas`, não `phenology_features`. Confirmar com o
    usuário qual DB/regeração usar antes de re-treinar.

### Fase 1 — Correção de correção (antes de qualquer re-treino)
1. Remover `is_c4`, `is_c4_x_NDVI_peak_stage`, `is_c4_x_EVI_senescence_rate` de
   `engineer_features`.
2. Reverter `tune_xgb` para `cross_val_score` limpo (remover sample_weight balanceado);
   remover import `compute_sample_weight` se não usado.
3. Adicionar guarda: nenhuma feature engenheirada pode derivar de `crop_label`
   (evita reincidência do vazamento).
4. Filtro de `planting_date` válida em `load_data` (Fase 0).

### Fase 2 — Correção real do MILHO (maior impacto)
5. Adicionar feature booleana `is_safrinha` (derivada do mês de plantio, ex.: jan–mar).
   É feature, não rótulo → DB de 6k continua válido.
6. Tornar as janelas de estágio do MILHO sensíveis ao ciclo (escalar pela duração
   estimada em vez de 140d fixo).
7. Melhorar a estimativa de plantio para `plantio_nan` de MILHO (safra vs safrinha pelo
   mês da colheita) — relevante só para teste/produção, não para o treino "plantio real".

### Fase 3 — Validação honesta
8. Confirmar que o conjunto de teste é disjunto (região/tempo) do treino.
9. Reportar métricas no estilo produção (sem nenhuma feature derivada de `crop_label`).
10. Calibração de probabilidade por classe (confiança de 76% em erros de MILHO mostra
    que a confiança não é calibrada) → permitir flag de baixa confiança em MILHO.

### Fase 4 — Opcional (se 5–6 não bastarem)
11. Features estruturais de SAR (VH/VV): arquitetura de dossel C4 do milho difere de
    leguminosas C3 e é independente da sobreposição óptica.

## Ordem de execução proposta
Fase 1 (correção) → re-treino limpo p/ baseline confiável → Fase 2 (is_safrinha +
janelas por ciclo) → re-treino → Fase 3 (validação/calibração).

## Critério de sucesso
- Offline (held-out honesto) e produção concordando (gap < ~5 p.p.).
- Recall de MILHO de 68% → **80%+** sem inflar via vazamento.
