# Guia de Migração: V1 → V2

## Pré-requisitos

As versões V2 requerem as mesmas dependências que V1. Nada novo para instalar.

```bash
pip list | grep -E "torch|xgboost|sklearn|numpy"
```

Esperado:
- `torch >= 2.0`
- `xgboost >= 1.5`
- `scikit-learn >= 0.24`
- `numpy >= 1.20`
- `scipy >= 1.5` (novo, para padronização)

---

## Opção 1: Migração Gradual (Recomendado)

### Passo 1: Manter V1 como Baseline

Não deletar scripts originais:
```bash
ls -la src/models/hibrido_xgboost_v1/
# extrator.py ← original (keep)
# train_xgboost.py ← original (keep)
# extrator_v2.py ← novo
# train_xgboost_v2.py ← novo
```

### Passo 2: Executar V2 em Paralelo

```bash
# 1. Gerar features com V2
python src/models/hibrido_xgboost_v1/extrator_v2.py

# Verificar saída
ls src/models/hibrido_xgboost_v1/features_extraidas_v2/
# X_treino.npy  y_treino.npy
# X_teste.npy   y_teste.npy
# standardization_treino.npz
# standardization_teste.npz
```

### Passo 3: Comparar Resultados

```bash
# Train com V2
python src/models/hibrido_xgboost_v1/train_xgboost_v2.py

# Ver métricas no JSON
cat src/models/hibrido_xgboost_v1/metrics/xgb_metrics_*.json
```

### Passo 4: Validar Qualidade

Se V2 tiver F1-score ≥ V1:
```bash
# Aprovado para produção! Atualizar scripts
cp src/models/hibrido_xgboost_v1/extrator_v2.py \
   src/models/hibrido_xgboost_v1/extrator.py

cp src/models/hibrido_xgboost_v1/train_xgboost_v2.py \
   src/models/hibrido_xgboost_v1/train_xgboost.py

# Backup dos originais
mv src/models/hibrido_xgboost_v1/extrator_v2.py \
   src/models/hibrido_xgboost_v1/extrator_v2_backup.py
```

---

## Opção 2: Migração Imediata (Agressiva)

Para quem confia nos testes:

```bash
# 1. Backup dos originais
mkdir -p backups/
cp src/models/hibrido_xgboost_v1/extrator.py backups/extrator_v1.py
cp src/models/hibrido_xgboost_v1/train_xgboost.py backups/train_xgboost_v1.py

# 2. Usar V2
cp src/models/hibrido_xgboost_v1/extrator_v2.py \
   src/models/hibrido_xgboost_v1/extrator.py

cp src/models/hibrido_xgboost_v1/train_xgboost_v2.py \
   src/models/hibrido_xgboost_v1/train_xgboost.py

# 3. Limpar features antigas (elas serão regeneradas)
rm -rf src/models/hibrido_xgboost_v1/features_extraidas/

# 4. Rodar pipeline inteira
cd src/models/hibrido_xgboost_v1/
python extrator.py && python train_xgboost.py
```

---

## Opção 3: Sem Mudanças no V1 (Conservador)

Se não quer tocar no código de produção:

```bash
# Rodar V2 em diretório separado
mkdir -p experiments/hybrid_v2_test/

# Copiar V2
cp src/models/hibrido_xgboost_v1/extrator_v2.py experiments/hybrid_v2_test/
cp src/models/hibrido_xgboost_v1/train_xgboost_v2.py experiments/hybrid_v2_test/

# Ajustar paths no topo dos scripts se necessário
cd experiments/hybrid_v2_test/
python extrator_v2.py
python train_xgboost_v2.py

# Comparar com V1
diff metrics/ ../../src/models/hibrido_xgboost_v1/metrics/
```

---

## Diferenças Comportamentais

### Feature Extraction

| Aspecto | V1 | V2 |
|---------|----|----|
| Saída | `features_extraidas/` | `features_extraidas_v2/` |
| Shapes | Não validados | Validados em cada batch |
| Valores | Sem check de NaN | Check de NaN/Inf |
| Padronização | Não | Sim (opcional) |
| Logging | Básico | Detalhado + estatísticas |

### Training

| Aspecto | V1 | V2 |
|---------|----|----|
| Compatibilidade | Features V1 | V1 + V2 (auto-detecta) |
| Métricas | Acurácia + F1/classe | Expandido: Precision, Recall, etc |
| Early Stopping | Não | Sim (20 rounds) |
| Feature Importance | Não | Top 10 features |
| Logs | Simples | Estruturado e detalhado |

---

## Verificação Pós-Migração

### Checklist Imediato

Após executar V2 pela primeira vez:

```bash
# 1. Features foram geradas?
test -f features_extraidas_v2/X_treino.npy && echo "✓ X_treino" || echo "✗ X_treino"
test -f features_extraidas_v2/y_treino.npy && echo "✓ y_treino" || echo "✗ y_treino"
test -f features_extraidas_v2/X_teste.npy && echo "✓ X_teste" || echo "✗ X_teste"
test -f features_extraidas_v2/y_teste.npy && echo "✓ y_teste" || echo "✗ y_teste"

# 2. Modelo foi treinado?
test -f artifacts/xgboost_modelo.json && echo "✓ Modelo" || echo "✗ Modelo"

# 3. Métricas foram salvas?
test -f metrics/xgb_metrics_*.json && echo "✓ Métricas" || echo "✗ Métricas"

# 4. Ver último F1-score
tail -20 <(python train_xgboost_v2.py 2>&1) | grep "F1"
```

### Testes Funcionais

```bash
# 1. Verificar se extrator produz boas features
python -c "
import numpy as np
X = np.load('features_extraidas_v2/X_treino.npy')
y = np.load('features_extraidas_v2/y_treino.npy')

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'X é finite? {np.all(np.isfinite(X))}')
print(f'Classes: {np.unique(y)}')
"

# 2. Comparar features V1 vs V2
python -c "
import numpy as np

X1 = np.load('features_extraidas/X_treino.npy')
X2 = np.load('features_extraidas_v2/X_treino.npy')

# V2 pode estar standardizada, então comparar shape
print(f'V1 shape: {X1.shape}')
print(f'V2 shape: {X2.shape}')
print(f'Shapes match? {X1.shape == X2.shape}')
"
```

### Comparação de Resultados

```bash
# Ver métricas V1
echo "=== V1 Metrics ==="
tail -30 metrics/xgb_metrics_*.json | grep f1_macro

# Ver métricas V2
echo "=== V2 Metrics ==="
tail -30 artifacts/../metrics/xgb_metrics_*.json | grep f1_macro

# Diferença (V2 deve ser ≥ V1)
```

---

## Rollback (Se Necessário)

Se V2 der problemas, voltar para V1:

```bash
# 1. Restaurar scripts
cp backups/extrator_v1.py src/models/hibrido_xgboost_v1/extrator.py
cp backups/train_xgboost_v1.py src/models/hibrido_xgboost_v1/train_xgboost.py

# 2. Usar features antigas
rm -rf src/models/hibrido_xgboost_v1/features_extraidas_v2/

# 3. Reexecutar
cd src/models/hibrido_xgboost_v1/
python extrator.py
python train_xgboost.py

# 4. Investigar problema
# - Checkar logs para mensagens de erro
# - Abrir issue com stack trace completo
```

---

## Dicas de Otimização Pós-Migração

### Se quiser Features Não-Padronizadas

Editar `extrator_v2.py`:
```python
class ExtractorConfig:
    STANDARDIZE_FEATURES: bool = False  # ← Mudar aqui
```

### Se quiser Mais/Menos Logging

Editar `extrator_v2.py`:
```python
class ExtractorConfig:
    COMPUTE_STATISTICS: bool = False  # ← Desabilitar stats
    VALIDATE_SHAPES: bool = False     # ← Desabilitar validação (mais rápido)
```

### Se quiser Mais Árvores XGBoost

Editar `train_xgboost_v2.py`:
```python
xgb_params = {
    'n_estimators': 500,  # ← Era 300
    'learning_rate': 0.05,  # ← Reduzir pra árvores mais fracas
}
```

---

## Troubleshooting de Migração

### Erro: "modules named 'scipy' not found"

```bash
pip install scipy
```

### Erro: "ValueError: Shape mismatch..."

- Verificar se backbone mudou
- Confirmar que EfficientNetV7 está carregado corretamente
- Rodar com `VALIDATE_SHAPES=False` para desabilitar validação

### Features V2 e V1 têm shapes diferentes

Esperado se V2 tiver standardização ativa. Shapes devem ser iguais:
```python
X_v1.shape == X_v2.shape  # Deve ser True (linhas e colunas)
```

Se forem diferentes, problema maior. Verificar:
```python
print(f"V1 features (treino): {X_v1.shape[1]}")
print(f"V2 features (treino): {X_v2.shape[1]}")
# Deve ser 3845 em ambos
```

### V2 é mais lento

- `VALIDATE_SHAPES=True` adiciona 2-3% overhead
- Desabilitar se velocidade for crítica:
```python
class ExtractorConfig:
    VALIDATE_SHAPES: bool = False
```

### V2 usa mais memória

- `COMPUTE_STATISTICS=True` computa mais informação
- Desabilitar se memória for limitada:
```python
class ExtractorConfig:
    COMPUTE_STATISTICS: bool = False
```

---

## Próximos Passos

### Curto Prazo (Semanas)
- ✓ Executar V2 em paralelo com V1
- ✓ Comparar métricas
- ✓ Validar que V2 ≥ V1 em F1-score

### Médio Prazo (Meses)
- ✓ Deletar V1 scripts após 1-2 rodadas bem-sucedidas
- ✓ Integrar V2 em CI/CD
- ✓ Treinar novos modelos apenas com V2

### Longo Prazo (Trimestral)
- Considerar GPU XGBoost (`tree_method='gpu_hist'`)
- Feature engineering: quais features são informativas?
- Hyperparameter tuning automático (Optuna, GridSearch)

---

## Contato & Suporte

Se encontrar problemas:

1. Verificar logs completos:
```bash
python extrator_v2.py 2>&1 | tee extrator_v2.log
python train_xgboost_v2.py 2>&1 | tee train_xgboost_v2.log
```

2. Procurar por mensagens de erro (ERROR, WARNING)

3. Comparar com logs de V1 (se disponível)

4. Abrir issue com:
   - Versões de dependências (`pip freeze`)
   - Erro completo (stack trace)
   - Comando exato usado
