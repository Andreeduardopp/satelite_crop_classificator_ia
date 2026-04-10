# data_retreino — src

Pipeline de coleta, processamento e treinamento de modelos para classificação de culturas (milho, soja, trigo) a partir de imagens de satélite Sentinel-2.

## Estrutura

```
src/
├── treinamento/                        # Scripts de treinamento dos modelos
│   ├── treinar_classificador.py        # EfficientNetB0 com transfer learning (2 fases)
│   ├── treinar_classificador_vit.py    # Vision Transformer (ViT-B/16) com transfer learning (2 fases)
│   ├── treinar_transformer.py          # ViT via vit-keras
│   └── mlp.py                          # Protótipo MLP (experimental)
│
├── dados/                              # Ingestão e processamento de dados
│   ├── processamento_sentinel_Hub.py   # Requisições à API Sentinel Hub
│   ├── processamento_imagens.py        # Aplicação de máscara, threshold e cálculo de área
│   ├── gerador_dataframe.py            # Monta dataframe a partir de arquivos KML
│   ├── mover_processadas.py            # Organiza imagens processadas no filesystem
│   ├── google_engine.py                # Integração com Google Earth Engine
│   └── request_focusnet.py             # Requisições ao serviço FocusNet
│
├── banco/                              # Operações de banco de dados
│   ├── database.py                     # CRUD SQLite (criar tabela, importar CSV, exportar)
│   └── gerar_sample_treino.py          # Extrai amostra do banco real para sample_treino.db
│
├── avaliacao/                          # Avaliação e análise
│   ├── avaliar_imagens.py              # Verifica integridade e disponibilidade das imagens
│   └── stats.py                        # Análise estatística de distribuição por classe
│
├── pipeline.py                         # Orquestração: download → processamento → inferência
├── main.py                             # Inferência legada (modelo sigmoide)
└── utils.py                            # Utilitários compartilhados (cobertura de nuvens, temp)
```

## Fluxo geral

```
KMLs
  └─► gerador_dataframe.py     → dataframe.csv
        └─► pipeline.py        → baixa imagens (Sentinel Hub) + aplica máscaras
              └─► banco/       → persiste no SQLite
                    └─► gerar_sample_treino.py  → sample_treino.db
                          └─► treinamento/       → treina e salva modelo em modelos/
```

## Como rodar

Todos os scripts devem ser executados a partir do diretório `src/`:

```bash
# Gerar banco de amostra
python banco/gerar_sample_treino.py

# Avaliar integridade das imagens
python avaliacao/avaliar_imagens.py

# Treinar EfficientNet
python treinamento/treinar_classificador.py

# Treinar ViT
python treinamento/treinar_classificador_vit.py
```

## Dependências principais

- TensorFlow 2.x / Keras
- HuggingFace Transformers (`TFViTModel`)
- OpenCV, NumPy, scikit-learn
- sentinelhub, pandas, sqlite3
