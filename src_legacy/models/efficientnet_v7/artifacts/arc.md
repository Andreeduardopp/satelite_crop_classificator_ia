Treino para prever 5 culturas: soja, milho, trigo, aveia e feijão.

O modelo utiliza EfficientNetB0 como base, com FiLM(dia, mes_embedding) e Temporal Attention.

Foram utilizados 6000 registros para treino e 200 para teste.

primeiro teste - 2500 samples
2026-04-13 16:59:31,776 - INFO - RESULTADO FINAL - Acurácia: 0.8570 | F1-macro: 0.8562
2026-04-13 16:59:31,776 - INFO - ---
2026-04-13 16:59:31,776 - INFO -   F1 soja: 0.8204
2026-04-13 16:59:31,776 - INFO -   F1 milho: 0.7194
2026-04-13 16:59:31,777 - INFO -   F1 trigo: 0.9950
2026-04-13 16:59:31,777 - INFO -   F1 aveia: 0.9900
2026-04-13 16:59:31,777 - INFO -   F1 feijão: 0.7563
2026-04-13 16:59:31,777 - INFO - ---
2026-04-13 16:59:31,788 - INFO - Classification Report:
              precision    recall  f1-score   support

        soja       0.80      0.84      0.82       200
       milho       0.73      0.70      0.72       200
       trigo       0.99      0.99      0.99       200
       aveia       0.99      0.99      0.99       200
      feijão       0.77      0.74      0.76       200

    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.86      1000
weighted avg       0.86      0.86      0.86      1000

2026-04-13 16:59:31,791 - INFO - Confusion Matrix (linhas=real, colunas=predito):
2026-04-13 16:59:31,792 - INFO -                 soja     milho     trigo     aveia    feijão
2026-04-13 16:59:31,792 - INFO -       soja       169        15         0         0        16
2026-04-13 16:59:31,793 - INFO -      milho        26       141         1         3        29
2026-04-13 16:59:31,794 - INFO -      trigo         0         1       199         0         0
2026-04-13 16:59:31,794 - INFO -      aveia         0         1         0       199         0
2026-04-13 16:59:31,794 - INFO -     feijão        17        34         0         0       149
2026-04-13 16:59:31,795 - INFO - Tempo médio de inferência puro: 103.83 ms/talhão (1000 talhões)

-------------------------
segundo teste - 6000 max 2500 min samples samples
2026-04-13 17:41:27,631 - INFO - Pesos carregados com sucesso!
2026-04-13 17:41:27,631 - INFO - Iniciando inferência silenciosamente aguarde...
2026-04-13 17:43:29,400 - INFO - RESULTADO FINAL - Acurácia: 0.8770 | F1-macro: 0.8763
2026-04-13 17:43:29,400 - INFO - ---
2026-04-13 17:43:29,400 - INFO -   F1 soja: 0.8530
2026-04-13 17:43:29,400 - INFO -   F1 milho: 0.7544
2026-04-13 17:43:29,400 - INFO -   F1 trigo: 1.0000
2026-04-13 17:43:29,400 - INFO -   F1 aveia: 0.9975
2026-04-13 17:43:29,400 - INFO -   F1 feijão: 0.7763
2026-04-13 17:43:29,400 - INFO - ---
2026-04-13 17:43:29,411 - INFO - Classification Report:
              precision    recall  f1-score   support

        soja       0.82      0.89      0.85       200
       milho       0.76      0.74      0.75       200
       trigo       1.00      1.00      1.00       200
       aveia       1.00      1.00      1.00       200
      feijão       0.80      0.76      0.78       200

    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.88      1000
weighted avg       0.88      0.88      0.88      1000

2026-04-13 17:43:29,413 - INFO - Confusion Matrix (linhas=real, colunas=predito):
2026-04-13 17:43:29,413 - INFO -                 soja     milho     trigo     aveia    feijão
2026-04-13 17:43:29,415 - INFO -       soja       177        15         0         0         8
2026-04-13 17:43:29,415 - INFO -      milho        20       149         0         1        30
2026-04-13 17:43:29,415 - INFO -      trigo         0         0       200         0         0
2026-04-13 17:43:29,415 - INFO -      aveia         0         0         0       200         0
2026-04-13 17:43:29,415 - INFO -     feijão        18        31         0         0       151
2026-04-13 17:43:29,415 - INFO - Tempo médio de inferência puro: 121.76 ms/talhão (1000 talhões)

---------------
Terceiro test 9000 max talhão 

2026-04-14 07:17:59,120 - INFO - RESULTADO FINAL - Acurácia: 0.8552 | F1-macro: 0.8547
2026-04-14 07:17:59,120 - INFO - ---
2026-04-14 07:17:59,121 - INFO -   F1 soja: 0.8226
2026-04-14 07:17:59,121 - INFO -   F1 milho: 0.7386
2026-04-14 07:17:59,121 - INFO -   F1 trigo: 1.0000
2026-04-14 07:17:59,122 - INFO -   F1 aveia: 0.9940
2026-04-14 07:17:59,122 - INFO -   F1 feijão: 0.7183
2026-04-14 07:17:59,122 - INFO - ---
2026-04-14 07:17:59,128 - INFO - Classification Report:
              precision    recall  f1-score   support

        soja       0.80      0.84      0.82       250
       milho       0.77      0.71      0.74       250
       trigo       1.00      1.00      1.00       250
       aveia       0.99      1.00      0.99       250
      feijão       0.71      0.72      0.72       250

    accuracy                           0.86      1250
   macro avg       0.85      0.86      0.85      1250
weighted avg       0.85      0.86      0.85      1250

2026-04-14 07:17:59,129 - INFO - Confusion Matrix (linhas=real, colunas=predito):
2026-04-14 07:17:59,129 - INFO -                 soja     milho     trigo     aveia    feijão       6-04-14 07:17:59,129 - INFO -       soja       211        17         0         0        22
    accuracy                           0.86      1250
   macro avg       0.85      0.86      0.85      1250
weighted avg       0.85      0.86      0.85      1250

2026-04-14 07:17:59,129 - INFO - Confusion Matrix (linhas=real, colunas=predito):
2026-04-14 07:17:59,129 - INFO -                 soja     milho     trigo     aveia    feijão    
2026-04-14 07:17:59,129 - INFO -       soja       211        17         0         0        22    
2026-04-14 07:17:59,131 - INFO -      milho        21       178         0         1        50    
2026-04-14 07:17:59,131 - INFO -      trigo         0         0       250         0         0    
2026-04-14 07:17:59,131 - INFO -      aveia         0         0         0       249         1    
2026-04-14 07:17:59,131 - INFO -     feijão        31        37         0         1       181    
2026-04-14 07:17:59,131 - INFO - Tempo médio de inferência puro: 55.44 ms/talhão (1250 talhões)  