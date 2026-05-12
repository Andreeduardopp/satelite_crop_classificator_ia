import sqlite3
import json
import ast

import os
# Assumes run from root
db_path = os.path.join("data", "raw", "sample_teste_250.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cultures = ['soja', 'milho', 'trigo', 'aveia', 'feijão']
results = {}

for culture in cultures:
    cursor.execute("SELECT id, cultura, mes, imagens_processadas FROM culturas WHERE cultura = ? LIMIT 1;", (culture,))
    row = cursor.fetchone()
    if not row:
        # try without encoding just in case
        cursor.execute("SELECT id, cultura, mes, imagens_processadas FROM culturas WHERE cultura LIKE ? LIMIT 1;", (culture[:4]+'%',))
        row = cursor.fetchone()
        
    if row:
        try:
            images = ast.literal_eval(row[3])
        except:
            images = row[3]
        results[culture] = {
            "id": row[0],
            "cultura": row[1],
            "mes": row[2],
            "imagens_processadas": images
        }

print(json.dumps(results, indent=2, ensure_ascii=False))
conn.close()
