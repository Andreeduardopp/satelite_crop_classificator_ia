import sqlite3
import ast

conn = sqlite3.connect('src/bkp/26-04-06.dados.db')
rows = conn.execute("SELECT cultura, imagens_processadas FROM culturas WHERE imagens_processadas IS NOT NULL AND imagens_processadas != '[]'").fetchall()

counts = {}
for c, imgs in rows:
    try:
        p = ast.literal_eval(imgs)
        # Permite 2 ou 3 imagens
        if isinstance(p, list) and 2 <= len(p) <= 3:
            counts[c] = counts.get(c, 0) + 1
    except:
        pass

print("Contagem V8 (2 ou 3 Imagens):")
for c in sorted(counts, key=counts.get, reverse=True):
    print(f"{c}: {counts[c]}")
