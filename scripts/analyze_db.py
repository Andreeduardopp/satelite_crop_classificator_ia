import sqlite3

import os
# Assumes run from root
db_path = os.path.join("data", "raw", "sample_teste_250.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(DISTINCT cultura) FROM culturas;")
num_cultures = cursor.fetchone()[0]

cursor.execute("SELECT DISTINCT cultura FROM culturas;")
cultures = [row[0] for row in cursor.fetchall()]

print(f"Total distinct cultures: {num_cultures}")
print("Cultures:")
for c in cultures:
    print(f" - {c}")

conn.close()
