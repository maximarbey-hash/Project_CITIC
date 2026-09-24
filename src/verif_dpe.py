import pandas as pd
df = pd.read_parquet("data/processed/dataset.parquet")
df["ecart_quartier_%"] = 100 * (df["prix_m2"] / df["knn30_prix_m2"] - 1)
df["taille"] = pd.cut(df["surface"], [0, 30, 50, 80, 400], labels=["<30 m²", "30-50", "50-80", "80+"])
lettres = dict(enumerate("ABCDEFG", 1))
print(df.groupby("dpe_classe")["ecart_quartier_%"].median().round(1).rename(index=lettres))
print(df.pivot_table("ecart_quartier_%", "dpe_classe", "taille", "median").round(1).rename(index=lettres))