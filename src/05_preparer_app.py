"""
ÉTAPE 6a — Préparer les fichiers dont le site a besoin
======================================================
Le site ne doit pas embarquer les 450 000 ventes ni les fichiers bruts :
on ne lui donne que le strict nécessaire, dans le dossier app/.

  app/data/ventes_recentes.parquet  ventes des 24 derniers mois, prix RÉACTUALISÉS
  app/data/gares.parquet            gares et stations
  app/data/indice_prix.parquet      indice de prix par zone
  app/data/batiments_dpe.parquet    époque de construction par adresse (DPE)
  app/data/dpe_marche.parquet       analyse du DPE pour l'onglet marché
  app/models/                       les modèles compressés (.txt.gz) + config.json
  app/assets/importance_variables.png

Lancer depuis la racine du projet :  python src/05_preparer_app.py
(à relancer après chaque réentraînement du modèle)
"""
from pathlib import Path
import gzip
import shutil
import numpy as np
import pandas as pd

FENETRE_MOIS = 24
NOMS_ZONES = ["Paris", "Petite couronne", "Grande couronne"]
APP = Path("app")


def analyse_dpe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque étiquette DPE et chaque zone :
      - part des ventes et prix/m² médian BRUT (trompeur : mélange lieu et taille)
      - effet de l'étiquette À QUARTIER ET TAILLE ÉGAUX, par rapport à D :
        régression de log(prix / prix des 30 voisins) sur l'étiquette + la taille
    """
    d = df.dropna(subset=["dpe_classe", "knn30_prix_m2"]).copy()
    d["lettre"] = d["dpe_classe"].astype(int).map(dict(enumerate("ABCDEFG", 1)))
    d["taille"] = pd.cut(d["surface"], [0, 25, 35, 50, 70, 100, 1000]).astype(str)
    d["y"] = np.log(d["prix_m2"] / d["knn30_prix_m2"])
    lignes = []
    zones = {"Île-de-France": d, **{n: d[d["zone"] == i] for i, n in enumerate(NOMS_ZONES)}}
    for nom_zone, z in zones.items():
        X = pd.concat([pd.get_dummies(z["lettre"]).drop(columns="D", errors="ignore"),
                       pd.get_dummies(z["taille"], drop_first=True)], axis=1).astype(float)
        X.insert(0, "constante", 1.0)
        coef = pd.Series(np.linalg.lstsq(X.to_numpy(), z["y"].to_numpy(), rcond=None)[0],
                         index=X.columns)
        for lettre, g in z.groupby("lettre"):
            lignes.append({
                "zone": nom_zone, "classe": lettre, "nb_ventes": len(g),
                "part_ventes_%": 100 * len(g) / len(z),
                "prix_m2_median_brut": g["prix_m2"].median(),
                "effet_vs_D_%": 0.0 if lettre == "D" else 100 * (np.exp(coef.get(lettre, 0)) - 1),
            })
    return pd.DataFrame(lignes)


def main():
    for sous_dossier in ["data", "models", "assets"]:
        (APP / sous_dossier).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet("data/processed/dataset.parquet")
    indice = pd.read_parquet("data/processed/indice_prix.parquet")

    # 1. Ventes des 24 derniers mois
    dernier = df["mois_idx"].max()
    rec = df[df["mois_idx"] > dernier - FENETRE_MOIS].copy()

    # 2. Réactualisation au niveau de prix du dernier mois (même logique que l'étape 4)
    rec["nom_zone"] = rec["zone"].map(dict(enumerate(NOMS_ZONES)))
    rec["mois"] = rec["date"].dt.to_period("M").dt.to_timestamp()
    rec = rec.merge(indice.rename(columns={"zone": "nom_zone"}), on=["nom_zone", "mois"], how="left")
    derniers = indice.sort_values("mois").groupby("zone")["indice_base100"].last()
    rec["prix_m2_actualise"] = rec["prix_m2"] * rec["nom_zone"].map(derniers) / rec["indice_base100"]

    colonnes = ["date", "lat", "lon", "prix_m2", "prix_m2_actualise", "surface",
                "nb_pieces", "adresse", "commune", "zone"]
    rec = rec[colonnes].dropna(subset=["lat", "lon", "prix_m2_actualise"])
    # Confidentialité (règles de réutilisation DVF) : le site publié ne contient que le nom
    # de la rue, jamais le numéro -> '12 B RUE DE LA ROQUETTE' devient 'RUE DE LA ROQUETTE'
    rec["adresse"] = rec["adresse"].str.replace(r"^\s*\d+\s*(?:[A-Z]\b)?\s*", "", regex=True)
    rec["zone"] = rec["zone"].astype(np.int8)
    rec.to_parquet(APP / "data" / "ventes_recentes.parquet", index=False)
    print(f"{len(rec):,} ventes récentes ({rec['date'].min():%m/%Y} -> {rec['date'].max():%m/%Y})")

    # 3. Analyse de marché du DPE (pour l'onglet « Le marché »)
    if "dpe_classe" in df.columns:
        dpe = analyse_dpe(df)
        dpe.to_parquet(APP / "data" / "dpe_marche.parquet", index=False)
        idf = dpe[dpe["zone"] == "Île-de-France"].set_index("classe")["effet_vs_D_%"].round(1)
        print("Effet de l'étiquette à quartier et taille égaux (vs D, Île-de-France) :")
        print(idf.to_string())

    # 4. Copie des autres fichiers
    shutil.copy("data/processed/gares.parquet", APP / "data")
    shutil.copy("data/processed/indice_prix.parquet", APP / "data")
    if Path("data/processed/batiments_dpe.parquet").exists():
        shutil.copy("data/processed/batiments_dpe.parquet", APP / "data")
    # Modèles compressés (.txt.gz, environ 5 fois plus légers) : indispensable pour
    # que l'envoi sur GitHub passe. On supprime d'abord les anciennes versions.
    for ancien in (APP / "models").glob("*"):
        ancien.unlink()
    for f in Path("models").glob("lgbm_*.txt"):
        with open(f, "rb") as src, gzip.open(APP / "models" / f"{f.name}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    shutil.copy("models/config.json", APP / "models")
    image = Path("reports/importance_variables.png")
    if image.exists():
        shutil.copy(image, APP / "assets")

    taille = sum(f.stat().st_size for f in APP.rglob("*") if f.is_file()) / 1e6
    print(f"Dossier app/ prêt ({taille:.0f} Mo au total). Fichiers les plus lourds :")
    for f in sorted(APP.rglob("*"), key=lambda f: f.stat().st_size, reverse=True)[:5]:
        print(f"  {f.stat().st_size / 1e6:6.1f} Mo  {f}")


if __name__ == "__main__":
    main()
