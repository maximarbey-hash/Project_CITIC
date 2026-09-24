"""
ÉTAPE 1 — Nettoyage des fichiers DVF bruts
==========================================
Entrée  : data/raw/ValeursFoncieres-*.txt  (fichiers DGFiP, séparateur "|")
Sortie  : data/interim/appartements.parquet (1 ligne = 1 vente d'un appartement)

Lancer depuis la racine du projet :  python src/01_nettoyage.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 0. PARAMÈTRES — c'est ici qu'on change le périmètre (France entière = None)
# --------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/interim/appartements.parquet")
DEPARTEMENTS = ["75", "77", "78", "91", "92", "93", "94", "95"]  # Île-de-France
CHUNK = 500_000                      # lignes lues à la fois (économise la RAM)

SURFACE_MIN, SURFACE_MAX = 9, 400    # m²
VALEUR_MIN = 10_000                  # €
PRIX_M2_MIN, PRIX_M2_MAX = 500, 30_000

# Colonnes utiles (nom DVF -> nom court en Python)
COLONNES = {
    "No disposition": "no_disposition",
    "Date mutation": "date",
    "Nature mutation": "nature_mutation",
    "Valeur fonciere": "valeur_fonciere",
    "No voie": "no_voie",
    "B/T/Q": "btq",
    "Type de voie": "type_voie",
    "Voie": "voie",
    "Code postal": "code_postal",
    "Commune": "commune",
    "Code departement": "code_dep",
    "Code commune": "code_commune",
    "Prefixe de section": "prefixe_section",
    "Section": "section",
    "No plan": "no_plan",
    "1er lot": "lot1",
    "Surface Carrez du 1er lot": "carrez1",
    "Surface Carrez du 2eme lot": "carrez2",
    "Surface Carrez du 3eme lot": "carrez3",
    "Surface Carrez du 4eme lot": "carrez4",
    "Surface Carrez du 5eme lot": "carrez5",
    "Nombre de lots": "nb_lots",
    "Type local": "type_local",
    "Surface reelle bati": "surface_bati",
    "Nombre pieces principales": "nb_pieces",
}


def en_nombre(serie: pd.Series) -> pd.Series:
    """'468000,00' -> 468000.0 (DVF utilise la virgule décimale)."""
    return pd.to_numeric(serie.str.replace(",", ".", regex=False), errors="coerce")


# --------------------------------------------------------------------------
# 1. LECTURE — morceau par morceau, en ne gardant que les ventes du périmètre
# --------------------------------------------------------------------------
def lire_fichier(chemin: Path, departements) -> pd.DataFrame:
    morceaux = []
    for chunk in pd.read_csv(chemin, sep="|", usecols=list(COLONNES), dtype=str,
                             encoding="utf-8", chunksize=CHUNK):
        chunk = chunk.rename(columns=COLONNES)
        garde = chunk["nature_mutation"] == "Vente"
        if departements is not None:
            garde &= chunk["code_dep"].isin(departements)
        morceaux.append(chunk[garde])
    return pd.concat(morceaux, ignore_index=True)


# --------------------------------------------------------------------------
# 2. UNE VENTE = PLUSIEURS LIGNES -> on reconstitue les mutations
# --------------------------------------------------------------------------
def construire_ventes(df: pd.DataFrame, annee: str) -> pd.DataFrame:
    df = df[df["valeur_fonciere"].notna()].copy()

    # Pas d'identifiant de vente dans les fichiers publics : on en fabrique un.
    # Même date + même prix + même commune + même disposition = même vente.
    cle = ["date", "valeur_fonciere", "code_dep", "code_commune", "no_disposition"]
    df["id_mutation"] = annee + "|" + df[cle].fillna("").agg("|".join, axis=1)

    # Un même local peut apparaître plusieurs fois (une ligne par parcelle) : dédoublonnage
    locaux = df[df["type_local"].notna()].drop_duplicates(
        ["id_mutation", "type_local", "surface_bati", "nb_pieces", "lot1"])

    # Composition de chaque vente : combien d'appartements, maisons, dépendances...
    compo = pd.crosstab(locaux["id_mutation"], locaux["type_local"])
    for col in ["Appartement", "Maison", "Dépendance",
                "Local industriel. commercial ou assimilé"]:
        if col not in compo:
            compo[col] = 0

    # On garde les ventes d'UN seul appartement (+ éventuelles caves/parkings)
    ok = compo[(compo["Appartement"] == 1) & (compo["Maison"] == 0)
               & (compo["Local industriel. commercial ou assimilé"] == 0)]
    ventes = locaux[(locaux["type_local"] == "Appartement")
                    & locaux["id_mutation"].isin(ok.index)].copy()
    ventes["nb_dependances"] = ventes["id_mutation"].map(ok["Dépendance"])
    return ventes


# --------------------------------------------------------------------------
# 3. TYPAGE ET VARIABLES DÉRIVÉES
# --------------------------------------------------------------------------
def typer(ventes: pd.DataFrame) -> pd.DataFrame:
    v = ventes
    v["date"] = pd.to_datetime(v["date"], format="%d/%m/%Y")
    v["annee"] = v["date"].dt.year
    v["valeur_fonciere"] = en_nombre(v["valeur_fonciere"])
    v["surface_bati"] = en_nombre(v["surface_bati"])
    v["nb_pieces"] = en_nombre(v["nb_pieces"])
    v["nb_lots"] = en_nombre(v["nb_lots"])

    # Surface : la Carrez (loi, plus fiable) si elle est cohérente avec le bâti
    carrez = pd.concat([en_nombre(v[f"carrez{i}"]) for i in range(1, 6)], axis=1)
    v["carrez"] = carrez.sum(axis=1, min_count=1)
    coherent = v["carrez"].between(0.7 * v["surface_bati"], 1.2 * v["surface_bati"])
    v["surface"] = np.where(coherent, v["carrez"], v["surface_bati"])
    v["prix_m2"] = v["valeur_fonciere"] / v["surface"]

    # Codes géographiques propres
    v["code_postal"] = v["code_postal"].str.zfill(5)
    dep2 = v["code_dep"].str[:2]                        # 971 -> 97 (DOM)
    v["code_insee"] = dep2 + v["code_commune"].str.zfill(3)
    v["id_parcelle"] = (v["code_insee"] + v["prefixe_section"].fillna("000").str.zfill(3)
                        + v["section"].str.zfill(2) + v["no_plan"].str.zfill(4))

    # Adresse lisible (servira au géocodage)
    parties = v[["no_voie", "btq", "type_voie", "voie"]].fillna("")
    v["adresse"] = parties.agg(" ".join, axis=1).str.split().str.join(" ")
    return v


# --------------------------------------------------------------------------
# 4. FILTRES DE QUALITÉ
# --------------------------------------------------------------------------
def filtrer(v: pd.DataFrame) -> pd.DataFrame:
    n0 = len(v)
    v = v[(v["valeur_fonciere"] >= VALEUR_MIN)
          & v["surface"].between(SURFACE_MIN, SURFACE_MAX)
          & (v["nb_pieces"] >= 1)
          & v["prix_m2"].between(PRIX_M2_MIN, PRIX_M2_MAX)].copy()
    print(f"  filtres globaux     : {n0:>9,} -> {len(v):,}")

    # Filtre local : on retire les prix/m² très éloignés de ceux de LEUR commune
    # (z-score robuste sur le log, pour les communes avec assez de ventes)
    v["log_pm2"] = np.log(v["prix_m2"])
    g = v.groupby("code_insee")["log_pm2"]
    mediane = g.transform("median")
    mad = g.transform(lambda x: (x - x.median()).abs().median())
    z = (v["log_pm2"] - mediane) / (1.4826 * mad.replace(0, np.nan))
    aberrant = (g.transform("size") >= 30) & (z.abs() > 3.5)
    n1 = len(v)
    v = v[~aberrant].drop(columns="log_pm2")
    print(f"  filtre par commune  : {n1:>9,} -> {len(v):,}")
    return v


# --------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# --------------------------------------------------------------------------
def main(raw_dir=RAW_DIR, out_path=OUT_PATH, departements=DEPARTEMENTS):
    fichiers = sorted(Path(raw_dir).glob("ValeursFoncieres-*.txt"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier DVF trouvé dans {raw_dir}")

    toutes = []
    for f in fichiers:
        annee = f.stem.split("-")[-1]
        print(f"{f.name}")
        brut = lire_fichier(f, departements)
        ventes = construire_ventes(brut, annee)
        print(f"  lignes gardées      : {len(brut):>9,}  ->  ventes d'1 appart : {len(ventes):,}")
        toutes.append(ventes)

    v = typer(pd.concat(toutes, ignore_index=True))
    print("Filtres qualité (toutes années)")
    v = filtrer(v)

    colonnes_finales = [
        "id_mutation", "date", "annee", "valeur_fonciere", "prix_m2", "surface",
        "surface_bati", "carrez", "nb_pieces", "nb_dependances", "nb_lots",
        "adresse", "code_postal", "commune", "code_dep", "code_insee", "id_parcelle",
    ]
    v = v[colonnes_finales].sort_values("date").reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    v.to_parquet(out_path, index=False)
    print(f"\n{len(v):,} ventes enregistrées dans {out_path}")
    print(v.groupby("annee")["prix_m2"].agg(["size", "median"]).round(0))
    return v


if __name__ == "__main__":
    main()
