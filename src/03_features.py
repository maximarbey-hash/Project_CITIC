"""
ÉTAPE 4 — Construction des variables (features) du modèle
=========================================================
Entrée  : data/interim/appartements_geo_dpe.parquet (ou appartements_geo.parquet)
Sortie  : data/processed/dataset.parquet   (1 ligne = 1 vente + toutes ses variables)
          data/processed/gares.parquet     (réutilisé plus tard par le site)

Variables créées :
  - Bien       : surface, pièces, surface par pièce, dépendances
  - Temps      : t_mois (nombre de mois depuis janv. 2020)
  - Transports : distance à la gare la plus proche, au métro/RER/train, nb de gares à 1 km
  - Centralité : distance au centre de Paris
  - Indice     : indice de prix « à qualité constante » par zone et par mois
                 (Paris / petite couronne / grande couronne), sauvegardé à part
  - Voisinage  : prix/m² médian des ventes voisines des 24 MOIS PRÉCÉDENTS,
                 RÉACTUALISÉS au mois de la vente grâce à l'indice
                 (jamais de ventes futures -> pas de triche, cf. explications)

Lancer depuis la racine du projet :  python src/03_features.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree
from sklearn.preprocessing import OneHotEncoder

# Fichier enrichi par les DPE (étape 3b) s'il existe, sinon fichier géocodé simple
IN_PATH = next(p for p in [Path("data/interim/appartements_geo_dpe.parquet"),
                           Path("data/interim/appartements_geo.parquet")] if p.exists())
OUT_PATH = Path("data/processed/dataset.parquet")
GARES_CSV = Path("data/external/gares.csv")
GARES_OUT = Path("data/processed/gares.parquet")
INDICE_OUT = Path("data/processed/indice_prix.parquet")
URL_GARES = ("https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/"
             "emplacement-des-gares-idf-data-generalisee/exports/csv?delimiter=%3B")

RAYON_TERRE = 6_371_000          # m
CENTRE_PARIS = (48.8534, 2.3488)  # parvis de Notre-Dame (point zéro des routes de France)
FENETRE_MOIS = 24                 # historique utilisé pour les prix voisins
K_PROCHE, K_LARGE = 10, 30        # nombre de voisins
ZONES = {"75": 0, "92": 1, "93": 1, "94": 1}   # 0 Paris, 1 petite couronne, 2 le reste
NOMS_ZONES = ["Paris", "Petite couronne", "Grande couronne"]


def en_radians(df: pd.DataFrame) -> np.ndarray:
    """BallTree en métrique 'haversine' attend [lat, lon] en radians."""
    return np.radians(df[["lat", "lon"]].to_numpy())


# --------------------------------------------------------------------------
# 1. GARES ET STATIONS (Île-de-France Mobilités, open data)
# --------------------------------------------------------------------------
def charger_gares() -> pd.DataFrame:
    if not GARES_CSV.exists():
        print("Téléchargement des gares IDFM...")
        GARES_CSV.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(URL_GARES, timeout=120)
            r.raise_for_status()
            GARES_CSV.write_bytes(r.content)
        except requests.RequestException as e:
            raise SystemExit(
                f"Téléchargement impossible ({e}).\n"
                "Télécharge le fichier à la main : https://data.iledefrance-mobilites.fr/"
                "explore/dataset/emplacement-des-gares-idf-data-generalisee/export/ "
                "-> CSV, puis enregistre-le sous data/external/gares.csv")

    g = pd.read_csv(GARES_CSV, sep=";")
    col_geo = next(c for c in g.columns if "geo_point" in c.lower())
    latlon = g[col_geo].astype(str).str.split(",", expand=True).astype(float)
    g["lat"], g["lon"] = latlon[0], latlon[1]

    col_mode = next((c for c in ["mode", "res_com"] if c in g.columns), None)
    g["mode"] = g[col_mode].astype(str).str.upper() if col_mode else "INCONNU"
    g["lourd"] = g["mode"].str.contains("METRO|RER|TRAIN")   # vs tram, val, funiculaire
    g = g[["lat", "lon", "mode", "lourd"]].dropna(subset=["lat", "lon"])
    print(f"{len(g):,} gares/stations chargées ({g['lourd'].sum():,} métro/RER/train)")
    return g


def ajouter_transports(df: pd.DataFrame, gares: pd.DataFrame) -> pd.DataFrame:
    X = en_radians(df)
    for nom, sous_ens in [("gare", gares), ("metro_rer", gares[gares["lourd"]])]:
        arbre = BallTree(en_radians(sous_ens), metric="haversine")
        dist, _ = arbre.query(X, k=1)
        df[f"dist_{nom}_m"] = dist[:, 0] * RAYON_TERRE
    arbre = BallTree(en_radians(gares), metric="haversine")
    df["nb_gares_1km"] = arbre.query_radius(X, r=1000 / RAYON_TERRE, count_only=True)
    return df


# --------------------------------------------------------------------------
# 2. DISTANCE AU CENTRE DE PARIS
# --------------------------------------------------------------------------
def distance_m(lat, lon, lat0, lon0):
    """Formule de haversine : distance à vol d'oiseau sur la sphère terrestre."""
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 2 * RAYON_TERRE * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------
# 3. INDICE DE PRIX À QUALITÉ CONSTANTE (méthode hédonique)
# --------------------------------------------------------------------------
def calculer_indice(df: pd.DataFrame):
    """
    Régression : log(prix/m²) = effet(zone x mois) + effet(commune) + effet(taille)
                                + effet(pièces) + effet(dépendances) + bruit
    Le coefficient « zone x mois » mesure l'évolution du prix d'un MÊME type de
    bien au MÊME endroit : c'est l'inflation immobilière « like-for-like ».
    """
    d = pd.DataFrame({
        "zone_mois": df["zone"].astype(str) + "_" + df["mois_idx"].astype(str),
        "commune": df["code_insee"],
        "taille": pd.qcut(df["surface"], 10, labels=False, duplicates="drop").astype(str),
        "pieces": df["nb_pieces"].clip(1, 6).astype(int).astype(str),
        "dependances": df["nb_dependances"].clip(0, 2).astype(int).astype(str),
    })
    enc = OneHotEncoder(handle_unknown="ignore")
    X = enc.fit_transform(d)
    y = np.log(df["prix_m2"].to_numpy())
    modele = Ridge(alpha=1.0, solver="sparse_cg").fit(X, y)

    # On récupère les coefficients zone x mois -> tableau [zone, mois]
    noms = enc.get_feature_names_out()
    n_mois = int(df["mois_idx"].max()) + 1
    table = np.full((len(NOMS_ZONES), n_mois), np.nan)
    for nom, coef in zip(noms, modele.coef_):
        if nom.startswith("zone_mois_"):
            z, m = map(int, nom.removeprefix("zone_mois_").split("_"))
            table[z, m] = coef
    # Mois sans vente dans une zone : on reprend le mois précédent
    table = pd.DataFrame(table).T.ffill().bfill().T.to_numpy().copy()
    table -= table[:, [0]]                 # base : premier mois = 0 (en log)
    return table


def sauver_indice(table: np.ndarray, premier_mois: pd.Period):
    lignes = []
    for z, nom in enumerate(NOMS_ZONES):
        for m in range(table.shape[1]):
            lignes.append({"zone": nom, "mois": (premier_mois + m).to_timestamp(),
                           "indice_base100": 100 * np.exp(table[z, m])})
    indice = pd.DataFrame(lignes)
    indice.to_parquet(INDICE_OUT, index=False)
    # Affichage : indice de décembre de chaque année
    dec = indice[indice["mois"].dt.month == 12].copy()
    dec["annee"] = dec["mois"].dt.year
    print("\nIndice de prix à qualité constante (base 100 = premier mois), en décembre :")
    print(dec.pivot(index="annee", columns="zone", values="indice_base100")
             [NOMS_ZONES].round(1).to_string())


# --------------------------------------------------------------------------
# 4. PRIX DES VENTES VOISINES — uniquement dans le PASSÉ, réactualisés
# --------------------------------------------------------------------------
def ajouter_prix_voisins(df: pd.DataFrame, indice: np.ndarray) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    mois = df["mois_idx"].to_numpy()
    zone = df["zone"].to_numpy()
    X = en_radians(df)
    log_prix = np.log(df["prix_m2"].to_numpy())
    resultats = np.full((len(df), 3), np.nan)

    for m in np.unique(mois):
        cibles = np.flatnonzero(mois == m)
        historique = np.flatnonzero((mois < m) & (mois >= m - FENETRE_MOIS))
        if len(historique) < K_LARGE:
            continue  # premiers mois : pas encore assez d'historique -> reste vide (NaN)
        # Réactualisation : prix d'une vente passée ramené au niveau de prix du mois m
        zh, mh = zone[historique], mois[historique]
        log_prix_actualise = log_prix[historique] - indice[zh, mh] + indice[zh, m]

        arbre = BallTree(X[historique], metric="haversine")
        dist, idx = arbre.query(X[cibles], k=K_LARGE)   # voisins triés du + proche au + loin
        voisins = log_prix_actualise[idx]
        resultats[cibles, 0] = np.exp(np.median(voisins[:, :K_PROCHE], axis=1))
        resultats[cibles, 1] = np.exp(np.median(voisins, axis=1))
        resultats[cibles, 2] = dist[:, -1] * RAYON_TERRE   # rayon qui contient les 30 voisins

    df["knn10_prix_m2"], df["knn30_prix_m2"], df["knn30_rayon_m"] = resultats.T
    return df


# --------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# --------------------------------------------------------------------------
def main():
    df = pd.read_parquet(IN_PATH)
    print(f"Lecture de {IN_PATH}")
    df = df[df["lat"].notna()].copy()
    print(f"{len(df):,} ventes géolocalisées")

    # Bien
    df["surface_par_piece"] = df["surface"] / df["nb_pieces"]
    # Temps
    df["t_mois"] = (df["date"].dt.year - 2020) * 12 + df["date"].dt.month
    # Centralité
    df["dist_centre_paris_m"] = distance_m(df["lat"], df["lon"], *CENTRE_PARIS)
    # Transports
    gares = charger_gares()
    df = ajouter_transports(df, gares)
    # Indice de prix à qualité constante
    periode = df["date"].dt.to_period("M")
    premier_mois = periode.min()
    df["mois_idx"] = (periode - premier_mois).apply(lambda x: x.n)
    df["zone"] = df["code_dep"].map(ZONES).fillna(2).astype(int)
    print("Calcul de l'indice de prix à qualité constante...")
    indice = calculer_indice(df)
    sauver_indice(indice, premier_mois)
    # Voisinage (prix réactualisés)
    print("\nCalcul des prix voisins, mois par mois...")
    df = ajouter_prix_voisins(df, indice)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    gares.to_parquet(GARES_OUT, index=False)

    # Bilan
    print(f"\n{len(df):,} ventes enregistrées dans {OUT_PATH}")
    sans_histo = df["knn30_prix_m2"].isna().mean()
    print(f"Ventes sans historique voisin (tout début de période) : {sans_histo:.1%}")
    print("\nMédianes des nouvelles variables :")
    cols = ["dist_gare_m", "dist_metro_rer_m", "nb_gares_1km", "dist_centre_paris_m",
            "knn10_prix_m2", "knn30_prix_m2", "knn30_rayon_m"]
    print(df[cols].median().round(0).to_string())
    corr = df[["prix_m2", "knn10_prix_m2", "knn30_prix_m2"]].corr(method="spearman")["prix_m2"]
    print("\nCorrélation (Spearman) avec le vrai prix/m² :")
    print(corr.drop("prix_m2").round(3).to_string())


if __name__ == "__main__":
    main()
