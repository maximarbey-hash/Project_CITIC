"""
ÉTAPE 2 — Géocodage des adresses (adresse -> latitude / longitude)
==================================================================
Entrée  : data/interim/appartements.parquet
Sortie  : data/interim/appartements_geo.parquet   (+ colonnes lat, lon, geo_score, geo_type)

Service : API de géocodage de la Géoplateforme (IGN), gratuite, sans clé.
          Envoi par lots de fichiers CSV (max 50 Mo / 200 000 lignes par envoi).

Le script est REPRENABLE : chaque lot géocodé est sauvegardé dans un cache.
Si ça plante ou si tu coupes, relance-le : il repart là où il s'était arrêté.

Lancer depuis la racine du projet :  python src/02_geocodage.py
"""
from pathlib import Path
import io
import time
import pandas as pd
import requests

IN_PATH = Path("data/interim/appartements.parquet")
OUT_PATH = Path("data/interim/appartements_geo.parquet")
CACHE_DIR = Path("data/interim/geocodage_cache")
URL = "https://data.geopf.fr/geocodage/search/csv"
TAILLE_LOT = 20_000          # adresses par envoi (bien sous la limite de 200 000)
SCORE_MIN = 0.5              # en dessous, on considère le géocodage peu fiable

# Abréviations de types de voie utilisées par la DGFiP -> forme complète
ABREVIATIONS = {
    "AV": "AVENUE", "BD": "BOULEVARD", "ALL": "ALLEE", "CHE": "CHEMIN",
    "RTE": "ROUTE", "PL": "PLACE", "IMP": "IMPASSE", "QUA": "QUAI",
    "CRS": "COURS", "SQ": "SQUARE", "PAS": "PASSAGE", "VLA": "VILLA",
    "SEN": "SENTIER", "RPT": "ROND-POINT", "PRO": "PROMENADE",
    "FG": "FAUBOURG", "RES": "RESIDENCE", "CHS": "CHAUSSEE", "PRV": "PARVIS",
}


# --------------------------------------------------------------------------
# 1. PRÉPARER LA LISTE DES ADRESSES UNIQUES
# --------------------------------------------------------------------------
def developper(adresse: str) -> str:
    """'12 AV DU GEN SARRAIL' -> '12 AVENUE DU GEN SARRAIL'."""
    return " ".join(ABREVIATIONS.get(mot, mot) for mot in adresse.split())


def preparer_adresses(df: pd.DataFrame) -> pd.DataFrame:
    # Beaucoup de ventes partagent la même adresse (même immeuble) :
    # on ne géocode chaque adresse qu'une fois -> beaucoup moins d'appels.
    cols = ["adresse", "code_postal", "commune", "code_insee"]
    uniques = df[cols].drop_duplicates().reset_index(drop=True)
    uniques = uniques[uniques["adresse"].str.len() > 0]
    uniques["adresse_api"] = uniques["adresse"].map(developper)
    uniques["id_adresse"] = range(len(uniques))
    return uniques


# --------------------------------------------------------------------------
# 2. ENVOYER UN LOT À L'API (avec nouvelles tentatives en cas d'échec)
# --------------------------------------------------------------------------
def geocoder_lot(lot: pd.DataFrame, essais: int = 5) -> pd.DataFrame:
    csv = lot[["id_adresse", "adresse_api", "code_postal", "commune", "code_insee"]]
    contenu = csv.to_csv(index=False).encode("utf-8")
    parametres = [
        ("columns", "adresse_api"), ("columns", "code_postal"), ("columns", "commune"),
        ("citycode", "code_insee"),   # oblige le résultat à être dans la bonne commune
    ]
    for tentative in range(1, essais + 1):
        try:
            r = requests.post(URL, files={"data": ("lot.csv", contenu, "text/csv")},
                              data=parametres, timeout=600)
            if r.status_code == 200:
                res = pd.read_csv(io.StringIO(r.content.decode("utf-8")), dtype=str)
                if "latitude" not in res.columns:
                    raise ValueError(f"Réponse inattendue, colonnes : {list(res.columns)}")
                return res
            print(f"    HTTP {r.status_code} (tentative {tentative}) : {r.text[:150]}")
        except requests.RequestException as e:
            print(f"    Erreur réseau (tentative {tentative}) : {e}")
        time.sleep(10 * tentative)   # on attend de plus en plus longtemps
    raise RuntimeError("Échec du géocodage après plusieurs tentatives. Relance le script plus tard.")


# --------------------------------------------------------------------------
# 3. BOUCLE SUR LES LOTS, AVEC CACHE
# --------------------------------------------------------------------------
def geocoder_tout(uniques: pd.DataFrame) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_lots = (len(uniques) - 1) // TAILLE_LOT + 1
    resultats = []
    for i in range(n_lots):
        fichier_cache = CACHE_DIR / f"lot_{i:04d}.parquet"
        if fichier_cache.exists():
            resultats.append(pd.read_parquet(fichier_cache))
            continue
        lot = uniques.iloc[i * TAILLE_LOT:(i + 1) * TAILLE_LOT]
        debut = time.time()
        res = geocoder_lot(lot)
        res = res[["id_adresse", "latitude", "longitude", "result_score",
                   "result_type", "result_label"]]
        res.to_parquet(fichier_cache, index=False)
        resultats.append(res)
        print(f"  lot {i + 1}/{n_lots} géocodé en {time.time() - debut:.0f} s")
    return pd.concat(resultats, ignore_index=True)


# --------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# --------------------------------------------------------------------------
def main():
    df = pd.read_parquet(IN_PATH)
    print(f"{len(df):,} ventes, du {df['date'].min():%d/%m/%Y} au {df['date'].max():%d/%m/%Y}")

    uniques = preparer_adresses(df)
    print(f"{len(uniques):,} adresses uniques à géocoder "
          f"({(len(uniques) - 1) // TAILLE_LOT + 1} lots de {TAILLE_LOT:,})")

    geo = geocoder_tout(uniques)
    geo["id_adresse"] = geo["id_adresse"].astype(int)
    geo = geo.rename(columns={"latitude": "lat", "longitude": "lon",
                              "result_score": "geo_score", "result_type": "geo_type",
                              "result_label": "geo_label"})
    for c in ["lat", "lon", "geo_score"]:
        geo[c] = pd.to_numeric(geo[c], errors="coerce")

    # On rattache les coordonnées à chaque vente
    uniques = uniques.merge(geo, on="id_adresse", how="left")
    cols = ["adresse", "code_postal", "commune", "code_insee"]
    df = df.merge(uniques[cols + ["lat", "lon", "geo_score", "geo_type", "geo_label"]],
                  on=cols, how="left")

    # Géocodage peu fiable -> coordonnées effacées (on ne garde que du sûr)
    peu_fiable = df["geo_score"] < SCORE_MIN
    df.loc[peu_fiable, ["lat", "lon"]] = None

    # Bilan
    ok = df["lat"].notna()
    print(f"\nVentes géocodées de façon fiable : {ok.mean():.1%} ({ok.sum():,} / {len(df):,})")
    print("Précision obtenue (housenumber = au numéro près, street = à la rue) :")
    print(df.loc[ok, "geo_type"].value_counts(normalize=True).round(3).to_string())
    print(f"Score médian : {df.loc[ok, 'geo_score'].median():.2f}")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nFichier enregistré : {OUT_PATH}")


if __name__ == "__main__":
    main()
