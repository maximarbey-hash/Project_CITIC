"""
ÉTAPE 3b — Croisement avec les DPE de l'ADEME
=============================================
Entrée  : data/interim/appartements_geo.parquet
Sorties : data/external/dpe_cache/*.parquet          (DPE téléchargés, par département)
          data/interim/appartements_geo_dpe.parquet  (ventes + variables DPE)
          data/processed/batiments_dpe.parquet       (époque de construction par adresse,
                                                      réutilisée par le site)

Source : ADEME, « DPE Logements existants (depuis juillet 2021) », jeu dpe03existant,
         API Data Fair, Licence Ouverte 2.0.

Trois nouvelles variables :
  - dpe_classe        : étiquette énergie du logement vendu (A=1 ... G=7)
  - etage             : étage du logement vendu (si renseigné dans le DPE)
  - annee_construction: époque de l'IMMEUBLE (médiane des DPE de l'adresse)

Rapprochement vente <-> DPE : même adresse (normalisée BAN des deux côtés),
DPE établi dans les 24 mois avant la vente, surface à ±8 %.

Le téléchargement est REPRENABLE (un fichier par département).
Lancer depuis la racine du projet :  python src/03b_dpe.py
"""
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import requests

IN_PATH = Path("data/interim/appartements_geo.parquet")
OUT_PATH = Path("data/interim/appartements_geo_dpe.parquet")
BATIMENTS_OUT = Path("data/processed/batiments_dpe.parquet")
CACHE = Path("data/external/dpe_cache")
API = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant"
DEPARTEMENTS = ["75", "77", "78", "91", "92", "93", "94", "95"]

OBLIGATOIRES = ["date_etablissement_dpe", "etiquette_dpe", "surface_habitable_logement",
                "adresse_ban", "type_batiment", "code_departement_ban"]
OPTIONNELS = ["annee_construction", "periode_construction", "numero_etage_appartement"]

FENETRE_AVANT_JOURS, MARGE_APRES_JOURS = 730, 30
ECART_SURFACE_MAX = 0.08


# --------------------------------------------------------------------------
# 1. TÉLÉCHARGEMENT
# --------------------------------------------------------------------------
def champs_disponibles() -> list[str]:
    """Lit le schéma du jeu de données pour ne demander que des colonnes qui existent."""
    schema = requests.get(API, timeout=60).json()["schema"]
    cles = {c["key"] for c in schema}
    manquants = [c for c in OBLIGATOIRES if c not in cles]
    if manquants:
        proches = sorted(k for k in cles if any(m.split("_")[0] in k for m in manquants))
        raise SystemExit(f"Colonnes introuvables dans l'API ADEME : {manquants}\n"
                         f"Colonnes proches disponibles : {proches}\n"
                         "-> envoie ce message à Claude pour adapter le script.")
    optionnels = [c for c in OPTIONNELS if c in cles]
    print(f"Colonnes optionnelles disponibles : {optionnels}")
    return OBLIGATOIRES + optionnels


def get_avec_relance(url, params=None, essais=6):
    for tentative in range(1, essais + 1):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"    HTTP {r.status_code} (tentative {tentative})")
        except requests.RequestException as e:
            print(f"    Erreur réseau (tentative {tentative}) : {e}")
        time.sleep(5 * tentative)
    raise SystemExit("L'API ADEME ne répond pas. Relance le script plus tard (il reprendra).")


def telecharger_departement(dep: str, champs: list[str]) -> pd.DataFrame:
    fichier = CACHE / f"dpe_{dep}.parquet"
    if fichier.exists():
        return pd.read_parquet(fichier)
    params = {
        "size": 10000,
        "select": ",".join(champs),
        "qs": f'code_departement_ban:"{dep}" AND type_batiment:"appartement"',
    }
    pages, url, n = [], f"{API}/lines", 0
    reponse = get_avec_relance(url, params)
    while True:
        pages.append(pd.DataFrame(reponse.get("results", [])))
        n += len(pages[-1])
        total = reponse.get("total", "?")
        print(f"  {dep} : {n:,} / {total:,} DPE" if isinstance(total, int) else f"  {dep} : {n:,}",
              end="\r")
        suivant = reponse.get("next")
        if not suivant or pages[-1].empty:
            break
        time.sleep(0.2)                     # on reste poli avec le serveur
        reponse = get_avec_relance(suivant)
    print()
    df = pd.concat(pages, ignore_index=True)
    df.to_parquet(fichier, index=False)
    return df


# --------------------------------------------------------------------------
# 2. NETTOYAGE DES DPE
# --------------------------------------------------------------------------
def periode_en_annee(texte) -> float:
    """'1948-1974' -> 1961 ; 'avant 1948' -> 1930 ; 'après 2021' -> 2023."""
    if not isinstance(texte, str):
        return np.nan
    nombres = [int(n) for n in re.findall(r"\d{4}", texte)]
    if not nombres:
        return np.nan
    if "avant" in texte.lower():
        return nombres[0] - 18
    if "apr" in texte.lower():
        return nombres[0] + 2
    return float(np.mean(nombres))


def normaliser_adresse(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def nettoyer_dpe(dpe: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "cle": normaliser_adresse(dpe["adresse_ban"]),
        "date_dpe": pd.to_datetime(dpe["date_etablissement_dpe"], errors="coerce"),
        "surface_dpe": pd.to_numeric(dpe["surface_habitable_logement"], errors="coerce"),
        "dpe_classe": dpe["etiquette_dpe"].map({l: i for i, l in enumerate("ABCDEFG", 1)}),
    })
    annee = pd.to_numeric(dpe["annee_construction"], errors="coerce") \
        if "annee_construction" in dpe else pd.Series(np.nan, index=dpe.index)
    if "periode_construction" in dpe:
        annee = annee.fillna(dpe["periode_construction"].map(periode_en_annee))
    d["annee_construction"] = annee.where(annee.between(1600, 2030))
    if "numero_etage_appartement" in dpe:
        etage = pd.to_numeric(dpe["numero_etage_appartement"], errors="coerce")
        d["etage"] = etage.where(etage.between(-1, 60))
    return d.dropna(subset=["cle", "date_dpe", "surface_dpe"])


# --------------------------------------------------------------------------
# 3. RAPPROCHEMENT VENTE <-> DPE
# --------------------------------------------------------------------------
def rapprocher(ventes: pd.DataFrame, dpe: pd.DataFrame) -> pd.DataFrame:
    cols_dpe = ["cle", "date_dpe", "surface_dpe", "dpe_classe"] + \
               (["etage"] if "etage" in dpe else [])
    cand = ventes[["id_vente", "cle", "date", "surface"]].merge(dpe[cols_dpe], on="cle")
    delai = (cand["date"] - cand["date_dpe"]).dt.days
    cand = cand[(delai <= FENETRE_AVANT_JOURS) & (delai >= -MARGE_APRES_JOURS)]
    cand["ecart"] = (cand["surface_dpe"] - cand["surface"]).abs() / cand["surface"]
    cand = cand[cand["ecart"] <= ECART_SURFACE_MAX]
    # Meilleur candidat : surface la plus proche, puis DPE le plus récent
    meilleur = (cand.sort_values(["ecart", "date_dpe"], ascending=[True, False])
                    .drop_duplicates("id_vente"))
    return meilleur[["id_vente", "dpe_classe"] + (["etage"] if "etage" in dpe else [])]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    champs = champs_disponibles()

    print("Téléchargement des DPE (appartements) par département...")
    dpe = pd.concat([telecharger_departement(dep, champs) for dep in DEPARTEMENTS],
                    ignore_index=True)
    dpe = nettoyer_dpe(dpe)
    print(f"{len(dpe):,} DPE d'appartements exploitables")

    # Époque de construction de chaque IMMEUBLE (toutes les DPE de l'adresse)
    batiments = (dpe.dropna(subset=["annee_construction"])
                    .groupby("cle")["annee_construction"].median().rename("annee_construction")
                    .reset_index())
    BATIMENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    batiments.to_parquet(BATIMENTS_OUT, index=False)

    ventes = pd.read_parquet(IN_PATH).reset_index(drop=True)
    ventes["id_vente"] = np.arange(len(ventes))
    ventes["cle"] = normaliser_adresse(ventes["geo_label"])

    # Rapprochement, département par département (limite la mémoire utilisée)
    morceaux = []
    for dep in DEPARTEMENTS:
        v = ventes[ventes["code_dep"] == dep]
        d = dpe[dpe["cle"].isin(v["cle"].unique())]
        morceaux.append(rapprocher(v, d))
        print(f"  {dep} : {len(morceaux[-1]):,} ventes rapprochées d'un DPE sur {len(v):,}")
    lien = pd.concat(morceaux, ignore_index=True)

    ventes = ventes.merge(lien, on="id_vente", how="left")
    ventes = ventes.merge(batiments, on="cle", how="left")
    ventes = ventes.drop(columns=["id_vente", "cle"])
    ventes.to_parquet(OUT_PATH, index=False)

    # Bilan
    recentes = ventes["date"] >= "2022-01-01"
    print(f"\nVentes avec étiquette DPE : {ventes['dpe_classe'].notna().mean():.1%} "
          f"(depuis 2022 : {ventes.loc[recentes, 'dpe_classe'].notna().mean():.1%})")
    if "etage" in ventes:
        print(f"Ventes avec étage          : {ventes['etage'].notna().mean():.1%}")
    print(f"Ventes avec époque immeuble: {ventes['annee_construction'].notna().mean():.1%}")
    print("\nPrix/m² médian par étiquette DPE :")
    print(ventes.groupby("dpe_classe")["prix_m2"].agg(["size", "median"])
                .rename(index=dict(enumerate("ABCDEFG", 1))).round(0).to_string())
    print(f"\nFichier enregistré : {OUT_PATH}")


if __name__ == "__main__":
    main()
