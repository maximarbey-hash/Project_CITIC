"""
Moteur de l'application : toute la logique « métier », sans interface.
  - geocoder()   : adresse tapée -> liste d'adresses candidates avec coordonnées
  - Estimateur   : calcule les variables d'un bien, prédit le prix, explique l'estimation

Séparer le moteur de l'interface (app.py) permet de le tester sans lancer le site.
"""
from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import requests
from sklearn.neighbors import BallTree

BASE = Path(__file__).parent
RAYON_TERRE = 6_371_000
CENTRE_PARIS = (48.8534, 2.3488)
ZONES = {"75": 0, "92": 1, "93": 1, "94": 1}
NOMS_ZONES = ["Paris", "Petite couronne", "Grande couronne"]
DEPARTEMENTS_COUVERTS = {"75", "77", "78", "91", "92", "93", "94", "95"}
URL_GEOCODAGE = "https://data.geopf.fr/geocodage/search"

# Regroupement des variables pour une explication lisible
GROUPES = {
    "Prix des ventes voisines": ["log_knn10", "log_knn30", "knn30_rayon_m"],
    "Situation géographique": ["lat", "lon", "dep", "zone", "dist_centre_paris_m"],
    "Transports": ["dist_gare_m", "dist_metro_rer_m", "nb_gares_1km"],
    "Surface et pièces": ["surface", "nb_pieces", "surface_par_piece"],
    "Caves et parkings": ["nb_dependances"],
    "Évolution du marché": ["t_mois"],
    "Performance énergétique (DPE)": ["dpe_classe"],
    "Étage": ["etage"],
    "Époque de l'immeuble": ["annee_construction"],
}


def geocoder(adresse: str, limite: int = 5) -> list[dict]:
    """Interroge l'API de géocodage de la Géoplateforme (IGN)."""
    r = requests.get(URL_GEOCODAGE, params={"q": adresse, "limit": limite, "index": "address"},
                     timeout=10)
    r.raise_for_status()
    resultats = []
    for f in r.json().get("features", []):
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        resultats.append({
            "label": p.get("label", ""), "lat": lat, "lon": lon,
            "code_insee": p.get("citycode", ""), "score": p.get("score", 0),
            "type": p.get("type", ""),
        })
    return resultats


def normaliser_adresse(texte: str) -> str:
    """Même normalisation que l'étape 3b, pour retrouver l'immeuble dans la table DPE."""
    return " ".join(str(texte).lower().split())


def _radians(lat, lon) -> np.ndarray:
    return np.radians(np.column_stack([np.atleast_1d(lat), np.atleast_1d(lon)]))


def _distance_m(lat, lon, lat0, lon0) -> float:
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return float(2 * RAYON_TERRE * np.arcsin(np.sqrt(a)))


class Estimateur:
    def __init__(self, base: Path = BASE):
        self.config = json.loads((base / "models" / "config.json").read_text(encoding="utf-8"))
        noms = ["median"] + [f"q{round(100 * q)}" for f in self.config["fourchettes"].values()
                             for q in f["quantiles"]]
        self.modeles = {nom: self._charger_modele(base / "models" / f"lgbm_{nom}.txt")
                        for nom in noms}
        self.ventes = pd.read_parquet(base / "data" / "ventes_recentes.parquet")
        self.indice = pd.read_parquet(base / "data" / "indice_prix.parquet")
        gares = pd.read_parquet(base / "data" / "gares.parquet")
        fichier_bat = base / "data" / "batiments_dpe.parquet"
        self.batiments = (pd.read_parquet(fichier_bat).set_index("cle")["annee_construction"]
                          if fichier_bat.exists() else pd.Series(dtype=float))

        # Index spatiaux construits une seule fois (recherche des plus proches voisins)
        self.arbre_ventes = BallTree(_radians(self.ventes["lat"], self.ventes["lon"]),
                                     metric="haversine")
        self.arbre_gares = BallTree(_radians(gares["lat"], gares["lon"]), metric="haversine")
        lourdes = gares[gares["lourd"]]
        self.arbre_lourdes = BallTree(_radians(lourdes["lat"], lourdes["lon"]), metric="haversine")

    @staticmethod
    def _charger_modele(chemin: Path) -> lgb.Booster:
        """Charge la version compressée (.txt.gz) si elle existe, sinon le .txt."""
        compresse = chemin.with_name(chemin.name + ".gz")
        if compresse.exists():
            with gzip.open(compresse, "rt", encoding="utf-8") as f:
                return lgb.Booster(model_str=f.read())
        return lgb.Booster(model_file=str(chemin))

    # ------------------------------------------------------------------
    def annee_immeuble(self, label: str):
        """Époque de construction de l'immeuble d'après les DPE de l'adresse (ou None)."""
        annee = self.batiments.get(normaliser_adresse(label))
        return None if annee is None or pd.isna(annee) else float(annee)

    def variables(self, lat, lon, code_insee, surface, nb_pieces, nb_dependances,
                  label="", dpe_classe=None, etage=None):
        """Construit exactement les mêmes variables qu'à l'entraînement.
        dpe_classe (1=A ... 7=G) et etage sont facultatifs : None = inconnu."""
        pt = _radians(lat, lon)
        code_dep = code_insee[:2]

        dist_gare = self.arbre_gares.query(pt, k=1)[0][0, 0] * RAYON_TERRE
        dist_lourde = self.arbre_lourdes.query(pt, k=1)[0][0, 0] * RAYON_TERRE
        nb_gares = int(self.arbre_gares.query_radius(pt, r=1000 / RAYON_TERRE, count_only=True)[0])

        dist, idx = self.arbre_ventes.query(pt, k=30)
        voisins = self.ventes.iloc[idx[0]].copy()
        voisins["distance_m"] = dist[0] * RAYON_TERRE
        log_prix = np.log(voisins["prix_m2_actualise"].to_numpy())

        ligne = {
            "surface": surface, "nb_pieces": nb_pieces,
            "surface_par_piece": surface / nb_pieces, "nb_dependances": nb_dependances,
            "t_mois": self.config["t_mois_max"],          # on estime au niveau de prix actuel
            "lat": lat, "lon": lon,
            "dep": int(code_dep.replace("2A", "201").replace("2B", "202")),
            "zone": ZONES.get(code_dep, 2),
            "dist_centre_paris_m": _distance_m(lat, lon, *CENTRE_PARIS),
            "dist_gare_m": dist_gare, "dist_metro_rer_m": dist_lourde, "nb_gares_1km": nb_gares,
            "log_knn10": float(np.median(log_prix[:10])),
            "log_knn30": float(np.median(log_prix)),
            "knn30_rayon_m": float(voisins["distance_m"].iloc[-1]),
            # Inconnu -> NaN : LightGBM sait traiter les valeurs manquantes
            "dpe_classe": np.nan if dpe_classe is None else dpe_classe,
            "etage": np.nan if etage is None else etage,
            "annee_construction": self.annee_immeuble(label) or np.nan,
        }
        X = pd.DataFrame([ligne])[self.config["variables"]]
        return X, voisins

    # ------------------------------------------------------------------
    def estimer(self, lat, lon, code_insee, surface, nb_pieces, nb_dependances,
                label="", dpe_classe=None, etage=None) -> dict:
        X, voisins = self.variables(lat, lon, code_insee, surface, nb_pieces, nb_dependances,
                                    label, dpe_classe, etage)
        log_central = self.modeles["median"].predict(X)[0]
        fourchettes = {}
        for niveau, f in self.config["fourchettes"].items():
            q_bas, q_haut = (f"q{round(100 * q)}" for q in f["quantiles"])
            log_bas = self.modeles[q_bas].predict(X)[0] - f["correction_log"]
            log_haut = self.modeles[q_haut].predict(X)[0] + f["correction_log"]
            # Sécurité : la fourchette doit encadrer l'estimation centrale
            fourchettes[int(niveau)] = (float(np.exp(min(log_bas, log_central))),
                                        float(np.exp(max(log_haut, log_central))))

        # Explication : contributions de chaque variable (valeurs de Shapley, calculées
        # nativement par LightGBM). En log, les contributions s'additionnent ;
        # en prix, elles se multiplient -> on les exprime en % d'effet.
        contrib = self.modeles["median"].predict(X, pred_contrib=True)[0]
        par_variable = dict(zip(self.config["variables"], contrib[:-1]))
        explication = pd.DataFrame([
            {"facteur": groupe, "effet_log": sum(par_variable[v] for v in variables
                                                  if v in par_variable)}
            for groupe, variables in GROUPES.items()
            if any(v in par_variable for v in variables)      # groupes présents dans le modèle
        ])
        explication["effet_%"] = 100 * (np.exp(explication["effet_log"]) - 1)

        return {
            "prix_m2": float(np.exp(log_central)),
            "fourchettes": fourchettes,            # {50: (bas, haut), 80: (bas, haut)}
            "prix_reference": float(np.exp(contrib[-1])),   # point de départ du modèle
            "explication": explication.sort_values("effet_log", key=abs, ascending=False),
            "voisins": voisins.head(10),
            "variables": X.iloc[0].to_dict(),
        }
