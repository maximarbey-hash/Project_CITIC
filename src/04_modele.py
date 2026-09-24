"""
ÉTAPE 5 — Modélisation
======================
Entrée  : data/processed/dataset.parquet
Sorties : models/lgbm_median.txt + lgbm_q10/q25/q75/q90.txt (bornes des fourchettes)
          models/config.json            (liste des variables, métriques, dernier mois...)
          reports/importance_variables.png

Démarche :
  1. Découpage PAR DATE : train (passé) / validation / test (6 derniers mois)
  2. Baseline naïve : prix/m² médian des 10 ventes voisines -> le score à battre
  3. LightGBM (cible = log du prix/m²), arrêt précoce sur la validation
  4. Deux fourchettes, chacune calibrée sur la validation pour tenir sa promesse :
     - resserrée à 50 % (quantiles 25 % et 75 %)
     - large à 80 %     (quantiles 10 % et 90 %)
  5. Évaluation honnête sur le test, puis réentraînement final sur TOUTES les données

Lancer depuis la racine du projet :  python src/04_modele.py
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")                 # génère les images sans ouvrir de fenêtre
import matplotlib.pyplot as plt

# Avertissement sans conséquence des versions récentes de LightGBM
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

IN_PATH = Path("data/processed/dataset.parquet")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MOIS_TEST, MOIS_VALID = 6, 6
# niveau de confiance -> (quantile bas, quantile haut)
FOURCHETTES = {50: (0.25, 0.75), 80: (0.10, 0.90)}

# Variables utilisées : uniquement des infos que l'utilisateur du site pourra fournir
# (adresse -> tout le géographique ; surface, pièces, dépendances ; date = aujourd'hui)
VARIABLES = [
    "surface", "nb_pieces", "surface_par_piece", "nb_dependances",
    "t_mois",
    "lat", "lon", "dep", "zone", "dist_centre_paris_m",
    "dist_gare_m", "dist_metro_rer_m", "nb_gares_1km",
    "log_knn10", "log_knn30", "knn30_rayon_m",
]
# Variables issues des DPE (étape 3b) : ajoutées seulement si elles existent
VARIABLES_DPE = ["dpe_classe", "etage", "annee_construction"]
CATEGORIELLES = ["dep", "zone"]

PARAMS = dict(
    learning_rate=0.05, num_leaves=127, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, n_estimators=5000, verbose=-1,
)


# --------------------------------------------------------------------------
# 1. PRÉPARATION
# --------------------------------------------------------------------------
def preparer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dep"] = df["code_dep"].replace({"2A": "201", "2B": "202"}).astype(int)
    df["log_knn10"] = np.log(df["knn10_prix_m2"])
    df["log_knn30"] = np.log(df["knn30_prix_m2"])
    df["y"] = np.log(df["prix_m2"])       # on prédit le LOG du prix/m²
    return df


def decouper(df: pd.DataFrame):
    """Découpage chronologique : on apprend sur le passé, on teste sur le futur."""
    dernier = df["mois_idx"].max()
    debut_test = dernier - MOIS_TEST + 1
    debut_valid = debut_test - MOIS_VALID
    train = df[df["mois_idx"] < debut_valid]
    valid = df[(df["mois_idx"] >= debut_valid) & (df["mois_idx"] < debut_test)]
    test = df[df["mois_idx"] >= debut_test]
    for nom, d in [("train", train), ("validation", valid), ("test", test)]:
        print(f"  {nom:<11}: {len(d):>8,} ventes  "
              f"({d['date'].min():%m/%Y} -> {d['date'].max():%m/%Y})")
    return train, valid, test


# --------------------------------------------------------------------------
# 2. MÉTRIQUES — exprimées en % d'erreur, lisibles par tout le monde
# --------------------------------------------------------------------------
def metriques(vrai_prix: np.ndarray, prix_estime: np.ndarray) -> dict:
    erreur = np.abs(prix_estime - vrai_prix) / vrai_prix
    return {
        "erreur_mediane_%": round(100 * np.median(erreur), 1),
        "erreur_moyenne_%": round(100 * np.mean(erreur), 1),
        "part_a_10%_pres": round(100 * np.mean(erreur <= 0.10), 1),
        "part_a_20%_pres": round(100 * np.mean(erreur <= 0.20), 1),
    }


# --------------------------------------------------------------------------
# 3. ENTRAÎNEMENT
# --------------------------------------------------------------------------
def entrainer(X, y, X_val=None, y_val=None, n_estimators=None, **objectif):
    params = {**PARAMS, **objectif}
    if n_estimators:
        params["n_estimators"] = n_estimators
    modele = lgb.LGBMRegressor(**params)
    if X_val is not None:
        modele.fit(X, y, eval_set=[(X_val, y_val)], categorical_feature=CATEGORIELLES,
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        modele.fit(X, y, categorical_feature=CATEGORIELLES)
    return modele


def main():
    df = preparer(pd.read_parquet(IN_PATH))
    global VARIABLES
    VARIABLES = VARIABLES + [v for v in VARIABLES_DPE if v in df.columns]
    print(f"{len(VARIABLES)} variables : {', '.join(VARIABLES)}")
    print("Découpage chronologique :")
    train, valid, test = decouper(df)
    X_tr, X_va, X_te = train[VARIABLES], valid[VARIABLES], test[VARIABLES]
    vrai = test["prix_m2"].to_numpy()

    resultats = {}

    # --- Baseline : médiane des 10 ventes voisines, sans aucun modèle
    base = test["knn10_prix_m2"].fillna(test["knn30_prix_m2"]).fillna(df["prix_m2"].median())
    resultats["Baseline (médiane 10 voisins)"] = metriques(vrai, base.to_numpy())

    # --- LightGBM médian (objectif L1 sur le log = erreur relative robuste)
    print("\nEntraînement LightGBM (peut prendre quelques minutes)...")
    m_med = entrainer(X_tr, train["y"], X_va, valid["y"], objective="l1")
    n_arbres = m_med.best_iteration_
    print(f"  meilleur nombre d'arbres : {n_arbres}")
    pred = np.exp(m_med.predict(X_te))
    resultats["LightGBM"] = metriques(vrai, pred)

    # --- Fourchette : quantiles 10 % et 90 %
    # Calibration « conformelle » : on mesure sur la VALIDATION de combien il faut
    # élargir (ou resserrer) chaque fourchette pour qu'elle tienne sa promesse.
    print("Entraînement des modèles de fourchette...")
    y_va = valid["y"].to_numpy()
    fourchettes = {}
    for niveau, (q_bas, q_haut) in FOURCHETTES.items():
        m_bas = entrainer(X_tr, train["y"], n_estimators=n_arbres, objective="quantile", alpha=q_bas)
        m_haut = entrainer(X_tr, train["y"], n_estimators=n_arbres, objective="quantile", alpha=q_haut)
        ecarts = np.maximum(m_bas.predict(X_va) - y_va, y_va - m_haut.predict(X_va))
        n = len(ecarts)
        correction = float(np.quantile(ecarts, min(1.0, niveau / 100 * (n + 1) / n)))
        bas = np.exp(m_bas.predict(X_te) - correction)
        haut = np.exp(m_haut.predict(X_te) + correction)
        fourchettes[str(niveau)] = {
            "quantiles": [q_bas, q_haut],
            "correction_log": round(correction, 4),
            "couverture": round(float(np.mean((vrai >= bas) & (vrai <= haut))), 3),
            "demi_largeur_mediane_%": round(float(50 * np.median((haut - bas) / pred)), 1),
        }

    # --- Affichage des résultats
    print("\n=== RÉSULTATS SUR LE TEST (6 derniers mois, jamais vus) ===")
    print(pd.DataFrame(resultats).T.to_string())
    print()
    for niveau, f in fourchettes.items():
        print(f"Fourchette à {niveau} % : vrai prix dedans dans {100 * f['couverture']:.1f} % des cas "
              f"(objectif {niveau} %), largeur médiane ±{f['demi_largeur_mediane_%']:.0f} %")

    print("\nErreur médiane par zone (LightGBM) :")
    noms_zones = {0: "Paris", 1: "Petite couronne", 2: "Grande couronne"}
    for z, nom in noms_zones.items():
        masque = (test["zone"] == z).to_numpy()
        if masque.any():
            m = metriques(vrai[masque], pred[masque])
            print(f"  {nom:<16}: {m['erreur_mediane_%']:>4} %  "
                  f"({m['part_a_10%_pres']} % des ventes à 10 % près)")

    if "dpe_classe" in test:
        avec = test["dpe_classe"].notna().to_numpy()
        print("\nErreur médiane selon la disponibilité du DPE (LightGBM) :")
        for nom, masque in [("ventes AVEC DPE", avec), ("ventes SANS DPE", ~avec)]:
            if masque.any():
                m = metriques(vrai[masque], pred[masque])
                print(f"  {nom:<16}: {m['erreur_mediane_%']:>4} %  ({masque.sum():,} ventes)")

    # --- Importance des variables (gain = amélioration apportée par la variable)
    REPORTS_DIR.mkdir(exist_ok=True)
    imp = pd.Series(m_med.booster_.feature_importance("gain"), index=VARIABLES)
    imp = (100 * imp / imp.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax, color="#2a6f97")
    ax.set_xlabel("Part de l'importance totale (%)")
    ax.set_title("Ce qui compte le plus dans le prix au m²")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "importance_variables.png", dpi=150)
    print(f"\nGraphique enregistré : {REPORTS_DIR / 'importance_variables.png'}")

    # --- Modèles FINAUX : réentraînés sur toutes les données (pour le site)
    print("\nRéentraînement final sur toutes les données...")
    X_all, y_all = df[VARIABLES], df["y"]
    n_final = int(n_arbres * 1.1)   # un peu plus d'arbres car un peu plus de données
    MODELS_DIR.mkdir(exist_ok=True)
    objectifs = {"median": dict(objective="l1")}
    for q_bas, q_haut in FOURCHETTES.values():
        for q in (q_bas, q_haut):
            objectifs[f"q{round(100 * q)}"] = dict(objective="quantile", alpha=q)
    for nom, objectif in objectifs.items():
        m = entrainer(X_all, y_all, n_estimators=n_final, **objectif)
        m.booster_.save_model(MODELS_DIR / f"lgbm_{nom}.txt")

    config = {
        "variables": VARIABLES,
        "categorielles": CATEGORIELLES,
        "t_mois_max": int(df["t_mois"].max()),
        "date_derniere_vente": f"{df['date'].max():%Y-%m-%d}",
        "n_arbres": n_final,
        "metriques_test": resultats,
        "fourchettes": fourchettes,
    }
    (MODELS_DIR / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    print(f"Modèles et configuration enregistrés dans {MODELS_DIR}/")


if __name__ == "__main__":
    main()
