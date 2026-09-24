"""
Site d'estimation — interface Streamlit
Lancer depuis la racine du projet :  streamlit run app/app.py
"""
import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from moteur import BASE, DEPARTEMENTS_COUVERTS, NOMS_ZONES, Estimateur, geocoder

# Logo : dépose ton fichier dans app/assets/ sous le nom logo.png (ou .jpg / .svg)
def trouver_logo():
    """Cherche une image dont le nom commence par « logo » (majuscules, espaces et
    double extension tolérés), dans app/assets puis dans assets à la racine."""
    for dossier in [BASE / "assets", BASE.parent / "assets"]:
        if dossier.is_dir():
            for f in sorted(dossier.iterdir()):
                if f.name.lower().startswith("logo") and \
                        f.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
                    return f
    return None


LOGO = trouver_logo()


def icone_onglet():
    """Icône de l'onglet du navigateur : elle doit être CARRÉE.
    Priorité à un fichier app/assets/icone.png s'il existe ; sinon on centre le logo
    dans un carré transparent (au lieu de l'étirer)."""
    from PIL import Image
    icone = BASE / "assets" / "icone.png"
    source = icone if icone.exists() else LOGO
    if source is None or source.suffix.lower() == ".svg":
        return "🏠"
    try:
        img = Image.open(source).convert("RGBA")
        cote = max(img.size)
        carre = Image.new("RGBA", (cote, cote), (0, 0, 0, 0))
        carre.paste(img, ((cote - img.width) // 2, (cote - img.height) // 2))
        return carre
    except Exception:
        return "🏠"
# Nom qui apparaît dans les mentions légales : remplace par ton prénom et ton nom
EDITEUR = "[Prénom Nom], étudiant du programme X-HEC Data Science for Business"

st.set_page_config(page_title="CITIC · Estimation d'appartements en Île-de-France",
                   page_icon=icone_onglet(), layout="wide")

VERT, ORANGE = "#2a9d8f", "#e76f51"


@st.cache_resource(show_spinner="Chargement du modèle…")
def charger_estimateur() -> Estimateur:
    return Estimateur()     # chargé une seule fois, puis gardé en mémoire


def euros(x: float) -> str:
    return f"{x:,.0f} €".replace(",", "\u202f")


est = charger_estimateur()
cfg = est.config
metr = cfg["metriques_test"]["LightGBM"]

# --------------------------------------------------------------------------
# EN-TÊTE
# --------------------------------------------------------------------------
if LOGO:
    st.image(str(LOGO), width=220)       # ajuste la largeur (en pixels) selon ton logo
else:
    st.title("🏠 CITIC")
    # Aide au diagnostic (visible seulement quand tu lances le site sur ton ordinateur)
    contenu = [f.name for f in (BASE / "assets").iterdir()] if (BASE / "assets").is_dir() else None
    print(f"[logo] Aucun logo trouvé. Dossier cherché : {BASE / 'assets'} -> contenu : {contenu}")
st.markdown("#### Estimez le prix d'un appartement en Île-de-France, à partir des ventes réelles")
st.caption(f"Modèle entraîné sur les ventes notariales DVF 2021 → "
           f"{pd.Timestamp(cfg['date_derniere_vente']):%B %Y} · "
           f"erreur médiane de {metr['erreur_mediane_%']} % sur des ventes jamais vues")

onglet_estimer, onglet_marche, onglet_methode = st.tabs(
    ["📍 Estimer un bien", "📈 Le marché", "🔍 La méthode"])


# --------------------------------------------------------------------------
# ONGLET 1 — ESTIMER
# --------------------------------------------------------------------------
def afficher_resultat(choix: dict, r: dict, surface: float, pieces: int, dependances: int):
    st.divider()
    st.markdown(f"**{choix['label']}** · {surface:.0f} m² · {pieces} pièce(s)"
                + (f" · {dependances} cave(s)/parking(s)" if dependances else ""))

    c1, c2, c3 = st.columns(3)
    c1.metric("Prix au m² estimé", euros(r["prix_m2"]))
    c2.metric("Prix total estimé", euros(r["prix_m2"] * surface))
    bas50, haut50 = r["fourchettes"][50]
    bas80, haut80 = r["fourchettes"][80]
    c2.caption(f"Fourchette probable : {euros(bas50 * surface)} à {euros(haut50 * surface)}")
    c3.metric("Fourchette probable du prix au m²", f"{euros(bas50)} – {euros(haut50)}",
              help="Le prix réel se situe dans cette fourchette une fois sur deux.")
    c3.caption(f"Fourchette large (8 chances sur 10) : {euros(bas80)} – {euros(haut80)}/m²")

    st.info("Le modèle ne connaît ni l'état intérieur, ni la vue, ni l'exposition : "
            "ces éléments expliquent l'essentiel de l'écart possible au sein de la fourchette.",
            icon="ℹ️")

    # ---- Explication
    st.subheader("Pourquoi ce prix ?")
    expl = r["explication"].copy()
    expl["sens"] = np.where(expl["effet_%"] >= 0, "Fait monter le prix", "Fait baisser le prix")
    expl["texte"] = expl["effet_%"].map(lambda v: f"{v:+.1f} %")
    graphique = (
        alt.Chart(expl)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("effet_%:Q", title="Effet sur le prix au m² (%)"),
            y=alt.Y("facteur:N", sort=None, title=None),
            color=alt.Color("sens:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=["Fait monter le prix", "Fait baisser le prix"],
                                            range=[VERT, ORANGE])),
            tooltip=[alt.Tooltip("facteur:N", title="Facteur"),
                     alt.Tooltip("texte:N", title="Effet")],
        )
        .properties(height=260)
    )
    st.altair_chart(graphique, width="stretch")
    st.caption(f"Point de départ : prix de référence du modèle, {euros(r['prix_reference'])}/m². "
               "Chaque facteur le fait monter ou baisser ; les effets se multiplient pour donner "
               "l'estimation finale (méthode des valeurs de Shapley).")

    # ---- Ventes comparables
    st.subheader("Les ventes comparables autour du bien")
    voisins = r["voisins"]
    col_carte, col_table = st.columns([3, 2])
    with col_carte:
        carte = folium.Map(location=[choix["lat"], choix["lon"]], zoom_start=16,
                           tiles="OpenStreetMap")
        folium.Marker([choix["lat"], choix["lon"]], tooltip="Votre bien",
                      icon=folium.Icon(color="red", icon="home")).add_to(carte)
        for _, v in voisins.iterrows():
            folium.CircleMarker(
                [v["lat"], v["lon"]], radius=7, color=VERT, fill=True, fill_opacity=0.8,
                tooltip=(f"{v['date']:%m/%Y} · {v['surface']:.0f} m² · "
                         f"{euros(v['prix_m2'])}/m² (vendu)"),
            ).add_to(carte)
        st_folium(carte, height=420, use_container_width=True, returned_objects=[])
    with col_table:
        tableau = pd.DataFrame({
            "Date": voisins["date"].dt.strftime("%m/%Y"),
            "Rue": voisins["adresse"].str.title(),
            "m²": voisins["surface"].round(0).astype(int),
            "Pièces": voisins["nb_pieces"].astype(int),
            "€/m² vendu": voisins["prix_m2"].round(-1).astype(int),
            "€/m² aujourd'hui": voisins["prix_m2_actualise"].round(-1).astype(int),
            "Distance": voisins["distance_m"].round(0).astype(int).astype(str) + " m",
        })
        st.dataframe(tableau, hide_index=True, width="stretch", height=420)
    st.caption("« €/m² aujourd'hui » : prix de vente réactualisé avec l'indice de prix de la zone.")


with onglet_estimer:
    with st.form("recherche"):
        texte = st.text_input("Adresse du bien", placeholder="Ex. : 25 rue de la Roquette, Paris")
        chercher = st.form_submit_button("Rechercher l'adresse")

    if chercher and texte.strip():
        st.session_state.pop("resultat", None)
        try:
            st.session_state["candidats"] = geocoder(texte)
        except Exception:
            st.session_state["candidats"] = None
            st.error("Le service d'adresses ne répond pas pour le moment. Réessayez dans un instant.")

    candidats = st.session_state.get("candidats")
    if candidats is not None:
        if not candidats:
            st.warning("Aucune adresse trouvée. Vérifiez l'orthographe ou ajoutez la commune.")
        else:
            choix = st.selectbox("Choisissez l'adresse exacte", candidats,
                                 format_func=lambda c: c["label"])
            if choix["code_insee"][:2] not in DEPARTEMENTS_COUVERTS:
                st.warning("Cette adresse est hors Île-de-France : le modèle ne couvre "
                           "pour l'instant que la région parisienne.")
            else:
                c1, c2, c3 = st.columns(3)
                surface = c1.number_input("Surface (m²)", min_value=9.0, max_value=400.0,
                                          value=50.0, step=1.0)
                pieces = c2.number_input("Pièces principales", min_value=1, max_value=10, value=2)
                dependances = c3.number_input("Caves / parkings", min_value=0, max_value=5, value=0)

                utilise_dpe = "dpe_classe" in cfg["variables"]
                dpe_classe, etage = None, None
                if utilise_dpe:
                    st.caption("Facultatif — améliore la précision si vous les connaissez :")
                    c4, c5, c6 = st.columns(3)
                    choix_dpe = c4.selectbox("Étiquette énergie (DPE)",
                                             ["Je ne sais pas"] + list("ABCDEFG"))
                    dpe_classe = None if choix_dpe == "Je ne sais pas" else "ABCDEFG".index(choix_dpe) + 1
                    if "etage" in cfg["variables"]:
                        choix_etage = c5.selectbox("Étage", ["Je ne sais pas", "Rez-de-chaussée"]
                                                   + [str(i) for i in range(1, 31)])
                        if choix_etage == "Rez-de-chaussée":
                            etage = 0
                        elif choix_etage != "Je ne sais pas":
                            etage = int(choix_etage)
                    annee = est.annee_immeuble(choix["label"])
                    c6.markdown("**Époque de l'immeuble**")
                    c6.caption(f"Vers {annee:.0f}, d'après les DPE de l'adresse" if annee
                               else "Inconnue pour cette adresse")

                if st.button("Estimer le prix", type="primary"):
                    r = est.estimer(choix["lat"], choix["lon"], choix["code_insee"],
                                    surface, pieces, dependances,
                                    label=choix["label"], dpe_classe=dpe_classe, etage=etage)
                    st.session_state["resultat"] = (choix, r, surface, pieces, dependances)

    if "resultat" in st.session_state:
        afficher_resultat(*st.session_state["resultat"])


# --------------------------------------------------------------------------
# ONGLET 2 — LE MARCHÉ
# --------------------------------------------------------------------------
with onglet_marche:
    st.subheader("Indice des prix à qualité constante")
    st.markdown("Évolution du prix d'un **même type d'appartement au même endroit** "
                "(base 100 = janvier 2021). Calculé à partir des ventes DVF par régression hédonique.")
    indice = est.indice.copy()
    graphique = (
        alt.Chart(indice)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("mois:T", title=None),
            y=alt.Y("indice_base100:Q", title="Indice (base 100)", scale=alt.Scale(zero=False)),
            color=alt.Color("zone:N", title=None, sort=NOMS_ZONES,
                            legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("zone:N"), alt.Tooltip("mois:T", format="%m/%Y"),
                     alt.Tooltip("indice_base100:Q", format=".1f", title="Indice")],
        )
        .properties(height=380)
    )
    st.altair_chart(graphique, width="stretch")

    colonnes = st.columns(3)
    for col, zone in zip(colonnes, NOMS_ZONES):
        serie = indice[indice["zone"] == zone].sort_values("mois")["indice_base100"]
        if len(serie) > 12:
            col.metric(zone, f"{serie.iloc[-1]:.1f}",
                       f"{100 * (serie.iloc[-1] / serie.iloc[-13] - 1):+.1f} % sur 12 mois")


    # ---- Le DPE et les prix
    fichier_dpe = BASE / "data" / "dpe_marche.parquet"
    if fichier_dpe.exists():
        st.divider()
        st.subheader("Le DPE fait-il baisser les prix ?")
        dpe = pd.read_parquet(fichier_dpe)
        zone_dpe = st.radio("Zone", dpe["zone"].unique().tolist(), horizontal=True,
                            key="zone_dpe")
        z = dpe[dpe["zone"] == zone_dpe].sort_values("classe")
        couleurs = alt.Scale(domain=list("ABCDEFG"), range=["#1a9641", "#52b151", "#a6d96a",
                             "#f4e04d", "#fdae61", "#f46d43", "#d7191c"])
        col_brut, col_net = st.columns(2)
        with col_brut:
            st.markdown("**Prix médian brut par étiquette**")
            st.altair_chart(alt.Chart(z).mark_bar().encode(
                x=alt.X("classe:N", title="Étiquette DPE"),
                y=alt.Y("prix_m2_median_brut:Q", title="Prix médian (€/m²)"),
                color=alt.Color("classe:N", scale=couleurs, legend=None),
                tooltip=[alt.Tooltip("classe:N", title="Étiquette"),
                         alt.Tooltip("prix_m2_median_brut:Q", format=",.0f", title="€/m²"),
                         alt.Tooltip("part_ventes_%:Q", format=".1f", title="% des ventes")],
            ).properties(height=300), width="stretch")
            st.caption("Trompeur : les logements F et G sont surtout de petits appartements "
                       "anciens dans les quartiers les plus chers.")
        with col_net:
            st.markdown("**Effet réel, à quartier et taille égaux (par rapport à D)**")
            st.altair_chart(alt.Chart(z).mark_bar().encode(
                x=alt.X("classe:N", title="Étiquette DPE"),
                y=alt.Y("effet_vs_D_%:Q", title="Écart de prix vs D (%)"),
                color=alt.Color("classe:N", scale=couleurs, legend=None),
                tooltip=[alt.Tooltip("classe:N", title="Étiquette"),
                         alt.Tooltip("effet_vs_D_%:Q", format="+.1f", title="Effet (%)")],
            ).properties(height=300), width="stretch")
            st.caption("Chaque vente est comparée aux 30 ventes voisines, en contrôlant la surface.")
        part_fg = z.loc[z["classe"].isin(["F", "G"]), "part_ventes_%"].sum()
        st.markdown(f"Dans cette zone, **{part_fg:.0f} %** des appartements vendus avec un DPE "
                    "sont des passoires thermiques (F ou G). La comparaison des deux graphiques "
                    "illustre le **paradoxe de Simpson** : une corrélation brute peut s'inverser "
                    "une fois les facteurs de confusion pris en compte.")


# --------------------------------------------------------------------------
# ONGLET 3 — LA MÉTHODE
# --------------------------------------------------------------------------
with onglet_methode:
    st.subheader("Comment fonctionne l'estimation")
    st.markdown(f"""
**Les données.** Toutes les ventes d'appartements enregistrées par les notaires en Île-de-France
depuis 2021 (base publique *Demandes de valeurs foncières*), nettoyées puis géolocalisées
à l'adresse près. Les gares et stations viennent de l'open data d'Île-de-France Mobilités.

**Le modèle.** Un modèle de *gradient boosting* (LightGBM) prédit le prix au m² à partir de :
la surface et le nombre de pièces, la localisation fine, la proximité des transports,
l'étiquette énergie, l'étage et l'époque de l'immeuble (issus des DPE de l'ADEME), et surtout **le prix des ventes voisines des 24 derniers mois, réactualisé** au niveau
de prix actuel grâce à un indice calculé à qualité constante.

**La validation.** Le modèle a été entraîné sur les ventes jusqu'à mi-2025 puis testé sur les
6 derniers mois, qu'il n'avait jamais vus — exactement comme lorsqu'il estime votre bien.
""")
    resultats = pd.DataFrame(cfg["metriques_test"]).T.rename(columns={
        "erreur_mediane_%": "Erreur médiane (%)", "erreur_moyenne_%": "Erreur moyenne (%)",
        "part_a_10%_pres": "Ventes à 10 % près (%)", "part_a_20%_pres": "Ventes à 20 % près (%)"})
    st.dataframe(resultats, width="stretch")
    st.markdown(f"""
La *baseline* estime simplement le prix médian des 10 ventes voisines : c'est le score à battre.
Deux fourchettes sont affichées, chacune calibrée sur la période de validation (calibration conforme) :
la **fourchette probable** contient le vrai prix dans **{100 * cfg['fourchettes']['50']['couverture']:.1f} %**
des ventes du test (objectif 50 %), la **fourchette large** dans
**{100 * cfg['fourchettes']['80']['couverture']:.1f} %** (objectif 80 %).
""")
    image = BASE / "assets" / "importance_variables.png"
    if image.exists():
        st.image(str(image), caption="Importance de chaque variable dans le modèle")
    st.markdown("""
**Les limites.** Les ventes notariales ne décrivent ni l'état intérieur, ni l'exposition, ni la vue,
et les informations DPE ne sont disponibles que pour une partie des ventes : deux appartements
voisins de même surface peuvent donc se vendre à des prix très différents. L'estimation est un ordre de grandeur, pas une expertise.
""")


# --------------------------------------------------------------------------
# PIED DE PAGE — AVERTISSEMENT ET MENTIONS LÉGALES (visibles sur tous les onglets)
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ **Projet étudiant à visée pédagogique.** Les estimations sont produites automatiquement "
    "par un modèle statistique et peuvent être erronées. Elles ne constituent ni une expertise, "
    "ni un avis de valeur, ni un conseil, et ne doivent pas servir de base à une décision "
    "d'achat, de vente, de prêt ou de fiscalité."
)
with st.expander("Mentions légales"):
    st.markdown(f"""
**Éditeur.** Ce site est un projet étudiant personnel, non commercial, réalisé par {EDITEUR}.
Il n'est affilié à aucune administration, aucun notaire, ni aucun professionnel de l'immobilier.

**Nature des informations.** Les prix affichés sont des estimations statistiques indicatives,
calculées à partir de ventes passées. Ils ne reflètent pas la valeur réelle d'un bien donné,
qui dépend de nombreux éléments inconnus du modèle (étage, état, exposition, performance
énergétique, travaux, etc.). Aucune garantie n'est donnée quant à l'exactitude, l'exhaustivité
ou l'actualité des informations. L'éditeur ne s'engage pas à ce que les informations soient
exactes et décline toute responsabilité quant à l'usage qui en est fait. Pour connaître la valeur
d'un bien, adressez-vous à un professionnel (notaire, agent immobilier, expert).

**Sources des données.**
- Ventes : base *Demandes de valeurs foncières* (DVF), Direction générale des Finances publiques,
  Licence Ouverte 2.0. Les données sont utilisées à des fins statistiques ; leur réutilisation
  ne doit pas permettre la réidentification des personnes.
- Géocodage des adresses : service de géocodage de la Géoplateforme (IGN).
- Gares et stations : Île-de-France Mobilités, Licence Ouverte 2.0.
- Diagnostics de performance énergétique : ADEME, jeu « DPE Logements existants », Licence Ouverte 2.0.
- Fond de carte : © les contributeurs d'OpenStreetMap.

**Données personnelles.** Le site ne crée pas de compte et n'enregistre pas les recherches.
L'adresse saisie est transmise au service de géocodage de l'IGN pour être localisée.

**Hébergement.** Streamlit Community Cloud (Snowflake Inc.).
""")
