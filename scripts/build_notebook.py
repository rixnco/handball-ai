"""Genere le notebook local pivot-ai-handball a partir de cellules definies en Python.

Permet de tenir le contenu en code Python lisible plutot que de hand-editer le JSON
.ipynb. Idempotent : peut etre relance.

Usage : python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "pivot_ai.ipynb"


def md(source: str) -> dict:
    """Cree une cellule markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
    }


def code(source: str) -> dict:
    """Cree une cellule code."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
    }


CELLULES = [
    md(
        "# pivot-ai-handball\n"
        "\n"
        "Ouvre ce notebook dans le dépôt avec le noyau Python du projet. Dépose ta vidéo dans `data/raw_clips/`, puis indique son nom dans la cellule de configuration. Le notebook cherche la calibration à côté de la vidéo. Les résultats sont écrits dans `data/outputs/`."
    ),
    md(
        "## 1. Configurer le match"
    ),
    code(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "import pivot_ai\n"
        "from pivot_ai.config import chemin_correspondances_video\n"
        "\n"
        "courant = Path.cwd().resolve()\n"
        "racine = next(\n"
        "    (p for p in (courant, *courant.parents) if (p / \"scripts\" / \"build_notebook.py\").is_file()),\n"
        "    None,\n"
        ")\n"
        "if racine is None:\n"
        "    raise RuntimeError(\"Ouvre le notebook depuis le dépôt pivot-ai-handball.\")\n"
        "\n"
        "CLIPS_DIR = racine / \"data\" / \"raw_clips\"\n"
        "CLIP = str(CLIPS_DIR / \"mon_clip.mp4\")\n"
        "CHEMIN_CORRESPONDANCES = chemin_correspondances_video(CLIP)\n"
        "OUTPUTS_DIR = racine / \"data\" / \"outputs\"\n"
        "OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "print(f\"pivot_ai version : {pivot_ai.__version__}\")\n"
        "print(\"Clips disponibles :\", [p.name for p in sorted(CLIPS_DIR.glob(\"*.mp4\"))])"
    ),
    md("## 2. Verifier GPU"),
    code(
        "import torch\n"
        "print(f\"CUDA dispo : {torch.cuda.is_available()}\")\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"GPU : {torch.cuda.get_device_name(0)}\")\n"
        "    print(f\"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go\")"
    ),
    md(
        "## 3. Choisir un clip et relever les points d'homographie\n"
        "\n"
        "Affiche une frame du clip. **Survole avec la souris** pour relever les pixels des points caracteristiques du terrain.\n"
        "\n"
        "**Points a relever (au minimum 4, idealement 6-8 pour activer RANSAC) :**\n"
        "\n"
        "| Nom | Description | Coords terrain (m) |\n"
        "|-----|-------------|---------------------|\n"
        "| A | Coin terrain haut-gauche (proche bas image) | (0, 0) |\n"
        "| B | Coin terrain haut-droite | (40, 0) |\n"
        "| C | Coin terrain bas-droite (proche haut image) | (40, 20) |\n"
        "| D | Coin terrain bas-gauche | (0, 20) |\n"
        "| E | Ligne mediane cote bas | (20, 0) |\n"
        "| F | Ligne mediane cote haut | (20, 20) |\n"
        "| M_prime | Poteau de but gauche, cote bas | (0, 8.5) |\n"
        "| N_prime | Poteau de but gauche, cote haut | (0, 11.5) |\n"
        "| M | Poteau de but droit, cote bas | (40, 8.5) |\n"
        "| N | Poteau de but droit, cote haut | (40, 11.5) |\n"
        "| K | Sommet demi-cercle 6m gauche | (6, 10) |\n"
        "| L | Sommet demi-cercle 6m droite | (34, 10) |\n"
        "| O_prime | Sommet arc 9m gauche | (9, 10) |\n"
        "| O | Sommet arc 9m droite | (31, 10) |\n"
        "\n"
        "Voir `pivot_ai/config.py` (POINTS_TERRAIN_M) pour la liste complete."
    ),
    code(
        "import cv2\n"
        "import plotly.express as px\n"
        "\n"
        "if not Path(CLIP).is_file():\n"
        "    raise FileNotFoundError(f\"Indique le nom de ton clip dans la cellule de configuration : {CLIP}\")\n"
        "\n"
        "cap = cv2.VideoCapture(CLIP)\n"
        "total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "fps = cap.get(cv2.CAP_PROP_FPS)\n"
        "FRAME_REF = total // 2\n"
        "cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_REF)\n"
        "ret, frame_ref = cap.read()\n"
        "cap.release()\n"
        "if not ret or fps <= 0:\n"
        "    raise RuntimeError(f\"Impossible de lire le clip : {CLIP}\")\n"
        "\n"
        "print(f\"Clip : {total} frames @ {fps:.1f} fps ({total/fps:.1f}s)\")\n"
        "\n"
        "fig = px.imshow(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB))\n"
        "fig.update_layout(width=1300, height=730, dragmode='pan',\n"
        "                  title=f\"Frame {FRAME_REF} — survole pour relever les pixels\")\n"
        "fig.show()"
    ),
    code(
        "import json\n"
        "\n"
        "if not CHEMIN_CORRESPONDANCES.is_file():\n"
        "    raise FileNotFoundError(\n"
        "        f\"Crée {CHEMIN_CORRESPONDANCES} avec les points relevés sur ce clip. \"\n"
        "        \"Utilise python -m scripts.calibrer_terrain avec le chemin du clip depuis la racine du dépôt.\"\n"
        "    )\n"
        "\n"
        "correspondances = json.loads(CHEMIN_CORRESPONDANCES.read_text(encoding=\"utf-8\"))\n"
        "print(f\"Calibration chargée : {len(correspondances)} points\")"
    ),
    md(
        "## 4. Modele de detection\n"
        "\n"
        "Le pipeline exige un modele YOLO entraine sur le handball via\n"
        "`notebooks/train_handball_yolo.ipynb` : il distingue joueurs, gardiens, arbitres et ballon.\n"
        "Si le poids panorama existe dans `models/`, il est sélectionné en priorité ; modifie\n"
        "`CHEMIN_MODELE_HANDBALL` pour choisir un autre fichier."
    ),
    code(
        "from pivot_ai.detection import Detector\n"
        "\n"
        "DOSSIER_MODELES = racine / \"models\"\n"
        "poids_panorama = DOSSIER_MODELES / \"handball_yolo26m_panorama.pt\"\n"
        "poids_standard = DOSSIER_MODELES / \"handball_yolo26m.pt\"\n"
        "CHEMIN_MODELE_HANDBALL = poids_panorama if poids_panorama.is_file() else poids_standard\n"
        "\n"
        "if not CHEMIN_MODELE_HANDBALL.is_file():\n"
        "    raise FileNotFoundError(\n"
        "        f\"Modele handball absent : {CHEMIN_MODELE_HANDBALL}. \"\n"
        "        \"Entraine-le via notebooks/train_handball_yolo.ipynb.\"\n"
        "    )\n"
        "detecteur = Detector(CHEMIN_MODELE_HANDBALL)\n"
        "print(f\"Modele handball : {CHEMIN_MODELE_HANDBALL}\")\n"
        "\n"
        "# Backend de tracking :\n"
        "#  - \"bytetrack\": mouvement seul, rapide. MEILLEUR sur plan large broadcast\n"
        "#    (KPI mesures : moins de reprises d'ID que BoT-SORT sur ce type de source).\n"
        "#  - \"botsort\"  : ReID par apparence. Utile seulement si les joueurs sont assez\n"
        "#    gros a l'image (plan serre) ; sur plan large il degrade (empreintes bruitees).\n"
        "TRACKER = \"bytetrack\"\n"
        "print(f\"Tracker : {TRACKER}\")"
    ),
    md(
        "## 5. Test rapide sur clip court\n"
        "\n"
        "Avant le pipeline complet, on lance une passe rapide pour valider l'enchainement :\n"
        "- subsample agressif (1 frame sur 5)\n"
        "- pas de video radar ni decoupage : on veut juste verifier que la chaine tient debout\n"
        "\n"
        "Sur un clip de 60s a 25fps : ~300 inferences au lieu de 1500."
    ),
    code(
        "from pivot_ai.pipeline import traiter_match_complet\n"
        "import time\n"
        "\n"
        "DOSSIER_TEST = f\"{OUTPUTS_DIR}/_test_rapide\"\n"
        "\n"
        "t0 = time.time()\n"
        "resultat_test = traiter_match_complet(\n"
        "    chemin_video=CLIP,\n"
        "    correspondances_homographie=correspondances,\n"
        "    dossier_sortie=DOSSIER_TEST,\n"
        "    subsample=5,\n"
        "    generer_video_radar=False,\n"
        "    decouper_actions=False,\n"
        "    detecteur=detecteur,\n"
        "    min_frames_track=10,  # ecarte les fragments de tracking courts\n"
        "    tracker=TRACKER,\n"
        ")\n"
        "duree = time.time() - t0\n"
        "\n"
        "meta = resultat_test.metadonnees\n"
        "print(f\"\\nTest rapide OK en {duree:.1f}s (tracker={meta['tracker']})\")\n"
        "print(f\"Pistes totales : {meta['nb_joueurs_avant_filtre']}\")\n"
        "print(f\"--- KPI qualite tracking ---\")\n"
        "print(f\"Reprises d'ID (switches suspectes) : {meta['nb_reprises_id']}\")\n"
        "print(f\"Joueurs estimes (apres recollage)  : {meta['nb_joueurs_estimes']}\")\n"
        "print(f\"Longueur moyenne de piste          : {meta['longueur_track_moyenne']} frames\")\n"
        "print(f\"Joueurs exploitables (apres filtre): {meta['nb_joueurs_stats']}\")\n"
        "print(f\"\\nStats (10 premieres lignes) :\")\n"
        "print(resultat_test.stats_joueurs.head(10))"
    ),
    md(
        "## 6. Pipeline complet\n"
        "\n"
        "Le pipeline traite une image sur deux avec `subsample=2`."
    ),
    code(
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    torch.cuda.empty_cache()\n"
        "\n"
        "t0 = time.time()\n"
        "resultat = traiter_match_complet(\n"
        "    chemin_video=CLIP,\n"
        "    correspondances_homographie=correspondances,\n"
        "    dossier_sortie=OUTPUTS_DIR,\n"
        "    subsample=2,\n"
        "    generer_video_radar=True,\n"
        "    decouper_actions=True,\n"
        "    detecteur=detecteur,\n"
        "    min_frames_track=10,  # ecarte les fragments de tracking courts\n"
        "    tracker=TRACKER,\n"
        ")\n"
        "duree_min = (time.time() - t0) / 60\n"
        "\n"
        "meta = resultat.metadonnees\n"
        "print(f\"\\nPipeline complet termine en {duree_min:.1f} min (tracker={meta['tracker']})\")\n"
        "\n"
        "print(\"\\n=== KPI QUALITE TRACKING ===\")\n"
        "print(f\"Pistes totales                     : {meta['nb_joueurs_avant_filtre']}\")\n"
        "print(f\"Reprises d'ID (switches suspectes) : {meta['nb_reprises_id']}\")\n"
        "print(f\"Joueurs estimes (apres recollage)  : {meta['nb_joueurs_estimes']}\")\n"
        "print(f\"Longueur moyenne de piste          : {meta['longueur_track_moyenne']} frames\")\n"
        "print(f\"Joueurs exploitables (apres filtre): {meta['nb_joueurs_stats']}\")\n"
        "\n"
        "print(\"\\n=== STATS PAR JOUEUR ===\")\n"
        "print(resultat.stats_joueurs)\n"
        "\n"
        "print(f\"\\n=== ACTIONS DETECTEES : {len(resultat.actions_detectees)} ===\")\n"
        "for i, action in enumerate(resultat.actions_detectees):\n"
        "    print(f\"  Action {i+1}: frames {action.frame_debut}-{action.frame_fin} \"\n"
        "          f\"({action.duree_s:.1f}s, {action.nb_joueurs_moyen:.1f} joueurs moyens)\")\n"
        "\n"
        "print(f\"\\n=== CLIPS DECOUPES : {len(resultat.clips_decoupes)} ===\")\n"
        "for clip in resultat.clips_decoupes:\n"
        "    print(f\"  {clip}\")\n"
        "\n"
        "print(f\"\\n=== ARTEFACTS ===\")\n"
        "print(f\"  CSV  : {OUTPUTS_DIR}/stats_joueurs.csv\")\n"
        "print(f\"  Parquet : {OUTPUTS_DIR}/stats_joueurs.parquet\")\n"
        "print(f\"  Video radar SBS : {resultat.chemin_video_radar}\")"
    ),
    md("## 7. Visualiser quelques frames du rendu final"),
    code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "video_radar = resultat.chemin_video_radar\n"
        "cap = cv2.VideoCapture(str(video_radar))\n"
        "total_radar = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "\n"
        "fig, axes = plt.subplots(5, 1, figsize=(20, 22))\n"
        "for ax, idx in zip(axes, [0, total_radar//4, total_radar//2, 3*total_radar//4, total_radar-1]):\n"
        "    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)\n"
        "    ret, frame = cap.read()\n"
        "    if ret:\n"
        "        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\n"
        "        ax.set_title(f\"Frame {idx}/{total_radar}\")\n"
        "        ax.axis('off')\n"
        "cap.release()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 8. Heatmap d'un joueur\n"
        "\n"
        "Visualise la densite de presence d'un joueur sur le terrain. On prend par defaut le tracker le plus present."
    ),
    code(
        "import polars as pl\n"
        "import plotly.express as px\n"
        "from pivot_ai.stats import generer_heatmap_joueur\n"
        "\n"
        "# Tracker le plus present dans le clip\n"
        "stats_tries = resultat.stats_joueurs.sort(\"nb_frames_detectees\", descending=True)\n"
        "tracker_choisi = int(stats_tries[\"tracker_id\"][0])\n"
        "print(f\"Tracker selectionne : {tracker_choisi} \"\n"
        "      f\"({stats_tries['nb_frames_detectees'][0]} frames detectees)\")\n"
        "print(stats_tries.filter(pl.col(\"tracker_id\") == tracker_choisi))\n"
        "\n"
        "# Heatmap reelle a partir des positions terrain stockees dans ResultatPipeline\n"
        "positions_joueur = resultat.positions_par_tracker.get(tracker_choisi, [])\n"
        "heatmap = generer_heatmap_joueur(positions_joueur, resolution=0.5, sigma_lissage=1.5)\n"
        "\n"
        "fig = px.imshow(\n"
        "    heatmap,\n"
        "    origin=\"lower\",\n"
        "    aspect=\"equal\",\n"
        "    color_continuous_scale=\"Hot\",\n"
        "    title=f\"Heatmap presence joueur tracker_id={tracker_choisi}\",\n"
        "    labels={\"x\": \"x_terrain (cellules de 0.5m)\", \"y\": \"y_terrain (cellules de 0.5m)\"},\n"
        ")\n"
        "fig.update_layout(width=900, height=500)\n"
        "fig.show()"
    ),
    md(
        "## 9. Largeur du bloc defensif au cours du temps\n"
        "\n"
        "Charge le DataFrame de la largeur du bloc defensif (MHB ou ADV) et le trace via plotly."
    ),
    code(
        "import polars as pl\n"
        "import plotly.express as px\n"
        "\n"
        "chemin_bloc_mhb = f\"{OUTPUTS_DIR}/largeur_bloc_defensif_mhb.csv\"\n"
        "chemin_bloc_adv = f\"{OUTPUTS_DIR}/largeur_bloc_defensif_adv.csv\"\n"
        "\n"
        "if os.path.exists(chemin_bloc_mhb):\n"
        "    df_mhb = pl.read_csv(chemin_bloc_mhb).with_columns(equipe=pl.lit(\"MHB\"))\n"
        "    df_adv = pl.read_csv(chemin_bloc_adv).with_columns(equipe=pl.lit(\"ADV\"))\n"
        "    df = pl.concat([df_mhb, df_adv])\n"
        "    fig = px.line(\n"
        "        df.to_pandas(),\n"
        "        x=\"temps_s\", y=\"largeur_y_m\", color=\"equipe\",\n"
        "        title=\"Largeur du bloc defensif au cours du temps\",\n"
        "        labels={\"temps_s\": \"Temps (s)\", \"largeur_y_m\": \"Largeur (m)\"},\n"
        "    )\n"
        "    fig.show()\n"
        "else:\n"
        "    print(\"Pas de fichier largeur_bloc_defensif_*.csv : classification d'equipes a peut-etre echoue (trop peu de joueurs ?)\")"
    ),
]


def construire_notebook() -> dict:
    return {
        "cells": CELLULES,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    notebook = construire_notebook()
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Notebook ecrit : {NOTEBOOK_PATH} ({len(CELLULES)} cellules)")


if __name__ == "__main__":
    main()
