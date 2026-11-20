"""Genere le notebook d'entrainement du modele YOLO handball.

Fine-tune un YOLO26 sur un export Roboflow Handball Detection
(4 classes players/goalkeeper/referees/ball) sur GPU, puis exporte
les poids pour usage dans le pipeline d'inference.

Usage : python scripts/build_train_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "train_handball_yolo.ipynb"


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
        "# Entrainement modele YOLO handball\n"
        "\n"
        "Fine-tune un YOLO26 sur un export **Handball Detection** (Roboflow Universe) :\n"
        "4 classes `players` / `goalkeeper` / `referees` / `ball`.\n"
        "\n"
        "Objectif : remplacer le YOLO26m COCO generique (qui detecte des 'person' et fragmente\n"
        "le tracking) par un detecteur specialise handball qui distingue joueurs, gardiens,\n"
        "arbitres et ballon.\n"
        "\n"
        "**Sortie** : `best.pt` sauvegardé dans `models/`, a passer ensuite a\n"
        "`ModeleConfig.pour_handball(...)` dans le notebook d'inference.\n"
        "\n"
        "**Prerequis** : GPU CUDA."
    ),
    md(
        "## 1. Préparer l’environnement\n"
        "\n"
        "Dans le dépôt local ou le devcontainer, utilise l’environnement Python du projet.\n"
        "\n"
        "Pour télécharger un export absent, définis `ROBOFLOW_API_KEY` dans ton environnement. Les exports déjà présents sont utilisables sans clé."
    ),
    code(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "os.environ.setdefault(\"YOLO_CONFIG_DIR\", \"/tmp\")\n"
        "os.environ.setdefault(\"MPLCONFIGDIR\", \"/tmp/matplotlib\")\n"
        "\n"
        "import torch\n"
        "\n"
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError(\"Active un GPU CUDA dans ton environnement.\")\n"
        "print(f\"GPU : {torch.cuda.get_device_name(0)}\")\n"
        "print(f\"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go\")"
    ),
    md(
        "## 2. Sélectionner le dataset\n"
        "\n"
        "`Handball-Detection-1` est l’export utilisé pour le fine-tuning panorama, sous licence CC BY 4.0.\n"
        "\n"
        "- [Handball-Detection-1, espace erics-workspace-it3ft](https://universe.roboflow.com/erics-workspace-it3ft/handball-detection-fj8rc-omtda/dataset/3)\n"
        "\n"
        "La provenance et les instructions d’installation manuelle figurent dans `data/datasets/README.md`."
    ),
    code(
        "from getpass import getpass\n"
        "\n"
        "EXPORTS = {\n"
        "    \"Handball-Detection-1\": (\"erics-workspace-it3ft\", \"handball-detection-fj8rc-omtda\", 3),\n"
        "}\n"
        "EXPORT = \"Handball-Detection-1\"\n"
        "\n"
        "courant = Path.cwd().resolve()\n"
        "racine = next(\n"
        "    (p for p in (courant, *courant.parents)\n"
        "     if (p / \"scripts\" / \"build_train_notebook.py\").is_file()),\n"
        "    courant,\n"
        ")\n"
        "dossier_dataset = racine / \"data\" / \"datasets\" / EXPORT\n"
        "if not (dossier_dataset / \"data.yaml\").is_file():\n"
        "    try:\n"
        "        from roboflow import Roboflow\n"
        "    except ImportError as exc:\n"
        "        raise RuntimeError(\"Installe roboflow pour télécharger un export absent.\") from exc\n"
        "\n"
        "    cle = os.environ.get(\"ROBOFLOW_API_KEY\")\n"
        "    if not cle:\n"
        "        cle = getpass(\"Clé API Roboflow : \")\n"
        "    workspace, projet, version = EXPORTS[EXPORT]\n"
        "    rf = Roboflow(api_key=cle)\n"
        "    dossier_dataset.parent.mkdir(parents=True, exist_ok=True)\n"
        "    rf.workspace(workspace).project(projet).version(version).download(\n"
        "        \"yolov8\", location=str(dossier_dataset)\n"
        "    )\n"
        "\n"
        "print(f\"Dataset : {dossier_dataset}\")"
    ),
    md(
        "## 3. Inspecter le dataset"
    ),
    code(
        "from tempfile import TemporaryDirectory\n"
        "\n"
        "import yaml\n"
        "\n"
        "cfg = yaml.safe_load((dossier_dataset / \"data.yaml\").read_text(encoding=\"utf-8\"))\n"
        "cfg[\"path\"] = str(dossier_dataset)\n"
        "for cle_split, split in ((\"train\", \"train\"), (\"val\", \"valid\"), (\"test\", \"test\")):\n"
        "    images = dossier_dataset / split / \"images\"\n"
        "    if not images.is_dir():\n"
        "        raise FileNotFoundError(f\"Dossier d’images manquant : {images}\")\n"
        "    cfg[cle_split] = f\"{split}/images\"\n"
        "    print(f\"{split:6s} : {sum(p.is_file() for p in images.iterdir())} images\")\n"
        "\n"
        "# Ultralytics reçoit un YAML utilisable depuis n’importe quel répertoire.\n"
        "config_temporaire = TemporaryDirectory()\n"
        "data_yaml = str(Path(config_temporaire.name) / \"data.yaml\")\n"
        "Path(data_yaml).write_text(yaml.safe_dump(cfg), encoding=\"utf-8\")\n"
        "print(\"Classes (ordre du dataset) :\", cfg[\"names\"])"
    ),
    md(
        "## 4. Entraîner le modèle\n"
        "\n"
        "Le notebook reprend les poids `models/handball_<YOLO_BASE>.pt` s’ils existent et entraîne à 1280 px avec un batch automatique. Il utilise les poids COCO de `YOLO_BASE` (YOLO26m par défaut) si aucun poids local n’est disponible."
    ),
    code(
        "from ultralytics import YOLO\n"
        "\n"
        "YOLO_BASE = \"yolo26m\"\n"
        "poids_locaux = racine / \"models\" / (\"handball_\"+YOLO_BASE+\".pt\")\n"
        "MODELE_BASE = str(poids_locaux) if poids_locaux.is_file() else YOLO_BASE+\".pt\"\n"
        "EPOCHS = 50\n"
        "IMGSZ = 1280\n"
        "BATCH = -1\n"
        "\n"
        "model = YOLO(MODELE_BASE)\n"
        "resultats = model.train(\n"
        "    data=data_yaml,\n"
        "    epochs=EPOCHS,\n"
        "    imgsz=IMGSZ,\n"
        "    batch=BATCH,\n"
        "    device=0,\n"
        "    workers=8,\n"
        "    patience=15,\n"
        "    name=\"handball_\"+YOLO_BASE,\n"
        "    project=str(racine / \"data\" / \"training\"),\n"
        "    verbose=True,\n"
        ")\n"
        "print(f\"Meilleurs poids : {model.trainer.best}\")"
    ),
    md("## 5. Metriques de validation"),
    code(
        "metrics = model.val(data=data_yaml)\n"
        "print(f\"mAP50-95 : {metrics.box.map:.4f}\")\n"
        "print(f\"mAP50    : {metrics.box.map50:.4f}\")\n"
        "print(f\"mAP75    : {metrics.box.map75:.4f}\")\n"
        "print(\"\\nmAP50 par classe :\")\n"
        "for i, nom in enumerate(cfg.get(\"names\", [])):\n"
        "    try:\n"
        "        print(f\"  {nom:12s} : {metrics.box.ap50[i]:.4f}\")\n"
        "    except (IndexError, TypeError):\n"
        "        pass"
    ),
    md(
        "## 6. Exporter les poids"
    ),
    code(
        "import shutil\n"
        "\n"
        "dossier_modeles = racine / \"models\"\n"
        "dossier_modeles.mkdir(parents=True, exist_ok=True)\n"
        "chemin_cible = dossier_modeles / (\"handball_\"+YOLO_BASE+\".pt\")\n"
        "shutil.copy2(model.trainer.best, chemin_cible)\n"
        "print(f\"Modèle enregistré : {chemin_cible}\")"
    ),
    md("## 7. Verification visuelle sur quelques images de validation"),
    code(
        "import matplotlib.pyplot as plt\n"
        "import cv2\n"
        "\n"
        "images_val = [str(p) for p in sorted((dossier_dataset / \"valid\" / \"images\").iterdir()) if p.is_file()]\n"
        "echantillon = images_val[:6] if len(images_val) >= 6 else images_val\n"
        "\n"
        "if echantillon:\n"
        "    preds = model.predict(echantillon, conf=0.35, verbose=False)\n"
        "    fig, axes = plt.subplots(2, 3, figsize=(20, 10))\n"
        "    for ax, pred in zip(axes.flat, preds):\n"
        "        annotee = pred.plot()  # BGR annote\n"
        "        ax.imshow(cv2.cvtColor(annotee, cv2.COLOR_BGR2RGB))\n"
        "        ax.axis('off')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print(\"Pas d'images de validation trouvees.\")"
    ),
    md(
        "## Suite : utiliser le modele dans le pipeline\n"
        "\n"
        "Dans le notebook d’inférence, utilise le chemin affiché à l’export pour pointer\n"
        "vers le modele handball :\n"
        "\n"
        "```python\n"
        "from pivot_ai.config import ModeleConfig\n"
        "\n"
        "config_handball = ModeleConfig.pour_handball(\n"
        "    \"CHEMIN_VERS_LE_MODELE_EXPORTE\"\n"
        ")\n"
        "resultat = traiter_match_complet(\n"
        "    chemin_video=CLIP,\n"
        "    correspondances_homographie=correspondances,\n"
        "    dossier_sortie=OUTPUTS_DIR,\n"
        "    subsample=2,\n"
        "    modele_config=config_handball,\n"
        ")\n"
        "```\n"
        "\n"
        "Le pipeline detectera alors joueurs / gardiens / arbitres / ballon distinctement,\n"
        "ce qui reduit fortement la fragmentation du tracking."
    ),
]


def construire_notebook() -> dict:
    return {
        "cells": CELLULES,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv (3.11.18)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.18"},
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
