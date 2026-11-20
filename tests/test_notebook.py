"""Tests de validation des notebooks locaux et de leurs builders."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

NOTEBOOK_INFERENCE = NOTEBOOKS_DIR / "pivot_ai.ipynb"
NOTEBOOK_TRAIN = NOTEBOOKS_DIR / "train_handball_yolo.ipynb"

TOUS_NOTEBOOKS = [NOTEBOOK_INFERENCE, NOTEBOOK_TRAIN]


# ---------------------------------------------------------------------------
# Checks generiques appliques aux deux notebooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_existe(notebook: Path) -> None:
    """Le fichier notebook est present."""
    assert notebook.exists(), f"Notebook absent : {notebook}"


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_json_valide(notebook: Path) -> None:
    """Le notebook est un JSON valide avec la structure attendue."""
    contenu = json.loads(notebook.read_text(encoding="utf-8"))
    assert isinstance(contenu, dict)
    assert "cells" in contenu
    assert "metadata" in contenu
    assert "nbformat" in contenu


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_utilise_le_noyau_python_local(notebook: Path) -> None:
    """Les notebooks utilisent un noyau Jupyter Python standard."""
    metadata = json.loads(notebook.read_text(encoding="utf-8"))["metadata"]
    assert metadata["kernelspec"]["name"] == "python3"
    assert metadata["language_info"]["name"] == "python"


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cellules_bien_typees(notebook: Path) -> None:
    """Chaque cellule a un cell_type valide et une source."""
    contenu = json.loads(notebook.read_text(encoding="utf-8"))
    for i, cell in enumerate(contenu["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}, f"Cellule {i} : type inconnu"
        assert "source" in cell, f"Cellule {i} sans source"


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cellules_code_parse_python(notebook: Path) -> None:
    """Chaque cellule code parse en Python valide (magics !/% remplaces par pass)."""
    contenu = json.loads(notebook.read_text(encoding="utf-8"))
    for i, cell in enumerate(contenu["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lignes = []
        for ligne in src.split("\n"):
            stripped = ligne.lstrip()
            if stripped.startswith(("!", "%")):
                indent = ligne[: len(ligne) - len(stripped)]
                lignes.append(indent + "pass")
            else:
                lignes.append(ligne)
        try:
            ast.parse("\n".join(lignes))
        except SyntaxError as exc:
            raise AssertionError(f"{notebook.name} cellule {i} : SyntaxError {exc}") from exc


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_pas_de_token_en_dur(notebook: Path) -> None:
    """Aucun token/cle en dur dans les notebooks."""
    contenu = notebook.read_text(encoding="utf-8")
    assert "ghp_" not in contenu
    assert "github_pat_" not in contenu


# ---------------------------------------------------------------------------
# Checks specifiques au notebook d'inference
# ---------------------------------------------------------------------------


def test_inference_utilise_les_dossiers_locaux() -> None:
    """Le notebook retrouve les vidéos, la calibration et les sorties dans le dépôt."""
    contenu = NOTEBOOK_INFERENCE.read_text(encoding="utf-8")
    assert "raw_clips" in contenu
    assert "chemin_correspondances_video(CLIP)" in contenu
    assert "outputs" in contenu


def test_inference_appelle_pipeline() -> None:
    """Le notebook d'inference utilise traiter_match_complet."""
    contenu = NOTEBOOK_INFERENCE.read_text(encoding="utf-8")
    assert "traiter_match_complet" in contenu
    assert "pivot_ai.pipeline" in contenu


def test_inference_propose_modele_handball() -> None:
    """Le notebook d'inference propose de charger le modele handball fine-tune."""
    contenu = NOTEBOOK_INFERENCE.read_text(encoding="utf-8")
    assert "ModeleConfig.pour_handball" in contenu
    assert "modele_config=modele_config" in contenu


# ---------------------------------------------------------------------------
# Checks specifiques au notebook d'entrainement
# ---------------------------------------------------------------------------


def test_train_utilise_roboflow_et_dataset() -> None:
    """Le notebook d'entrainement telecharge le dataset handball via Roboflow."""
    contenu = NOTEBOOK_TRAIN.read_text(encoding="utf-8")
    assert "roboflow" in contenu.lower()
    assert "handball-detection-fj8rc" in contenu
    assert "erics-workspace-it3ft" in contenu
    assert "ROBOFLOW_API_KEY" in contenu


def test_train_entraine_et_exporte() -> None:
    """Le notebook entraîne un YOLO et exporte best.pt dans le dossier models."""
    contenu = NOTEBOOK_TRAIN.read_text(encoding="utf-8")
    assert "model.train" in contenu
    assert "YOLO_BASE" in contenu
    assert "yolo26m" in contenu
    assert "ModeleConfig.pour_handball" in contenu


# ---------------------------------------------------------------------------
# Idempotence des builders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "notebook"),
    [
        ("build_notebook.py", NOTEBOOK_INFERENCE),
        ("build_train_notebook.py", NOTEBOOK_TRAIN),
    ],
    ids=["inference", "train"],
)
def test_builder_notebook_idempotent(builder: str, notebook: Path) -> None:
    """Chaque builder reecrit le meme JSON octet pour octet."""
    script = SCRIPTS_DIR / builder
    assert script.exists()
    avant = notebook.read_bytes()
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True)
    apres = notebook.read_bytes()
    assert avant == apres, f"{builder} n'est pas idempotent"
