"""Tests du module detection, sans GPU ni vrais poids YOLO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import supervision as sv
import torch
from ultralytics.engine.results import Results

from pivot_ai.config import CLASSES_HANDBALL
from pivot_ai.detection import (
    Detector,
    build_class_map,
    detect_video,
    normalize_class_name,
    to_handball_classes,
)
from tests.conftest import generer_video_factice

# Ordre alphabetique des exports Roboflow : ball, goalkeeper, players, referees.
NOMS_ROBOFLOW = {0: "ball", 1: "goalkeeper", 2: "players", 3: "referees"}


class FauxYOLO:
    """Imite `ultralytics.YOLO` : une detection de chaque classe par image."""

    def __init__(self, chemin: str) -> None:
        self.chemin = chemin
        self.names = NOMS_ROBOFLOW
        self.appels: list[dict[str, Any]] = []

    def predict(self, frames: list[np.ndarray], **options: Any) -> list[Results]:
        self.appels.append(options)
        boites = torch.tensor(
            [[10, 10, 20, 20, 0.9, cid] for cid in self.names], dtype=torch.float32
        )
        return [Results(frame, path="", names=self.names, boxes=boites) for frame in frames]


@pytest.fixture
def poids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fichier de poids factice, charge par FauxYOLO."""
    monkeypatch.setattr("ultralytics.YOLO", FauxYOLO)
    chemin = tmp_path / "handball.pt"
    chemin.write_bytes(b"")
    return chemin


def _detections(class_ids: list[int]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.tile(np.array([0, 0, 1, 1], dtype=np.float32), (len(class_ids), 1)),
        confidence=np.full(len(class_ids), 0.9, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


# ---------------------------------------------------------------------------
# Correspondance des classes
# ---------------------------------------------------------------------------


def test_normalize_class_name() -> None:
    """Casse, separateurs et pluriel simple sont normalises."""
    assert normalize_class_name("Players") == "player"
    assert normalize_class_name("referees") == "referee"
    assert normalize_class_name("Goal_Keeper") == "goal keeper"
    assert normalize_class_name("ball") == "ball"


def test_build_class_map_ordre_roboflow() -> None:
    """L'ordre alphabetique Roboflow est converti vers CLASSES_HANDBALL par nom."""
    assert build_class_map(NOMS_ROBOFLOW) == {
        0: CLASSES_HANDBALL["ball"],
        1: CLASSES_HANDBALL["goalkeeper"],
        2: CLASSES_HANDBALL["players"],
        3: CLASSES_HANDBALL["referees"],
    }


def test_build_class_map_exclusion() -> None:
    """Une classe exclue, citee au singulier ou au pluriel, disparait."""
    correspondance = build_class_map(NOMS_ROBOFLOW, exclude=["Ball", "referee"])
    assert correspondance == {1: CLASSES_HANDBALL["goalkeeper"], 2: CLASSES_HANDBALL["players"]}


def test_build_class_map_ignore_classes_non_handball() -> None:
    """Une classe hors handball (crowd) est ignoree."""
    assert build_class_map({0: "players", 1: "crowd"}) == {0: CLASSES_HANDBALL["players"]}


@pytest.mark.parametrize(
    ("noms", "exclusions", "message"),
    [
        (NOMS_ROBOFLOW, ["crowd"], "Unknown excluded classes: crowd"),
        ({0: "person", 1: "car"}, [], "no handball class"),
        ({0: "players"}, ["players"], "No class left"),
    ],
    ids=["exclusion-inconnue", "modele-coco", "tout-exclu"],
)
def test_build_class_map_erreurs(
    noms: dict[int, str], exclusions: list[str], message: str
) -> None:
    """Configurations invalides : ValueError explicite."""
    with pytest.raises(ValueError, match=message):
        build_class_map(noms, exclusions)


def test_to_handball_classes_convertit_et_filtre() -> None:
    """Les ids sont convertis ; les classes hors correspondance sont ecartees."""
    sortie = to_handball_classes(_detections([2, 0, 9]), {0: 3, 2: 0})
    assert sortie.class_id.tolist() == [0, 3]


def test_to_handball_classes_vide() -> None:
    """Des detections vides passent sans erreur."""
    assert len(to_handball_classes(sv.Detections.empty(), {0: 0})) == 0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detector_poids_absents(tmp_path: Path) -> None:
    """Fichier de poids absent : FileNotFoundError avant tout chargement."""
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        Detector(tmp_path / "absent.pt")


def test_detector_confiance_invalide(poids: Path) -> None:
    """Seuil de confiance hors [0, 1] : ValueError."""
    with pytest.raises(ValueError, match="Confidence"):
        Detector(poids, confidence=1.5)


def test_detector_detect(poids: Path) -> None:
    """detect convertit les classes et transmet seuil et classes gardees a YOLO."""
    detecteur = Detector(poids, confidence=0.5, exclude=["ball"])
    images = [np.zeros((32, 32, 3), dtype=np.uint8)] * 2

    resultats = detecteur.detect(images)

    assert len(resultats) == 2
    attendu = sorted(
        CLASSES_HANDBALL[nom] for nom in ("goalkeeper", "players", "referees")
    )
    assert sorted(resultats[0].class_id.tolist()) == attendu
    options = detecteur.model.appels[0]
    assert options["conf"] == 0.5
    assert options["classes"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# detect_video
# ---------------------------------------------------------------------------


class DetecteurStub:
    """Enregistre la taille des lots recus et renvoie des detections vides."""

    def __init__(self) -> None:
        self.lots: list[int] = []

    def detect(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        self.lots.append(len(frames))
        return [sv.Detections.empty() for _ in frames]


def test_detect_video_echantillonne_par_lots(tmp_path: Path) -> None:
    """subsample, batch_size, start_frame et max_frames sont respectes."""
    video = tmp_path / "clip.mp4"
    generer_video_factice(video, nb_frames=30, fps=25.0, largeur=64, hauteur=48)
    stub = DetecteurStub()

    resultats = dict(detect_video(
        video, stub, subsample=3, batch_size=4, start_frame=5, max_frames=20,
    ))

    assert list(resultats) == [0, 3, 6, 9, 12, 15, 18]
    assert stub.lots == [4, 3]


def test_detect_video_est_paresseux(tmp_path: Path) -> None:
    """Le generateur ne detecte que le premier lot si on s'arrete au premier element."""
    video = tmp_path / "clip.mp4"
    generer_video_factice(video, nb_frames=20, fps=25.0, largeur=64, hauteur=48)
    stub = DetecteurStub()

    generateur = detect_video(video, stub, batch_size=4)
    index, _ = next(generateur)
    generateur.close()

    assert index == 0
    assert stub.lots == [4]


@pytest.mark.parametrize("parametres", [{"subsample": 0}, {"batch_size": 0}])
def test_detect_video_parametres_invalides(
    tmp_path: Path, parametres: dict[str, int]
) -> None:
    """subsample ou batch_size < 1 : ValueError des le premier element."""
    with pytest.raises(ValueError):
        next(detect_video(tmp_path / "x.mp4", DetecteurStub(), **parametres))
