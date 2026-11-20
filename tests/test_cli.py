"""Tests unitaires pour pivot_ai.cli."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from pivot_ai.cli import _charger_correspondances, commande_traiter, construire_parser, main
from pivot_ai.pipeline import ResultatPipeline
from tests.conftest import generer_video_factice

# ---------------------------------------------------------------------------
# _charger_correspondances
# ---------------------------------------------------------------------------


def test_charger_correspondances_json_valide(tmp_path: Path) -> None:
    """JSON 4 points : dict de Correspondance avec tuples coerces aux bons types."""
    chemin = tmp_path / "h.json"
    contenu = {
        "A": {"pixel": [100, 200], "terrain_m": [0.0, 0.0]},
        "B": {"pixel": [1000, 200], "terrain_m": [40.0, 0.0]},
        "C": {"pixel": [1000, 600], "terrain_m": [40.0, 20.0]},
        "D": {"pixel": [100, 600], "terrain_m": [0.0, 20.0]},
    }
    chemin.write_text(json.dumps(contenu), encoding="utf-8")

    res = _charger_correspondances(chemin)
    assert set(res.keys()) == {"A", "B", "C", "D"}
    assert res["A"]["pixel"] == (100, 200)
    assert res["A"]["terrain_m"] == (0.0, 0.0)
    assert isinstance(res["A"]["pixel"][0], int)
    assert isinstance(res["A"]["terrain_m"][0], float)


def test_charger_correspondances_ignore_cles_commentaire(tmp_path: Path) -> None:
    """Les cles commencant par _ sont ignorees (commentaires)."""
    chemin = tmp_path / "h.json"
    contenu = {
        "_commentaire": "doc inline",
        "A": {"pixel": [0, 0], "terrain_m": [0.0, 0.0]},
        "B": {"pixel": [1, 0], "terrain_m": [40.0, 0.0]},
        "C": {"pixel": [1, 1], "terrain_m": [40.0, 20.0]},
        "D": {"pixel": [0, 1], "terrain_m": [0.0, 20.0]},
    }
    chemin.write_text(json.dumps(contenu), encoding="utf-8")
    res = _charger_correspondances(chemin)
    assert "_commentaire" not in res
    assert len(res) == 4


def test_charger_correspondances_moins_de_4_points(tmp_path: Path) -> None:
    """Moins de 4 points : ValueError."""
    chemin = tmp_path / "h.json"
    contenu = {
        "A": {"pixel": [0, 0], "terrain_m": [0.0, 0.0]},
        "B": {"pixel": [1, 0], "terrain_m": [1.0, 0.0]},
    }
    chemin.write_text(json.dumps(contenu), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum 4 points"):
        _charger_correspondances(chemin)


def test_charger_correspondances_json_malforme(tmp_path: Path) -> None:
    """JSON casse : JSONDecodeError."""
    chemin = tmp_path / "h.json"
    chemin.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _charger_correspondances(chemin)


def test_charger_correspondances_structure_invalide(tmp_path: Path) -> None:
    """Un point sans 'pixel' ou 'terrain_m' : ValueError."""
    chemin = tmp_path / "h.json"
    contenu = {
        "A": {"pixel": [0, 0], "terrain_m": [0.0, 0.0]},
        "B": {"pixel": [1, 0], "terrain_m": [1.0, 0.0]},
        "C": {"pixel": [1, 1], "terrain_m": [1.0, 1.0]},
        "D": {"pixel": [0, 1]},  # manque terrain_m
    }
    chemin.write_text(json.dumps(contenu), encoding="utf-8")
    with pytest.raises(ValueError, match="terrain_m"):
        _charger_correspondances(chemin)


def test_charger_correspondances_fichier_absent(tmp_path: Path) -> None:
    """Fichier inexistant : FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _charger_correspondances(tmp_path / "absent.json")


def test_charger_correspondances_exemple_docs_charge() -> None:
    """L'exemple docs/correspondances_example.json est charge sans erreur."""
    chemin = Path(__file__).parent.parent / "docs" / "correspondances_example.json"
    if not chemin.exists():
        pytest.skip(f"Exemple absent : {chemin}")
    res = _charger_correspondances(chemin)
    assert len(res) >= 4


# ---------------------------------------------------------------------------
# commande_traiter (avec mock du pipeline)
# ---------------------------------------------------------------------------


def _ecrire_homographie_4pts(tmp_path: Path) -> Path:
    chemin = tmp_path / "h.json"
    chemin.write_text(
        json.dumps({
            "A": {"pixel": [0, 0], "terrain_m": [0.0, 0.0]},
            "B": {"pixel": [320, 0], "terrain_m": [40.0, 0.0]},
            "C": {"pixel": [320, 240], "terrain_m": [40.0, 20.0]},
            "D": {"pixel": [0, 240], "terrain_m": [0.0, 20.0]},
        }),
        encoding="utf-8",
    )
    return chemin


def test_commande_traiter_video_inexistante(tmp_path: Path) -> None:
    """Video absente : code de sortie 1."""
    h = _ecrire_homographie_4pts(tmp_path)
    parser = construire_parser()
    args = parser.parse_args([
        "traiter",
        "--video", str(tmp_path / "absent.mp4"),
        "--output", str(tmp_path / "out"),
        "--homographie", str(h),
        "--modele", str(tmp_path / "modele.pt"),
    ])
    assert commande_traiter(args) == 1


def test_commande_traiter_homographie_invalide(tmp_path: Path) -> None:
    """Homographie a 2 points : code de sortie 1, message d'erreur logge."""
    source = tmp_path / "vid.mp4"
    generer_video_factice(source, nb_frames=10, fps=25.0, largeur=320, hauteur=240)
    h = tmp_path / "h.json"
    h.write_text(
        json.dumps({"A": {"pixel": [0, 0], "terrain_m": [0.0, 0.0]}}),
        encoding="utf-8",
    )
    parser = construire_parser()
    args = parser.parse_args([
        "traiter",
        "--video", str(source),
        "--output", str(tmp_path / "out"),
        "--homographie", str(h),
        "--modele", str(tmp_path / "modele.pt"),
    ])
    assert commande_traiter(args) == 1


def test_commande_traiter_utilise_calibration_associee(tmp_path: Path) -> None:
    """Pipeline mocke : commande_traiter retourne 0 et appelle traiter_match_complet."""
    source = tmp_path / "vid.mp4"
    generer_video_factice(source, nb_frames=10, fps=25.0, largeur=320, hauteur=240)
    _ecrire_homographie_4pts(tmp_path).rename(tmp_path / "vid.mp4.correspondances.json")
    sortie = tmp_path / "out"

    resultat_mock = ResultatPipeline(
        chemin_video_source=source,
        chemin_video_radar=sortie / "match_radar_sbs.mp4",
        stats_joueurs=pl.DataFrame({"tracker_id": [1]}),
        positions_par_tracker={},
        actions_detectees=[],
        clips_decoupes=[],
        metadonnees={"nb_trackers_total": 3, "nb_joueurs_classes": 2},
    )

    parser = construire_parser()
    args = parser.parse_args([
        "traiter",
        "--video", str(source),
        "--output", str(sortie),
        "--subsample", "3",
        "--debut", "00:00:00:12",
        "--duree", "0.2",
        "--no-video-radar",
        "--no-decoupage",
        "--modele", "models/handball.pt",
        "--exclude", "ball",
    ])

    with (
        patch("pivot_ai.cli.Detector") as mock_detecteur,
        patch("pivot_ai.cli.traiter_match_complet", return_value=resultat_mock) as mock_pipe,
    ):
        code = commande_traiter(args)
        mock_detecteur.assert_called_once_with(
            Path("models/handball.pt"), confidence=0.35, exclude=["ball"]
        )
        assert code == 0
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args.kwargs
        assert kwargs["subsample"] == 3
        assert kwargs["chemin_video"] == source
        assert kwargs["debut"] == 0.12
        assert kwargs["duree"] == 0.2
        assert kwargs["generer_video_radar"] is False
        assert kwargs["decouper_actions"] is False
        assert kwargs["classer_equipes"] is False
        assert kwargs["calculer_stats"] is False
        assert len(kwargs["correspondances_homographie"]) == 4
        assert kwargs["detecteur"] is mock_detecteur.return_value


def test_commande_traiter_pipeline_leve_exception(tmp_path: Path) -> None:
    """Si traiter_match_complet leve une exception, code 1 et pas de crash."""
    source = tmp_path / "vid.mp4"
    generer_video_factice(source, nb_frames=10, fps=25.0, largeur=320, hauteur=240)
    h = _ecrire_homographie_4pts(tmp_path)

    parser = construire_parser()
    args = parser.parse_args([
        "traiter",
        "--video", str(source),
        "--output", str(tmp_path / "out"),
        "--homographie", str(h),
        "--modele", str(tmp_path / "modele.pt"),
    ])

    with (
        patch("pivot_ai.cli.Detector"),
        patch("pivot_ai.cli.traiter_match_complet", side_effect=RuntimeError("boom")),
    ):
        code = commande_traiter(args)
        assert code == 1


# ---------------------------------------------------------------------------
# main / parser
# ---------------------------------------------------------------------------


def test_main_sans_commande_renvoie_erreur(capsys: pytest.CaptureFixture[str]) -> None:
    """main() sans sous-commande : SystemExit (argparse)."""
    with pytest.raises(SystemExit):
        main([])


def test_parser_flags_par_defaut(tmp_path: Path) -> None:
    """Sans --no-* : generation video et decoupage actives."""
    h = _ecrire_homographie_4pts(tmp_path)
    parser = construire_parser()
    args = parser.parse_args([
        "traiter",
        "--video", "v.mp4",
        "--output", "o",
        "--homographie", str(h),
        "--modele", str(tmp_path / "modele.pt"),
    ])
    assert args.no_video_radar is False
    assert args.no_decoupage is False
    assert args.subsample == 2


def test_parser_exige_le_modele() -> None:
    """--modele est obligatoire : argparse quitte avec le code 2."""
    with pytest.raises(SystemExit) as exc:
        construire_parser().parse_args(["traiter", "--video", "v.mp4", "--output", "out"])
    assert exc.value.code == 2


def test_commande_traiter_modele_absent(tmp_path: Path) -> None:
    """Fichier de poids absent : code de sortie 1, pipeline non lance."""
    source = tmp_path / "vid.mp4"
    generer_video_factice(source, nb_frames=10, fps=25.0, largeur=320, hauteur=240)
    args = construire_parser().parse_args([
        "traiter",
        "--video", str(source),
        "--output", str(tmp_path / "out"),
        "--homographie", str(_ecrire_homographie_4pts(tmp_path)),
        "--modele", str(tmp_path / "absent.pt"),
    ])
    with patch("pivot_ai.cli.traiter_match_complet") as mock_pipe:
        assert commande_traiter(args) == 1
    mock_pipe.assert_not_called()
