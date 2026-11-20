"""Unit tests for handball_ai.cli."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from handball_ai.cli import _load_court_mapping, build_parser, main, process_command
from handball_ai.pipeline import PipelineResult
from tests.conftest import make_fake_video

# ---------------------------------------------------------------------------
# _load_court_mapping
# ---------------------------------------------------------------------------


def test_load_court_mapping_valid_json(tmp_path: Path) -> None:
    """4-point JSON: CourtPoint dict with tuples coerced to the right types."""
    path = tmp_path / "m.json"
    content = {
        "A": {"pixel": [100, 200], "court_m": [0.0, 0.0]},
        "B": {"pixel": [1000, 200], "court_m": [40.0, 0.0]},
        "C": {"pixel": [1000, 600], "court_m": [40.0, 20.0]},
        "D": {"pixel": [100, 600], "court_m": [0.0, 20.0]},
    }
    path.write_text(json.dumps(content), encoding="utf-8")

    res = _load_court_mapping(path)
    assert set(res) == {"A", "B", "C", "D"}
    assert res["A"]["pixel"] == (100, 200)
    assert res["A"]["court_m"] == (0.0, 0.0)
    assert isinstance(res["A"]["pixel"][0], int)
    assert isinstance(res["A"]["court_m"][0], float)


def test_load_court_mapping_ignores_comment_keys(tmp_path: Path) -> None:
    """Keys starting with _ are ignored (comments)."""
    path = tmp_path / "m.json"
    content = {
        "_comment": "inline doc",
        "A": {"pixel": [0, 0], "court_m": [0.0, 0.0]},
        "B": {"pixel": [1, 0], "court_m": [40.0, 0.0]},
        "C": {"pixel": [1, 1], "court_m": [40.0, 20.0]},
        "D": {"pixel": [0, 1], "court_m": [0.0, 20.0]},
    }
    path.write_text(json.dumps(content), encoding="utf-8")
    res = _load_court_mapping(path)
    assert "_comment" not in res
    assert len(res) == 4


def test_load_court_mapping_fewer_than_4_points(tmp_path: Path) -> None:
    """Fewer than 4 points: ValueError."""
    path = tmp_path / "m.json"
    content = {
        "A": {"pixel": [0, 0], "court_m": [0.0, 0.0]},
        "B": {"pixel": [1, 0], "court_m": [1.0, 0.0]},
    }
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 4 points"):
        _load_court_mapping(path)


def test_load_court_mapping_malformed_json(tmp_path: Path) -> None:
    """Broken JSON: JSONDecodeError."""
    path = tmp_path / "m.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _load_court_mapping(path)


def test_load_court_mapping_invalid_structure(tmp_path: Path) -> None:
    """A point without 'pixel' or 'court_m': ValueError."""
    path = tmp_path / "m.json"
    content = {
        "A": {"pixel": [0, 0], "court_m": [0.0, 0.0]},
        "B": {"pixel": [1, 0], "court_m": [1.0, 0.0]},
        "C": {"pixel": [1, 1], "court_m": [1.0, 1.0]},
        "D": {"pixel": [0, 1]},  # court_m missing
    }
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="court_m"):
        _load_court_mapping(path)


def test_load_court_mapping_missing_file(tmp_path: Path) -> None:
    """Missing file: FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _load_court_mapping(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "relative_path", ["docs/court_mapping_example.json", "data/court_mapping.json"]
)
def test_load_court_mapping_repository_examples(relative_path: str) -> None:
    """The court mappings shipped with the repository load without error."""
    path = Path(__file__).parent.parent / relative_path
    assert len(_load_court_mapping(path)) >= 4


# ---------------------------------------------------------------------------
# process_command (with a mocked pipeline)
# ---------------------------------------------------------------------------


def _write_4pt_mapping(tmp_path: Path) -> Path:
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps({
            "A": {"pixel": [0, 0], "court_m": [0.0, 0.0]},
            "B": {"pixel": [320, 0], "court_m": [40.0, 0.0]},
            "C": {"pixel": [320, 240], "court_m": [40.0, 20.0]},
            "D": {"pixel": [0, 240], "court_m": [0.0, 20.0]},
        }),
        encoding="utf-8",
    )
    return path


def test_process_command_missing_video(tmp_path: Path) -> None:
    """Missing video: exit code 1."""
    args = build_parser().parse_args([
        "process",
        "--video", str(tmp_path / "missing.mp4"),
        "--output", str(tmp_path / "out"),
        "--court-mapping", str(_write_4pt_mapping(tmp_path)),
        "--model", str(tmp_path / "model.pt"),
    ])
    assert process_command(args) == 1


def test_process_command_invalid_court_mapping(tmp_path: Path) -> None:
    """2-point court mapping: exit code 1 and logged error."""
    source = tmp_path / "vid.mp4"
    make_fake_video(source, nb_frames=10, fps=25.0, width=320, height=240)
    mapping = tmp_path / "m.json"
    mapping.write_text(json.dumps({"A": {"pixel": [0, 0], "court_m": [0.0, 0.0]}}), encoding="utf-8")
    args = build_parser().parse_args([
        "process",
        "--video", str(source),
        "--output", str(tmp_path / "out"),
        "--court-mapping", str(mapping),
        "--model", str(tmp_path / "model.pt"),
    ])
    assert process_command(args) == 1


def test_process_command_uses_mapping_next_to_video(tmp_path: Path) -> None:
    """Mocked pipeline: process_command returns 0 and calls process_match."""
    source = tmp_path / "vid.mp4"
    make_fake_video(source, nb_frames=10, fps=25.0, width=320, height=240)
    _write_4pt_mapping(tmp_path).rename(tmp_path / "vid.mp4.court_mapping.json")
    output = tmp_path / "out"

    mock_result = PipelineResult(
        source_video=source,
        radar_video=output / "match_radar.mp4",
        player_stats=pl.DataFrame({"tracker_id": [1]}),
        positions_by_tracker={},
        detected_actions=[],
        action_clips=[],
        metadata={"nb_trackers_total": 3, "nb_classified_players": 2},
    )

    args = build_parser().parse_args([
        "process",
        "--video", str(source),
        "--output", str(output),
        "--subsample", "3",
        "--start", "00:00:00:12",
        "--duration", "0.2",
        "--no-radar-video",
        "--no-clips",
        "--model", "models/handball.pt",
        "--exclude", "ball",
    ])

    with (
        patch("handball_ai.cli.Detector") as mock_detector,
        patch("handball_ai.cli.process_match", return_value=mock_result) as mock_pipe,
    ):
        assert process_command(args) == 0
        mock_detector.assert_called_once_with(
            Path("models/handball.pt"), confidence=0.35, exclude=["ball"]
        )
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args.kwargs
        assert kwargs["subsample"] == 3
        assert kwargs["video_path"] == source
        assert kwargs["start"] == 0.12
        assert kwargs["duration"] == 0.2
        assert kwargs["with_radar_video"] is False
        assert kwargs["with_clips"] is False
        assert kwargs["with_teams"] is False
        assert kwargs["with_stats"] is False
        assert len(kwargs["court_mapping"]) == 4
        assert kwargs["detector"] is mock_detector.return_value


def test_process_command_pipeline_raises(tmp_path: Path) -> None:
    """When process_match raises, exit code 1 and no crash."""
    source = tmp_path / "vid.mp4"
    make_fake_video(source, nb_frames=10, fps=25.0, width=320, height=240)
    args = build_parser().parse_args([
        "process",
        "--video", str(source),
        "--output", str(tmp_path / "out"),
        "--court-mapping", str(_write_4pt_mapping(tmp_path)),
        "--model", str(tmp_path / "model.pt"),
    ])

    with (
        patch("handball_ai.cli.Detector"),
        patch("handball_ai.cli.process_match", side_effect=RuntimeError("boom")),
    ):
        assert process_command(args) == 1


def test_process_command_missing_model(tmp_path: Path) -> None:
    """Missing weights file: exit code 1, pipeline not started."""
    source = tmp_path / "vid.mp4"
    make_fake_video(source, nb_frames=10, fps=25.0, width=320, height=240)
    args = build_parser().parse_args([
        "process",
        "--video", str(source),
        "--output", str(tmp_path / "out"),
        "--court-mapping", str(_write_4pt_mapping(tmp_path)),
        "--model", str(tmp_path / "missing.pt"),
    ])
    with patch("handball_ai.cli.process_match") as mock_pipe:
        assert process_command(args) == 1
    mock_pipe.assert_not_called()


# ---------------------------------------------------------------------------
# main / parser
# ---------------------------------------------------------------------------


def test_main_without_command_fails() -> None:
    """main() without a sub-command: SystemExit (argparse)."""
    with pytest.raises(SystemExit):
        main([])


def test_parser_requires_model() -> None:
    """--model is required: argparse exits with code 2."""
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["process", "--video", "v.mp4", "--output", "out"])
    assert exc.value.code == 2


def test_parser_default_flags() -> None:
    """Without --no-*: radar video and clips enabled."""
    args = build_parser().parse_args([
        "process", "--video", "v.mp4", "--output", "o", "--model", "m.pt",
    ])
    assert args.no_radar_video is False
    assert args.no_clips is False
    assert args.subsample == 2
    assert args.court_mapping is None
