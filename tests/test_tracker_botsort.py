"""BoT-SORT backend tests: yaml configuration and pipeline routing.

The real BoT-SORT + ReID behaviour (Ultralytics model.track) needs torch and
a GPU. Here we check what is testable locally: the configuration is valid and
the pipeline routes to the right backend.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from handball_ai.pipeline import process_match
from handball_ai.tracking import BOTSORT_CONFIG_PATH
from tests.conftest import make_fake_video

_MAPPING = {
    "A": {"pixel": (0, 0), "court_m": (0.0, 0.0)},
    "B": {"pixel": (320, 0), "court_m": (40.0, 0.0)},
    "C": {"pixel": (320, 240), "court_m": (40.0, 20.0)},
    "D": {"pixel": (0, 240), "court_m": (0.0, 20.0)},
}


def test_botsort_config_exists_and_valid() -> None:
    """The BoT-SORT configuration ships with the package and enables ReID."""
    assert BOTSORT_CONFIG_PATH.exists(), f"Missing configuration: {BOTSORT_CONFIG_PATH}"
    cfg = yaml.safe_load(BOTSORT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["tracker_type"] == "botsort"
    assert cfg["with_reid"] is True
    # Larger buffer to survive occlusions
    assert cfg["track_buffer"] >= 30


def test_botsort_config_inside_package() -> None:
    """The configuration lives under handball_ai/trackers/ (packaged)."""
    assert BOTSORT_CONFIG_PATH.parent.name == "trackers"
    assert BOTSORT_CONFIG_PATH.parent.parent.name == "handball_ai"


def test_pipeline_routes_to_botsort(tmp_path: Path) -> None:
    """tracker='botsort' calls track_video_botsort, not the ByteTrack path."""
    source = tmp_path / "match.mp4"
    make_fake_video(source, nb_frames=20, fps=25.0, width=320, height=240)

    # The botsort backend is mocked to return an empty tracking
    # (only the ROUTING is checked, not the real tracker).
    with (
        patch("handball_ai.pipeline.track_video_botsort", return_value=({}, {})) as mock_bot,
        patch("handball_ai.pipeline.detect_video") as mock_detect,
        patch("handball_ai.pipeline.track_detections") as mock_byte,
    ):
        process_match(
            video_path=source,
            court_mapping=_MAPPING,
            output_dir=tmp_path / "out",
            detector=object(),  # never used (mocked backend)
            subsample=5,
            with_radar_video=False,
            with_clips=False,
            tracker="botsort",
        )
    mock_bot.assert_called_once()
    mock_detect.assert_not_called()
    mock_byte.assert_not_called()


def test_pipeline_unknown_tracker_raises(tmp_path: Path) -> None:
    """An unknown backend raises ValueError."""
    source = tmp_path / "match.mp4"
    make_fake_video(source, nb_frames=10, fps=25.0, width=320, height=240)
    with pytest.raises(ValueError, match="Unknown tracker"):
        process_match(
            video_path=source,
            court_mapping=_MAPPING,
            output_dir=tmp_path / "out",
            detector=object(),
            tracker="deepsort",
        )
