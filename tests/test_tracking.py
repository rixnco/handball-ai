"""Tracking tests: frame sampling, tracker reset, class stabilization, configurations.

The real trackers run inside Ultralytics `model.track` and need YOLO weights;
here a stub detector checks what `track_video` controls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv
import yaml

from handball_ai.config import HANDBALL_CLASSES
from handball_ai.tracking import TRACKER_CONFIGS, majority_class, track_video
from tests.conftest import make_fake_video

PLAYER = HANDBALL_CLASSES["players"]
REFEREE = HANDBALL_CLASSES["referees"]


def _tracked(tracker_ids: list[int], class_ids: list[int]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.tile(np.array([0, 0, 10, 20], dtype=np.float32), (len(tracker_ids), 1)),
        confidence=np.full(len(tracker_ids), 0.9, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
    )


class StubTracker:
    """Records the calls and returns one tracked player per frame (tracker_id 1)."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.configs: list[Path] = []

    def reset_tracking(self) -> None:
        self.calls.append("reset")

    def track(self, frame: np.ndarray, tracker_config: str | Path) -> sv.Detections:
        self.calls.append("track")
        self.configs.append(Path(tracker_config))
        return _tracked([1], [PLAYER])


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    make_fake_video(path, nb_frames=20, fps=25.0, width=64, height=48)
    return path


# ---------------------------------------------------------------------------
# track_video
# ---------------------------------------------------------------------------


def test_track_video_resets_before_the_first_frame(video: Path) -> None:
    """The tracker is reset once, before any frame: no ID carries over between videos."""
    stub = StubTracker()
    track_video(video, stub)
    assert stub.calls[0] == "reset"
    assert stub.calls.count("reset") == 1
    assert stub.calls.count("track") == 20


def test_track_video_resets_on_each_video(video: Path) -> None:
    """Two videos with the same detector: one reset per video."""
    stub = StubTracker()
    track_video(video, stub)
    track_video(video, stub)
    assert stub.calls.count("reset") == 2
    assert stub.calls[20 + 1] == "reset"


def test_track_video_subsample(video: Path) -> None:
    """Only 1 frame out of `subsample` is tracked, indexed by its source position."""
    stub = StubTracker()
    tracked, final_class = track_video(video, stub, subsample=3)
    assert sorted(tracked) == [0, 3, 6, 9, 12, 15, 18]
    assert final_class == {1: PLAYER}


def test_track_video_start_and_max_frames(video: Path) -> None:
    """start_frame and max_frames limit the frames read; indices stay relative."""
    stub = StubTracker()
    tracked, _ = track_video(video, stub, start_frame=5, max_frames=8, subsample=2)
    assert sorted(tracked) == [0, 2, 4, 6]


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_track_video_uses_the_shipped_config(video: Path, name: str) -> None:
    """Each backend passes its packaged YAML to the detector."""
    stub = StubTracker()
    track_video(video, stub, tracker=name, max_frames=2)
    assert set(stub.configs) == {TRACKER_CONFIGS[name]}


@pytest.mark.parametrize(
    ("params", "message"),
    [({"tracker": "deepsort"}, "Unknown tracker"), ({"subsample": 0}, "subsample")],
    ids=["unknown-tracker", "zero-subsample"],
)
def test_track_video_invalid_parameters(video: Path, params: dict, message: str) -> None:
    """Invalid parameters raise ValueError before touching the tracker."""
    stub = StubTracker()
    with pytest.raises(ValueError, match=message):
        track_video(video, stub, **params)
    assert stub.calls == []


def test_track_video_missing_video(tmp_path: Path) -> None:
    """Unreadable video: RuntimeError."""
    with pytest.raises(RuntimeError, match="Cannot open video"):
        track_video(tmp_path / "missing.mp4", StubTracker())


# ---------------------------------------------------------------------------
# majority_class
# ---------------------------------------------------------------------------


def test_majority_class_stabilizes_flickering_class() -> None:
    """A tracker seen 2 times as player and once as referee is a player."""
    tracked = {
        0: _tracked([7], [PLAYER]),
        1: _tracked([7], [REFEREE]),
        2: _tracked([7, 8], [PLAYER, REFEREE]),
    }
    assert majority_class(tracked) == {7: PLAYER, 8: REFEREE}


def test_majority_class_ignores_untracked_frames() -> None:
    """Frames without tracker_id are skipped."""
    assert majority_class({0: sv.Detections.empty()}) == {}


# ---------------------------------------------------------------------------
# Tracker configurations
# ---------------------------------------------------------------------------


def _load(name: str) -> dict:
    return yaml.safe_load(TRACKER_CONFIGS[name].read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_tracker_config_shipped_in_package(name: str) -> None:
    """Each configuration lives under handball_ai/trackers/ with the right type."""
    path = TRACKER_CONFIGS[name]
    assert path.is_file()
    assert path.parent.name == "trackers"
    assert path.parent.parent.name == "handball_ai"
    cfg = _load(name)
    assert cfg["tracker_type"] == name
    assert cfg["track_low_thresh"] < cfg["track_high_thresh"] <= cfg["new_track_thresh"]


def test_botsort_config_enables_reid() -> None:
    """BoT-SORT uses ReID with the YOLO features and a longer buffer for occlusions."""
    cfg = _load("botsort")
    assert cfg["with_reid"] is True
    assert cfg["model"] == "auto"
    assert cfg["track_buffer"] > _load("bytetrack")["track_buffer"]
