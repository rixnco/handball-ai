"""Unit tests for handball_ai.pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from handball_ai.config import HANDBALL_CLASSES, CourtConfig
from handball_ai.homography import calibrate_homography
from handball_ai.pipeline import (
    COLOR_AWAY,
    COLOR_GK,
    COLOR_HOME,
    COLOR_REF,
    PipelineResult,
    filter_short_tracks,
    generate_radar_video,
    interpolate_positions,
    process_match,
    project_detections_to_court,
    tracker_color,
)
from handball_ai.stats import PlayerPosition
from tests.conftest import make_fake_video

# Identity-like homography: pixels = metres (handy for tests)
_TEST_MAPPING = {
    "A": {"pixel": (0, 0), "court_m": (0.0, 0.0)},
    "B": {"pixel": (40, 0), "court_m": (40.0, 0.0)},
    "C": {"pixel": (40, 20), "court_m": (40.0, 20.0)},
    "D": {"pixel": (0, 20), "court_m": (0.0, 20.0)},
}

# Pixels (0..320, 0..240) -> court (0..40, 0..20)
_VIDEO_MAPPING = {
    "A": {"pixel": (0, 0), "court_m": (0.0, 0.0)},
    "B": {"pixel": (320, 0), "court_m": (40.0, 0.0)},
    "C": {"pixel": (320, 240), "court_m": (40.0, 20.0)},
    "D": {"pixel": (0, 240), "court_m": (0.0, 20.0)},
}


def _tracked(
    bboxes: list[tuple[float, float, float, float]],
    tracker_ids: list[int],
    class_ids: list[int],
) -> sv.Detections:
    n = len(bboxes)
    return sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        confidence=np.ones(n, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
    )


# ---------------------------------------------------------------------------
# project_detections_to_court
# ---------------------------------------------------------------------------


def test_project_keeps_only_players() -> None:
    """Only 'players' trackers are kept."""
    homography = calibrate_homography(_TEST_MAPPING)
    # bbox bottom centre at pixel (20, 10) -> court (20, 10)
    dets = _tracked(bboxes=[(15, 0, 25, 10), (15, 0, 25, 10)], tracker_ids=[1, 2], class_ids=[0, 1])
    final_class = {1: 0, 2: 1}  # 1 player, 1 GK
    pos = project_detections_to_court(
        {0: dets}, final_class, homography, player_class_id=0, filter_zone=False,
    )
    assert set(pos) == {1}
    assert len(pos[1]) == 1
    assert math.isclose(pos[1][0].x_m, 20.0, abs_tol=1e-3)
    assert math.isclose(pos[1][0].y_m, 10.0, abs_tol=1e-3)
    assert pos[1][0].interpolated is False


def test_project_zone_filter_drops_benches() -> None:
    """With filter_zone=True, positions outside the margins are rejected."""
    homography = calibrate_homography(_TEST_MAPPING)
    # The bottom centre of the second tracker is at y=1.0 m, below margin_y_m=2.0
    dets = _tracked(bboxes=[(15, 0, 25, 10), (15, 0, 25, 1)], tracker_ids=[1, 2], class_ids=[0, 0])
    pos = project_detections_to_court(
        {0: dets}, {1: 0, 2: 0}, homography, player_class_id=0, filter_zone=True,
    )
    assert set(pos) == {1}


# ---------------------------------------------------------------------------
# interpolate_positions / filter_short_tracks
# ---------------------------------------------------------------------------


def test_interpolate_small_gap_fills_linearly() -> None:
    """Tracker seen at frames 0 and 4: linear interpolation at 1, 2, 3."""
    positions = {
        7: [
            PlayerPosition(frame_idx=0, x_m=0.0, y_m=0.0),
            PlayerPosition(frame_idx=4, x_m=4.0, y_m=8.0),
        ]
    }
    res = interpolate_positions(positions, max_gap_frames=10)
    assert len(res[7]) == 5
    # frame 1: alpha=0.25 -> x=1, y=2
    assert math.isclose(res[7][1].x_m, 1.0, abs_tol=1e-9)
    assert math.isclose(res[7][1].y_m, 2.0, abs_tol=1e-9)
    assert res[7][1].interpolated is True
    # frame 4: not interpolated
    assert res[7][4].frame_idx == 4
    assert res[7][4].interpolated is False


def test_interpolate_large_gap_not_filled() -> None:
    """Gap above max_gap_frames: no interpolation, segments preserved."""
    positions = {
        1: [
            PlayerPosition(frame_idx=0, x_m=0.0, y_m=0.0),
            PlayerPosition(frame_idx=100, x_m=10.0, y_m=10.0),
        ]
    }
    res = interpolate_positions(positions, max_gap_frames=10)
    assert len(res[1]) == 2
    assert all(not p.interpolated for p in res[1])


def test_interpolate_no_extrapolation_at_edges() -> None:
    """No interpolation before the first or after the last position."""
    positions = {
        1: [
            PlayerPosition(frame_idx=5, x_m=5.0, y_m=5.0),
            PlayerPosition(frame_idx=8, x_m=8.0, y_m=8.0),
        ]
    }
    res = interpolate_positions(positions, max_gap_frames=10)
    assert res[1][0].frame_idx == 5
    assert res[1][-1].frame_idx == 8


def test_filter_short_tracks() -> None:
    """Keeps trackers seen on >= min_frames real frames, drops fragments."""
    positions = {
        1: [PlayerPosition(fi, float(fi), 5.0) for fi in range(20)],  # 20 real
        2: [PlayerPosition(0, 1.0, 1.0), PlayerPosition(1, 2.0, 2.0)],  # 2 real
        3: [
            PlayerPosition(0, 1.0, 1.0, interpolated=False),
            PlayerPosition(1, 2.0, 2.0, interpolated=True),
            PlayerPosition(2, 3.0, 3.0, interpolated=True),
        ],  # a single real one (interpolated ones do not count)
    }
    assert set(filter_short_tracks(positions, min_frames=5)) == {1}


def test_filter_short_tracks_disabled() -> None:
    """min_frames <= 1 filters nothing."""
    positions = {1: [PlayerPosition(0, 0.0, 0.0)]}
    assert filter_short_tracks(positions, min_frames=0) == positions
    assert filter_short_tracks(positions, min_frames=1) == positions


# ---------------------------------------------------------------------------
# tracker_color
# ---------------------------------------------------------------------------


def test_tracker_color_by_role() -> None:
    """Distinct colours for HOME, AWAY, GK, REF."""
    cls = {
        1: HANDBALL_CLASSES["players"],
        2: HANDBALL_CLASSES["players"],
        3: HANDBALL_CLASSES["goalkeeper"],
        4: HANDBALL_CLASSES["referees"],
    }
    team = {1: 0, 2: 1}  # tracker 1 = cluster 0 (home), tracker 2 = cluster 1 (away)
    assert tracker_color(1, cls, team, home_cluster=0) == COLOR_HOME
    assert tracker_color(2, cls, team, home_cluster=0) == COLOR_AWAY
    assert tracker_color(3, cls, team, home_cluster=0) == COLOR_GK
    assert tracker_color(4, cls, team, home_cluster=0) == COLOR_REF


# ---------------------------------------------------------------------------
# generate_radar_video
# ---------------------------------------------------------------------------


def test_generate_radar_video_dimensions(tmp_path: Path) -> None:
    """The source and the 32:9 radar are stacked in a 1080p video."""
    source = tmp_path / "source.mp4"
    make_fake_video(source, nb_frames=10, fps=25.0, width=640, height=180)

    output = tmp_path / "radar.mp4"
    config = CourtConfig()
    res = generate_radar_video(
        source,
        {0: _tracked([(100, 50, 200, 200)], [1], [0])},
        {1: [PlayerPosition(frame_idx=fi, x_m=15.0, y_m=10.0) for fi in range(10)]},
        {1: HANDBALL_CLASSES["players"]},
        final_team={1: 0},
        home_cluster=0,
        output_path=output,
        court_config=config,
    )

    assert res == output
    assert output.exists()
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert width == config.broadcast_w == config.radar_w == 1920
    assert height == config.broadcast_h + config.radar_h == 1080
    assert nb >= 1


# ---------------------------------------------------------------------------
# process_match (with a stub detector)
# ---------------------------------------------------------------------------


class StubDetector:
    """Stub detector: returns a fixed player bbox on every frame.

    Tests the full pipeline without YOLO or GPU.
    """

    def __init__(self, bbox: tuple[float, float, float, float] = (140, 80, 180, 200)) -> None:
        self.bbox = bbox
        self.batch_sizes: list[int] = []

    def detect(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        self.batch_sizes.append(len(frames))
        return [
            sv.Detections(
                xyxy=np.array([self.bbox], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([HANDBALL_CLASSES["players"]], dtype=int),
            )
            for _ in frames
        ]


def test_process_match_end_to_end(tmp_path: Path) -> None:
    """Full pipeline with a stub detector: CSV/Parquet stats and a consistent result."""
    source = tmp_path / "match.mp4"
    make_fake_video(source, nb_frames=40, fps=25.0, width=320, height=240)
    output = tmp_path / "out"

    res = process_match(
        video_path=source,
        court_mapping=_VIDEO_MAPPING,
        output_dir=output,
        detector=StubDetector(),
        subsample=2,
        with_radar_video=False,  # speeds the test up
        with_clips=False,
    )

    assert isinstance(res, PipelineResult)
    assert res.source_video == source
    assert res.radar_video is None
    assert (output / "player_stats.csv").exists()
    assert (output / "player_stats.parquet").exists()
    assert res.metadata["subsample"] == 2
    assert res.metadata["nb_frames_total"] == 40
    assert res.metadata["nb_trackers_total"] >= 1


def test_process_match_with_radar_video(tmp_path: Path) -> None:
    """Pipeline with the radar video: the mp4 file is written."""
    source = tmp_path / "match.mp4"
    make_fake_video(source, nb_frames=10, fps=25.0, width=320, height=240)

    res = process_match(
        video_path=source,
        court_mapping=_VIDEO_MAPPING,
        output_dir=tmp_path / "out",
        detector=StubDetector(),
        subsample=1,
        with_radar_video=True,
        with_clips=False,
    )
    assert res.radar_video is not None
    assert res.radar_video.exists()


def test_process_match_missing_video(tmp_path: Path) -> None:
    """Missing source video: FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        process_match(
            video_path=tmp_path / "missing.mp4",
            court_mapping=_TEST_MAPPING,
            output_dir=tmp_path / "out",
            detector=StubDetector(),
        )


def test_process_match_detects_in_batches(tmp_path: Path) -> None:
    """The pipeline calls detect with batches of batch_size frames."""
    source = tmp_path / "match.mp4"
    make_fake_video(source, nb_frames=40, fps=25.0, width=320, height=240)
    stub = StubDetector()
    process_match(
        video_path=source,
        court_mapping=_VIDEO_MAPPING,
        output_dir=tmp_path / "out",
        detector=stub,
        subsample=2,
        with_radar_video=False,
        with_clips=False,
        batch_size=8,
    )
    # 40 frames / subsample 2 = 20 sampled frames -> batches of 8 + 8 + 4.
    assert stub.batch_sizes == [8, 8, 4]
