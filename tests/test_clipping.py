"""Unit tests for handball_ai.clipping."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from handball_ai.clipping import (
    Action,
    _count_players_per_frame,
    cut_video_clips,
    detect_actions,
)
from tests.conftest import make_fake_video

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _tracked_detections(tracker_ids: list[int], class_ids: list[int]) -> sv.Detections:
    """Build synthetic sv.Detections with tracker_id and class_id."""
    n = len(tracker_ids)
    assert n == len(class_ids)
    if n == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.zeros((n, 4), dtype=np.float32),
        confidence=np.ones(n, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
    )


def _build_sequence(profile: list[tuple[int, int]]) -> dict[int, sv.Detections]:
    """Build a frame -> Detections sequence from (frame_idx, nb_players).

    Always reuses tracker_id 1..N (all of class 0 = player).
    """
    return {fi: _tracked_detections(list(range(1, n + 1)), [0] * n) for fi, n in profile}


PLAYERS = {tid: 0 for tid in range(1, 8)}


# ---------------------------------------------------------------------------
# Players per frame
# ---------------------------------------------------------------------------


def test_count_players_ignores_non_player_classes() -> None:
    """Only trackers whose stabilized class == player_class_id are counted."""
    dets = _tracked_detections(tracker_ids=[1, 2, 3, 4], class_ids=[0, 0, 1, 2])
    final_class = {1: 0, 2: 0, 3: 1, 4: 2}  # 2 players, 1 gk, 1 ref
    assert _count_players_per_frame({0: dets}, final_class, player_class_id=0) == {0: 2}


def test_count_players_empty_frame() -> None:
    """A frame without detection = 0 players."""
    assert _count_players_per_frame({5: sv.Detections.empty()}, {}, player_class_id=0) == {5: 0}


# ---------------------------------------------------------------------------
# Action detection
# ---------------------------------------------------------------------------


def test_detect_actions_clean_sequence() -> None:
    """Sequence: 50 empty frames, 100 frames with 7 players, 50 empty -> 1 action."""
    fps = 25.0
    profile = [(i, 0) for i in range(50)]
    profile += [(i, 7) for i in range(50, 150)]  # 4 s action
    profile += [(i, 0) for i in range(150, 200)]

    actions = detect_actions(
        _build_sequence(profile),
        PLAYERS,
        player_class_id=0,
        fps=fps,
        start_threshold=6,
        end_threshold=4,
        min_action_duration_s=3.0,
        max_pause_s=3.0,
    )
    assert len(actions) == 1
    a = actions[0]
    assert a.start_frame == 50
    # Last frame with >= end_threshold players = 149
    assert a.end_frame == 149
    assert math.isclose(a.duration_s, (149 - 50) / fps, abs_tol=1e-6)
    assert a.duration_s >= 3.0
    assert math.isclose(a.mean_nb_players, 7.0, abs_tol=1e-6)
    assert a.attacking_team is None


def test_detect_actions_min_duration_filter() -> None:
    """A short action (0.8 s) is ignored with min_action_duration_s=3.0."""
    profile = [(i, 0) for i in range(20)]
    profile += [(i, 7) for i in range(20, 40)]  # ~0.8 s, too short
    profile += [(i, 0) for i in range(40, 200)]

    actions = detect_actions(
        _build_sequence(profile), PLAYERS, player_class_id=0, fps=25.0,
        min_action_duration_s=3.0,
    )
    assert actions == []


def test_detect_actions_short_pause_inside_action() -> None:
    """A pause shorter than max_pause_s does not break the action."""
    profile = [(i, 0) for i in range(25)]
    profile += [(i, 7) for i in range(25, 100)]   # 3 s of action
    profile += [(i, 2) for i in range(100, 120)]  # ~0.8 s pause (< 3 s)
    profile += [(i, 7) for i in range(120, 200)]  # 3.2 s restart
    profile += [(i, 0) for i in range(200, 250)]

    actions = detect_actions(
        _build_sequence(profile), PLAYERS, player_class_id=0, fps=25.0,
        min_action_duration_s=3.0, max_pause_s=3.0,
    )
    # A single action covering everything
    assert len(actions) == 1
    assert actions[0].start_frame == 25
    assert actions[0].end_frame == 199


def test_detect_actions_two_distinct_actions() -> None:
    """Two 4 s peaks separated by a pause longer than 3 s: 2 actions."""
    profile = [(i, 0) for i in range(20)]
    profile += [(i, 7) for i in range(20, 120)]   # action 1: 4 s
    profile += [(i, 0) for i in range(120, 240)]  # long pause: 4.8 s
    profile += [(i, 7) for i in range(240, 360)]  # action 2: 4.8 s
    profile += [(i, 0) for i in range(360, 400)]

    actions = detect_actions(
        _build_sequence(profile), PLAYERS, player_class_id=0, fps=25.0,
        min_action_duration_s=3.0, max_pause_s=3.0,
    )
    assert [a.start_frame for a in actions] == [20, 240]


def test_detect_actions_running_until_the_end() -> None:
    """An action still running at the end of the clip is kept."""
    profile = [(i, 0) for i in range(25)]
    profile += [(i, 7) for i in range(25, 200)]  # 7 s, never closed

    actions = detect_actions(
        _build_sequence(profile), PLAYERS, player_class_id=0, fps=25.0,
        min_action_duration_s=3.0,
    )
    assert len(actions) == 1
    assert actions[0].start_frame == 25
    assert actions[0].end_frame == 199


def test_detect_actions_empty() -> None:
    """No detection: no action."""
    assert detect_actions({}, {}, player_class_id=0, fps=25.0) == []


def test_detect_actions_invalid_fps() -> None:
    """fps <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="fps"):
        detect_actions({}, {}, player_class_id=0, fps=0.0)


def test_detect_actions_inconsistent_thresholds() -> None:
    """end_threshold > start_threshold raises ValueError."""
    with pytest.raises(ValueError, match="threshold"):
        detect_actions(
            {}, {}, player_class_id=0, fps=25.0, start_threshold=4, end_threshold=6,
        )


# ---------------------------------------------------------------------------
# Video clips (needs ffmpeg + cv2 to write the source)
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_cut_video_clips_end_to_end(tmp_path: Path) -> None:
    """Real ffmpeg cut: files are created with the expected names."""
    source = tmp_path / "source.mp4"
    make_fake_video(source, nb_frames=100, fps=25.0, width=160, height=120)  # 4 s

    actions = [
        Action(start_frame=0, end_frame=25, duration_s=1.0,
               attacking_team=None, mean_nb_players=6.0),
        Action(start_frame=50, end_frame=75, duration_s=1.0,
               attacking_team=None, mean_nb_players=7.0),
    ]
    paths = cut_video_clips(source, actions, tmp_path / "clips", prefix="test")

    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    assert paths[0].name == "test_000_1.0s.mp4"
    assert paths[1].name == "test_001_1.0s.mp4"


@needs_ffmpeg
def test_cut_video_clips_missing_source(tmp_path: Path) -> None:
    """Missing source: FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        cut_video_clips(tmp_path / "missing.mp4", [Action(0, 25, 1.0, None, 6.0)], tmp_path / "out")


@needs_ffmpeg
def test_cut_video_clips_empty_list(tmp_path: Path) -> None:
    """No action: no clip, but the folder is created."""
    source = tmp_path / "src.mp4"
    make_fake_video(source, nb_frames=50, fps=25.0, width=160, height=120)
    output = tmp_path / "out"
    assert cut_video_clips(source, [], output) == []
    assert output.exists()
